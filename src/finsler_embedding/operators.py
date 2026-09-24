import torch
from scipy.special import factorial
from math import gamma as _gamma
from tqdm import tqdm
from torchvision.ops import MLP


def gaussian_kernel(edges, dst_edges, eps: float, N: int, m: int = 2):
    dst_mat = torch.zeros((N, N), device=dst_edges.device, dtype=dst_edges.dtype)
    dst_mat[edges[:, 0], edges[:, 1]] = dst_edges

    W = torch.zeros_like(dst_mat)
    W[edges[:, 0], edges[:, 1]] = torch.exp(-dst_mat[edges[:, 0], edges[:, 1]] ** 2 / eps**2)
    W.fill_diagonal_(0)

    mu_0 = 1 / 2 * torch.lgamma(torch.tensor(m) / 2).exp()
    mu_1 = 1 / 2 * torch.lgamma(torch.tensor(m + 1) / 2).exp()

    return W, mu_0, mu_1


def apply_gaussian_kernel(edges, dst_edges, eps: float, N: int):
    dst_mat = torch.zeros((N, N), device=dst_edges.device, dtype=dst_edges.dtype)
    dst_mat[edges[:, 0], edges[:, 1]] = dst_edges

    W = torch.zeros_like(dst_mat)
    W[edges[:, 0], edges[:, 1]] = torch.exp(-dst_mat[edges[:, 0], edges[:, 1]] ** 2 / eps**2)
    W.fill_diagonal_(0)

    return W


def laplacian_kernel(
    edges: torch.Tensor, dst_edges: torch.Tensor, eps: float, N: int, m: int = 2
):
    dst_mat = torch.zeros((N, N), device=dst_edges.device, dtype=dst_edges.dtype)
    dst_mat[edges[:, 0], edges[:, 1]] = dst_edges

    W = torch.zeros_like(dst_mat)
    W[edges[:, 0], edges[:, 1]] = torch.exp(-dst_mat[edges[:, 0], edges[:, 1]] / eps)
    W.fill_diagonal_(0)

    mu_0 = factorial(m - 1)
    mu_1 = factorial(m)
    return W, mu_0, mu_1


def kernel_moments(m: int, kernel: str = "gaussian"):
    """
    Compute the moments of the kernel function K(r).
    If the kernel is Gaussian, K(r) = exp(-r^2 / 2), the moments are given by:
        mu_n = 1/2 * Gamma((m+n)/2)
    If the kernel is exponential, K(r) = exp(-r), the moments are given by:
        mu_n = Gamma(m+n)

    Parameters:
    ----------
    m : int
        The dimension of the space.
    kernel : str
        The type of kernel to use. Supported values are "gaussian" and "exponential".

    Returns:
    -------
    mu_0, mu_1, mu_2 : tuple of floats
        The moments of the kernel function K(r) for n = 0, 1, 2, respectively.
    """
    if kernel == "gaussian":
        return tuple(1 / 2 * _gamma((m + n) / 2) for n in (0, 1, 2))
    if kernel == "exponential":
        return tuple(_gamma(m + n) for n in (0, 1, 2))
    raise ValueError(f"unknown kernel {kernel!r}")


def get_Q_inv_theta(W: torch.Tensor, theta: int = 1):
    D = torch.sum(W, dim=1)
    D_prime = torch.sum(W, dim=0)
    Q = (D + D_prime) / 2
    Q_theta = Q**theta
    Q_inv_theta = 1 / Q_theta
    Q_inv_theta = torch.diag_embed(Q_inv_theta)
    return Q_inv_theta


def get_W_thetas(W: torch.Tensor, Q_inv_theta: torch.Tensor):
    W_theta = Q_inv_theta @ W @ Q_inv_theta
    W_theta_s = (W_theta + W_theta.T) / 2
    W_theta_a = (W_theta - W_theta.T) / 2
    return W_theta_s, W_theta_a


def construct_P_thetas(W_theta_s: torch.Tensor, W_theta_a: torch.Tensor):
    D_theta_s = torch.sum(W_theta_s, dim=1)
    D_theta_a = torch.sum(W_theta_a, dim=1)
    D_theta_a = torch.diag_embed(D_theta_a)
    D_theta_s_inv = 1 / D_theta_s

    Id = torch.eye(W_theta_s.shape[0]).to(W_theta_s.device).to(W_theta_s.dtype)
    P_theta_s = D_theta_s_inv[..., None] * W_theta_s - Id
    P_theta_a = D_theta_s_inv[..., None] * (W_theta_a - D_theta_a)
    return P_theta_s, P_theta_a


def construct_operators(W: torch.Tensor, eps: float, theta: int = 1):
    Q_inv_theta = get_Q_inv_theta(W, theta)
    W_theta_s, W_theta_a = get_W_thetas(W, Q_inv_theta)
    P_theta_s, P_theta_a = construct_P_thetas(W_theta_s, W_theta_a)
    L_theta_s = 1 / eps**2 * P_theta_s
    L_theta_a = 1 / eps * P_theta_a
    return L_theta_s, L_theta_a


def carre_du_champ(L_s: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Constructs the Carré du Champ operator Gamma_i^{kl} for the functions psi_k, psi_l with respect to the operator L^s.

        Gamma_i^{kl} = 1/2 [ L^s(psi_k psi_l) - psi_k L^s psi_l - psi_l L^s psi_k ]_i

    Parameters:
    ----------
    L_s : torch.Tensor (N, N)
        The symmetric operator L^s.
    psi: torch.Tensor (N, K)
        The low-dimensional embedding of the data points.

    Returns:
    -------
    Gamma : torch.Tensor (N, K, K)
        The Carré du Champ operator evaluated at each point.

    Remarks:
        It is a cometric, meaning it should be contracted with the pseudoinverse of the metric tensor, not the metric tensor itself.
    """
    N, K = psi.shape
    L_psi = L_s @ psi  # (N, K), computed once
    prod = psi[:, :, None] * psi[:, None, :]  # (N, K, K)
    L_prod = torch.einsum("ij,jkl->ikl", L_s, prod)  # L^s applied to psi_k psi_l
    cross = psi[:, :, None] * L_psi[:, None, :]
    Gamma = 0.5 * (L_prod - cross - cross.transpose(1, 2))
    return 0.5 * (Gamma + Gamma.transpose(1, 2))  # symmetrize


def truncated_pinv(A: torch.Tensor, rank: int, rcond: float = 1e-10):
    """
    Pseudo-inverse of a batch of symmetric PSD matrices, truncated to
    `rank` (= m).

    Gamma has rank m by construction; a default-tolerance pinv is
    numerically unreliable when K > m.

    Parameters:
    ----------
    A : torch.Tensor (..., N, N)
        A batch of symmetric positive semi-definite matrices.
    rank : int
        The rank to which the pseudo-inverse should be truncated.
    rcond : float
        Relative condition number for small eigenvalues. Eigenvalues smaller than rcond * max(eigenvalue) are treated as zero.

    Returns:
    -------
    pinv_A : torch.Tensor (..., N, N)
        The truncated pseudo-inverse of A.
        It verifies that A @ pinv_A @ A = A and pinv_A @ A @ pinv_A = pinv_A, and has rank at most `rank`.
    """
    evals, evecs = torch.linalg.eigh(A)  # ascending
    evals = evals[..., -rank:]
    evecs = evecs[..., -rank:]
    keep = evals > rcond * evals[..., -1:].clamp_min(0).abs()
    inv = torch.where(keep, 1.0 / evals.clamp_min(rcond), torch.zeros_like(evals))
    return evecs @ torch.diag_embed(inv) @ evecs.transpose(-1, -2)


def randers_constants(m: int, kernel: str = "gaussian"):
    """c_2, C and kappa_K"""
    mu0, mu1, mu2 = kernel_moments(m, kernel)
    c_2 = mu2 / (2.0 * m * mu0)  # [FIX-F] was 4*mu2/(m*mu0)
    C = (m + 1) * mu1 / (m * mu0)
    return c_2, C, c_2 / C**2


def leading_eigenvectors(W_s: torch.Tensor, n_components: int) -> torch.Tensor:
    """
    Leading non-trivial eigenvectors of the random-walk operator D^{-1} W_s (equivalently, the
    eigenvectors of L^s = (D^{-1} W_s - I) / eps^2 with the eigenvalues closest to 0), computed from
    the symmetric matrix D^{-1/2} W_s D^{-1/2}. The constant eigenvector is dropped.
    """
    d_inv_sqrt = W_s.sum(dim=1).rsqrt()
    S = d_inv_sqrt[:, None] * W_s * d_inv_sqrt[None, :]
    _, evecs = torch.linalg.eigh(S)  # ascending eigenvalues
    evecs = evecs[:, -(n_components + 1) : -1].flip(1)  # largest ones, without the trivial one
    return d_inv_sqrt[:, None] * evecs


def randers_approximation(
    W: torch.Tensor, epsilon: float, X_low=None, m: int = 2, kernel_type: str = "gaussian"
):

    Q_inv_theta = get_Q_inv_theta(W, theta=1)
    W_theta_s, W_theta_a = get_W_thetas(W, Q_inv_theta)
    P_theta_s, P_theta_a = construct_P_thetas(W_theta_s, W_theta_a)
    L_theta_s = 1 / epsilon**2 * P_theta_s
    L_theta_a = 1 / epsilon * P_theta_a

    c_2, C, kappa = randers_constants(m, kernel=kernel_type)

    if X_low is None:
        X_low = leading_eigenvectors(W_theta_s, m)

    Gamma = carre_du_champ(L_theta_s, X_low)
    V = L_theta_a @ X_low
    Gamma_pinv = truncated_pinv(Gamma, rank=m)

    c_norm_sq = kappa * torch.einsum("nij,ni,nj->n", Gamma_pinv, V, V)
    admissible = c_norm_sq < 1.0 / (m + 3)

    c_tilde = (kappa**0.5) * V
    H_tilde_inv = Gamma - (m + 2) * kappa * torch.einsum("ni,nj->nij", V, V)
    H_tilde = truncated_pinv(H_tilde_inv, rank=m)
    lambda_tilde = 1.0 - torch.einsum("nij,ni,nj->n", H_tilde, c_tilde, c_tilde)

    return {
        "W": W,
        "L_theta_s": L_theta_s,
        "L_theta_a": L_theta_a,
        "Gamma": Gamma,
        "V": V,
        "Gamma_pinv": Gamma_pinv,
        "c_norm_sq": c_norm_sq,
        "admissible": admissible,
        "c_tilde": c_tilde,
        "H_tilde": H_tilde,
        "lambda_tilde": lambda_tilde,
        "c_2": c_2,
        "C": C,
        "kappa": kappa,
        "X_low": X_low,
    }


def interpolate_V(X_low: torch.Tensor, V: torch.Tensor, dim: int, n_epochs: int = 500):
    # Train a NN to interpolate V
    model = MLP(in_channels=dim, hidden_channels=[64, 64, dim])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_list = []
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        V_pred = model(X_low)
        loss = torch.nn.functional.mse_loss(V_pred, V)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")
        loss_list.append(loss.item())
    return model, loss_list
