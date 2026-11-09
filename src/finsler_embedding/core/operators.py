import torch


def get_Q_inv_theta(W: torch.Tensor, theta: int = 1):
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if not torch.isfinite(W).all() or (W < 0).any():
        raise ValueError("W must contain finite, nonnegative values")
    if not torch.isfinite(torch.as_tensor(theta)) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")
    D = torch.sum(W, dim=1)
    D_prime = torch.sum(W, dim=0)
    Q = (D + D_prime) / 2
    if (Q <= 0).any():
        raise ValueError("every node must have a positive symmetrized degree")
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
    if W_theta_s.shape != W_theta_a.shape or W_theta_s.ndim != 2:
        raise ValueError(
            "symmetric and antisymmetric kernels must be square matrices of equal shape"
        )
    D_theta_s = torch.sum(W_theta_s, dim=1)
    if (D_theta_s <= 0).any() or not torch.isfinite(D_theta_s).all():
        raise ValueError("every node must have a positive symmetric degree")
    D_theta_a = torch.sum(W_theta_a, dim=1)
    D_theta_a = torch.diag_embed(D_theta_a)
    D_theta_s_inv = 1 / D_theta_s

    Id = torch.eye(W_theta_s.shape[0]).to(W_theta_s.device).to(W_theta_s.dtype)
    P_theta_s = D_theta_s_inv[..., None] * W_theta_s - Id
    P_theta_a = D_theta_s_inv[..., None] * (W_theta_a - D_theta_a)
    return P_theta_s, P_theta_a


def construct_operators(W: torch.Tensor, eps: float, theta: int = 1):
    if not torch.isfinite(torch.as_tensor(eps)) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive")
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
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must contain square matrices")
    if not isinstance(rank, int) or not 1 <= rank <= A.shape[-1]:
        raise ValueError("rank must be an integer between 1 and the matrix dimension")
    if not torch.isfinite(torch.as_tensor(rcond)) or rcond < 0:
        raise ValueError("rcond must be finite and nonnegative")
    evals, evecs = torch.linalg.eigh(A)  # ascending
    evals = evals[..., -rank:]
    evecs = evecs[..., -rank:]
    keep = evals > rcond * evals[..., -1:].clamp_min(0).abs()
    inv = torch.where(keep, 1.0 / evals.clamp_min(rcond), torch.zeros_like(evals))
    return evecs @ torch.diag_embed(inv) @ evecs.transpose(-1, -2)


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
    if X_low is None:
        q_inv_theta = get_Q_inv_theta(W, theta=1)
        W_theta_s, _ = get_W_thetas(W, q_inv_theta)
        X_low = leading_eigenvectors(W_theta_s, m)
    from .randers import reconstruct_randers

    return reconstruct_randers(W, epsilon, X_low, m, kernel_type)

