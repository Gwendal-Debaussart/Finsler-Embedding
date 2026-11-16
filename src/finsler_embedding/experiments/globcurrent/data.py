"""GlobCurrent surface currents as a directed dataset (Zermelo navigation).

The data are the daily Eulerian surface currents of the Agulhas region (GlobCurrent, 0.25 degree grid,
81 x 41 nodes, 15E-35E and 40S-30S), as redistributed in the example data of Parcels:
https://github.com/Parcels-code/parcels-data/tree/main/data/GlobCurrent_example_data

The file contains a velocity field, not directed distances, so the directed distances are modelled by
Zermelo navigation: a vessel sails at constant speed ``s`` relative to the water, in a current ``W(x)``
with ``|W| < s``. The time to go from ``x`` to ``x + v`` is the Randers metric

    F(x, v) = (sqrt(lambda |v|^2 + <w, v>^2) - <w, v>) / lambda,    w = W / s,  lambda = 1 - |w|^2,

whose unit ball is the unit disc centred at ``w``. It is expressed in "still-water kilometres": ``F(x, v) = |v|``
without current. The ground truth of the method is therefore known in closed form: the centroid of the
unit ball is ``c = w`` and its Binet-Legendre cometric is ``g_BL^{-1} = I + 4 w w^T``, so that
``||c||^2_{g_BL} = |w|^2 / (1 + 4 |w|^2)``.

The points live on the sphere (km, ambient R^3) and the tangent frame at a node is (east, north), in which
the current has its components ``(u, v)``. Unlike the Randers drift of the synthetic datasets, the current
has vorticity (eddies and the Agulhas retroflection), so the asymmetry of the distances is not the
gradient of a potential.
"""

from pathlib import Path
import urllib.request

import numpy as np
import torch
from scipy.io import netcdf_file
from scipy.ndimage import distance_transform_edt

EARTH_RADIUS_KM = 6371.0

# Pinned commit of the data repository, so that the downloaded file does not change under the experiments
PARCELS_DATA_COMMIT = "53d2b60e87b4e58a8a6b3f3c2d44d841b3c71c8d"
URL = (
    "https://raw.githubusercontent.com/Parcels-code/parcels-data/{commit}/"
    "data/GlobCurrent_example_data/{date}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
)
DEFAULT_DATE = "20020101"  # first day of the example data: 2002-01-01 to 2002-12-31 (and 2003-01-01)
DEFAULT_CACHE = Path(__file__).resolve().parents[4] / "data" / "globcurrent"  # <repository>/data/globcurrent, whatever the working directory


def download(date: str = DEFAULT_DATE, cache: Path = DEFAULT_CACHE) -> Path:
    """Path of the file of one day (``YYYYMMDD``), downloaded (about 55 kB) into ``cache`` if it is not there."""
    url = URL.format(commit=PARCELS_DATA_COMMIT, date=date)
    path = Path(cache) / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as response:
            partial.write_bytes(response.read())
        partial.rename(path)
    return path


def load_currents(date: str = DEFAULT_DATE, cache: Path = DEFAULT_CACHE) -> dict[str, np.ndarray]:
    """Currents of one day: ``lon`` (nx,), ``lat`` (ny,) in degrees, ``u`` and ``v`` (ny, nx) in m/s (NaN on land)."""
    with netcdf_file(download(date, cache), mmap=False) as f:
        arrays = {
            "lon": f.variables["lon"][:],
            "lat": f.variables["lat"][:],
            "u": f.variables["eastward_eulerian_current_velocity"][0],
            "v": f.variables["northward_eulerian_current_velocity"][0],
        }
    return {name: np.array(value, dtype=np.float64) for name, value in arrays.items()}


def tangent_frame(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
    """Unit vectors (east, north) (N, 3, 2) of the sphere at longitude ``lon`` and latitude ``lat`` (radians)."""
    east = torch.stack([-lon.sin(), lon.cos(), torch.zeros_like(lon)], dim=1)
    north = torch.stack([-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()], dim=1)
    return torch.stack([east, north], dim=2)


def to_sphere(lon: torch.Tensor, lat: torch.Tensor, radius: float = EARTH_RADIUS_KM) -> torch.Tensor:
    """Cartesian coordinates (N, 3), in km, of longitude / latitude in radians."""
    return radius * torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=1)


class ZermeloMetric(torch.nn.Module):
    """Randers metric of Zermelo navigation in the surface currents, evaluated at points of the sphere in R^3.

    ``F(x, v)`` (km) is the length of the displacement ``v`` (ambient vector, km) at ``x``, in still-water
    kilometres. The current is the bilinear interpolation of the grid (land nodes are filled with the nearest
    ocean value, only to interpolate in the cells next to the coast), in units of the ship speed.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, ship_speed: float):
        super().__init__()
        land = np.isnan(u)
        nearest = distance_transform_edt(land, return_distances=False, return_indices=True)
        field = np.stack([u[tuple(nearest)], v[tuple(nearest)]]) / ship_speed  # (2, ny, nx), in units of s
        self.register_buffer("field", torch.as_tensor(field, dtype=torch.float32))
        self.lon0, self.dlon, self.nx = float(lon[0]), float(lon[1] - lon[0]), len(lon)
        self.lat0, self.dlat, self.ny = float(lat[0]), float(lat[1] - lat[0]), len(lat)

    def current(self, x: torch.Tensor) -> torch.Tensor:
        """Current ``w`` (B, 3) as an ambient vector, in units of the ship speed, at the points ``x`` (B, 3)."""
        lon, lat = torch.atan2(x[:, 1], x[:, 0]), torch.asin((x[:, 2] / x.norm(dim=1)).clamp(-1, 1))
        fx = ((lon.rad2deg() - self.lon0) / self.dlon).clamp(0, self.nx - 1)
        fy = ((lat.rad2deg() - self.lat0) / self.dlat).clamp(0, self.ny - 1)
        ix, iy = fx.floor().long().clamp(max=self.nx - 2), fy.floor().long().clamp(max=self.ny - 2)
        tx, ty = fx - ix, fy - iy
        f = self.field.to(x.dtype)
        w = (
            f[:, iy, ix] * ((1 - tx) * (1 - ty))
            + f[:, iy, ix + 1] * (tx * (1 - ty))
            + f[:, iy + 1, ix] * ((1 - tx) * ty)
            + f[:, iy + 1, ix + 1] * (tx * ty)
        )  # (2, B): east and north components
        frame = tangent_frame(lon, lat)
        return torch.einsum("bdk,kb->bd", frame, w)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = self.current(x)
        wv = (w * v).sum(dim=1)
        lam = (1 - (w * w).sum(dim=1)).clamp_min(1e-6)
        return ((lam * (v * v).sum(dim=1) + wv**2).sqrt() - wv) / lam


def make_globcurrent(N: int, beta: float, date: str = DEFAULT_DATE, cache: Path = DEFAULT_CACHE) -> dict:
    """Directed dataset of the surface currents: ``N`` ocean nodes drawn from the grid (all of them if ``N`` is larger).

    ``beta`` is the strength of the asymmetry: the ship speed is set so that the strongest current of the day
    is ``beta`` times the speed of the ship (``|w| <= beta < 1``, the analogue of ``||b||`` of the synthetic
    Randers metrics). The returned dictionary is the one of the other datasets of ``legacy.evaluation``;
    ``labels`` is the speed of the current in m/s, ``w`` the current in units of the ship speed in the (east,
    north) frame, and ``ship_speed`` is in m/s.
    """
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0, 1)")
    currents = load_currents(date, cache)
    lon_grid, lat_grid = np.meshgrid(currents["lon"], currents["lat"])
    ocean = ~np.isnan(currents["u"])
    speed = np.hypot(currents["u"], currents["v"])
    ship_speed = float(np.nanmax(speed) / beta)
    metric = ZermeloMetric(currents["lon"], currents["lat"], currents["u"], currents["v"], ship_speed)

    nodes = torch.nonzero(torch.as_tensor(ocean.ravel())).squeeze(1)
    if N < len(nodes):
        nodes = nodes[torch.randperm(len(nodes))[:N]]
    lon = torch.as_tensor(np.deg2rad(lon_grid.ravel()[nodes.numpy()]), dtype=torch.float32)
    lat = torch.as_tensor(np.deg2rad(lat_grid.ravel()[nodes.numpy()]), dtype=torch.float32)
    X = to_sphere(lon, lat)
    P = tangent_frame(lon, lat)
    w = torch.as_tensor(
        np.stack([currents["u"].ravel()[nodes.numpy()], currents["v"].ravel()[nodes.numpy()]], axis=1) / ship_speed,
        dtype=torch.float32,
    )
    # Unit ball of F: the unit disc centred at w, in the tangent plane
    truth = {"c": w, "g_bl_inv": torch.eye(2) + 4 * torch.einsum("ni,nj->nij", w, w)}
    return {
        "X_graph": X, "X_chart": X, "P": P, "F": metric, "truth": truth, "labels": speed.ravel()[nodes.numpy()],
        "w": w, "ship_speed": ship_speed, "lon": lon, "lat": lat,
    }
