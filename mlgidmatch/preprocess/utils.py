import numpy as np
from pygidsim.giwaxs_sim import GIWAXS
from typing import Tuple


def limit_int(q_2d: np.ndarray,  # (peaks_num, 2)
              intensity: np.ndarray,  # (peaks_num,)
              top_peaks: int,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return peaks with top intensities.

    Parameters
    ----------
    q_2d : np.ndarray
        Shape (peaks_num, 2). Peak positions in 2D q-space (q_xy, q_z).
    intensity : np.ndarray
        Shape (peaks_num,). Intensities corresponding to q_2d peaks.
    top_peaks : int
        Number of brightest peaks to select.

    Returns
    -------
    limited_q :  np.ndarray
        Shape (peaks_num, 2). Peak positions in 2D q-space after limitation.
    limited_int :  np.ndarray
        Shape (peaks_num,). Intensities after limitation.
    """
    sort_arg = np.argsort(intensity)[::-1][:top_peaks]
    limited_q = q_2d[sort_arg]
    limited_int = intensity[sort_arg]

    return limited_q, limited_int


def unique(q_2d: np.ndarray,
           intensity: np.ndarray,
           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return only unique peak positions.

    Parameters
    ----------
    q_2d : np.ndarray
        Shape (peaks_num, 2). Peak positions in 2D q-space (q_xy, q_z).
    intensity : np.ndarray
        Shape (peaks_num,). Intensities corresponding to q_2d peaks.

    Returns
    -------
    q_2d_unique : np.ndarray
        Shape (peaks_num, 2). Unique peak positions in 2D q-space.
    int_unique : np.ndarray
        Shape (peaks_num,). Intensities corresponding to unique q_2d positions.
    """
    clusters = GIWAXS.cluster_mask(q_2d.T, r=2e-2)
    counts_per_cluster = np.bincount(clusters)
    sum_x = np.bincount(clusters, weights=q_2d[:, 0])
    sum_y = np.bincount(clusters, weights=q_2d[:, 1])
    q_2d_unique = np.vstack((sum_x, sum_y)) / counts_per_cluster  # shape (2, peaks_num)
    int_unique = np.bincount(clusters, weights=intensity)
    # all_indices, indices_sum = GIWAXS.cluster_mask(q_2d.T, r=2e-2)
    # q_2d_unique = q_2d[all_indices]
    # int_unique = np.bincount(indices_sum, weights=intensity)
    q_2d_unique[np.abs(q_2d_unique) < 1e-4] = 0
    assert q_2d_unique.shape[1] == len(
        int_unique
    ), f"q_2d len: {q_2d_unique.shape[1]}, intensity len: {len(int_unique)}"

    return q_2d_unique.T, int_unique


def lorentz_correction_2d(q_2d: np.ndarray,  # (peaks_num, 2)
                          intensities: np.ndarray,  # (peaks_num,)
                          wavelength: float = 12_398 / 18_000,  # wavelength, Angstrom
                          ):
    """Apply Lorentz correction to intensities."""
    k = 2 * np.pi / wavelength

    condition_inME = (k - abs(q_2d[:, 0])) ** 2 > (k ** 2 - q_2d[:, 1] ** 2)  # condition if peaks are in Missing Edge
    L = np.empty_like(intensities)
    L[condition_inME] = 2 * k / np.linalg.norm(q_2d, axis=1)[condition_inME] ** 2
    L[~condition_inME] = 1 / q_2d[:, 0][~condition_inME]

    return L * intensities
