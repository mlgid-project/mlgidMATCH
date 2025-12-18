import numpy as np
from typing import Tuple, Union
import torch
from pygidsim.giwaxs_sim import GIWAXS


def limit_q(q_2d: np.ndarray,
            intensity: np.ndarray,
            q_range: Tuple[float, float],
            ):
    """
        Exclude peaks outside the q_range

        Parameters
        ----------
            q_2d : np.ndarray, shape - (peaks_num, 2)
                peak positions in 2D q-space — q_xy, q_z
            intensity : np.ndarray, shape - (peaks_num,)
            q_range : Tuple[float, float], (q_xy and q_z max)

        Return
        -------
            limited_q :  np.ndarray, shape - (peaks_num, 2)
                peak positions in 2d q-space — q_xy, q_z
            limited_int :  np.ndarray, shape - (peaks_num,)
                intensities after limitation
    """

    # excludes peaks outside the q_range
    # elements where q_z >=0 and q_xy <= limit and q_z <= limit
    q_2d[np.abs(q_2d) < 1e-4] = 0
    mask = ((q_2d[:, 1] >= 0) &
            (q_2d[:, 0] <= q_range[0]) &
            (q_2d[:, 1] <= q_range[1]))  # - (peaks_num, )

    limited_q = q_2d[mask]
    limited_int = intensity[mask]

    return limited_q, limited_int


def limit_int(q_2d: np.ndarray,  # (peaks_num, 2)
              intensity: np.ndarray,  # (peaks_num,)
              top_peaks: int,
              ):
    """
        Return peaks with top intensities

        Parameters
        ----------
            q_2d : np.ndarray, shape (peaks_num, 2)
                peak positions in 2d q-space — q_xy, q_z
            intensity : np.ndarray, shape - (peaks_num,)
            top_peaks : int
                how many brightest peaks to take

        Return
        -------
            limited_q :  np.ndarray, shape (peaks_num, 2)
                peak positions in 2d q-space — q_xy, q_z
            limited_int :  np.ndarray, shape - (peaks_num,)
                intensities after limitation
    """
    sort_arg = np.argsort(intensity)[::-1][:top_peaks]
    limited_q = q_2d[sort_arg]
    limited_int = intensity[sort_arg]

    return limited_q, limited_int


def unique(q_2d: np.ndarray,
           intensity: np.ndarray,
           ):
    """
        Return only unique peak positions

        Parameters
        ----------
            q_2d : np.ndarray, shape (peaks_num, 2)
                peak positions in 2d q-space — q_xy, q_z
            intensity : np.ndarray, shape (peaks_num,)

        Return
        -------
            q_2d_unique : np.ndarray, shape (peaks_num, 2)
                peak positions in 2d q-space — q_xy, q_z
            int_unique : np.ndarray, shape (peaks_num,)
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
    k = 2 * np.pi / wavelength

    condition_inME = (k - abs(q_2d[:, 0])) ** 2 > (k ** 2 - q_2d[:, 1] ** 2)  # condition if peaks are in Missing Edge
    L = np.empty_like(intensities)
    L[condition_inME] = 2 * k / np.linalg.norm(q_2d, axis=1)[condition_inME] ** 2
    L[~condition_inME] = 1 / q_2d[:, 0][~condition_inME]

    return L * intensities
