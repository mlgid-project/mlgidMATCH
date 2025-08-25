import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union
# from cif_matching.data.q_calculation.cif_preprocess import *
from scipy.optimize import linear_sum_assignment

from pygidsim.giwaxs_sim import GIWAXS
from mlgidmatch.cif_matching.utils import ExpConfig
from mlgidmatch.orient_matching.utils import SimConfig
from mlgidmatch.preprocess.rotate import rotate_vect


class Match_Orient():
    """
        A class to determine the crystallographic orientations of the experimental patterns.
        ...
        Attributes
        ----------
            config : TestExpConfig
                configuration of the test experiment
    """

    def __init__(self,
                 config: ExpConfig,
                 ):
        self.config = config

    def match(self,
              peaks_all: np.ndarray,  # (peaks_num_all, 2)
              intens_real_all: np.ndarray,  # (peaks_num_all,)
              probs: np.ndarray,  # (candidates_num,)
              q_range: Tuple[float, float],
              peaks_indices: np.ndarray,  # (peaks_num,)
              candidate_ind: np.ndarray,  # (candidates_num,)
              ):
        valid_indices = candidate_ind[np.where(probs >= 0.5)[0]]
        if len(valid_indices) == 0:
            return {}

        data_matched = self.test_sev_cifs(
            cif_indices_list=valid_indices,
            q_real_all=peaks_all,
            intens_real_all=intens_real_all,
            peaks_indices=peaks_indices,
            q_range=q_range,
        )

        (cif_indices_list, orients,
         q_sim_matched_list, indices_real_matched_all, indices_real_matched,
         metrics_sim, metrics_sim_150, metrics_real,
         metrics_sim_all, metrics_sim_150_all, metrics_real_add,) = data_matched
        data_matched = {
            str(key): {
                'cif': self.config.cif_class.pattern_3d.cif_list[cif_indices_list[key]],
                'orient': orients[key].astype(int),
                'probability': probs[cif_indices_list[key]],
                # 'q_sim_matched': q_sim_matched_list[key],
                # 'q_real_matched': q_real_matched_list[key],
                'indices_real_matched_all': indices_real_matched_all[key],
                '_indices_real_matched': indices_real_matched[key],
                # 'peaks_indices': peaks_indices[indices_real_matched[key]],
                # 'peaks_indices_input': peaks_indices,
                # 'metric_sim': metrics_sim[key],
                # 'metric_sim_150': metrics_sim_150[key],
                # 'metric_real': metrics_real[key],
                # 'metric_sim_all': metrics_sim_all[key],
                # 'metric_sim_150_all': metrics_sim_150_all[key],
                # 'metric_real_add': metrics_real_add[key],
            }
            for key in range(len(cif_indices_list))
        }
        return data_matched

    def test_sev_cifs(self,
                      cif_indices_list: np.ndarray,  # indices of cifs to test
                      q_real_all: np.ndarray,  # (peaks_num_all, ndim) - ndim=1 or 2
                      intens_real_all: np.ndarray,  # (peaks_num_all,)
                      peaks_indices: np.ndarray,  # shape (peaks_real_num,)
                      q_range: Tuple[float, float],
                      ):
        is_2d = (q_real_all.ndim == 2)
        cfg = self.config.cif_class
        need_sim_config = (is_2d and cfg.all_patterns_q2d is None) or (
                (not is_2d) and cfg.all_patterns_q1d is None)

        data_matched = [
            self.test_one_cif(
                q_real_all=q_real_all,
                intens_real_all=intens_real_all,
                peaks_indices=peaks_indices,
                q_sim_list=(
                    (cfg.all_patterns_q2d[idx] if cfg.all_patterns_q2d is not None else None)
                    if is_2d else
                    (cfg.all_patterns_q1d[idx] if cfg.all_patterns_q1d is not None else None)
                ),
                intens_sim_list=(
                    (cfg.all_patterns_int2d[idx] if cfg.all_patterns_int2d is not None else None)
                    if is_2d else
                    (cfg.all_patterns_int1d[idx] if cfg.all_patterns_int1d is not None else None)
                ),
                sim_config=SimConfig(
                    q_sim_3d=cfg.pattern_3d.q_3d[idx],
                    intens_sim_3d=cfg.pattern_3d.intensities[idx],
                    rec=cfg.pattern_3d.rec[idx],
                ) if need_sim_config
                else None,
                orientations=self.config.cif_class.pattern_3d.orientations[idx],
                q_range=q_range,
            ) for idx in cif_indices_list
        ]

        (orients,
         q_sim_matched_list, indices_real_matched_all, indices_real_matched,
         metrics_sim, metrics_sim_150, metrics_real,
         metrics_sim_all, metrics_sim_150_all, metrics_real_add,) = zip(*data_matched)

        max_real_metric = np.max(metrics_real)
        mask = np.where(
            (np.array(metrics_sim) >= 0)
            & (np.array(metrics_real) >= 0)
            & ((np.array(metrics_real) > max_real_metric * 0.5) |
               (np.array(metrics_sim_150_all) > 0.4)),
        )[0]

        q_sim_matched_list = [q_sim_matched_list[i] for i in mask]
        # q_real_matched_list = [q_real_all[indices_real_matched_all[i]] for i in mask]
        indices_real_matched_all = [indices_real_matched_all[i] for i in mask]
        indices_real_matched = [indices_real_matched[i] for i in mask]
        orients = [orients[i] for i in mask]
        metrics_sim = np.array(metrics_sim)[mask]
        metrics_sim_150 = np.array(metrics_sim_150)[mask]
        metrics_real = np.array(metrics_real)[mask]
        metrics_sim_all = np.array(metrics_sim_all)[mask]
        metrics_sim_150_all = np.array(metrics_sim_150_all)[mask]
        metrics_real_add = np.array(metrics_real_add)[mask]
        cif_indices_list = np.array(cif_indices_list)[mask]

        data_matched = (cif_indices_list, orients,
                        q_sim_matched_list, indices_real_matched_all, indices_real_matched,
                        metrics_sim, metrics_sim_150, metrics_real,
                        metrics_sim_all, metrics_sim_150_all, metrics_real_add,)
        return data_matched

    def test_one_cif(self,
                     q_real_all: np.ndarray,
                     intens_real_all: np.ndarray,
                     peaks_indices: np.ndarray,
                     q_sim_list: Union[List[np.ndarray], np.ndarray, None],
                     intens_sim_list: Union[List[np.ndarray], np.ndarray, None],
                     sim_config: Union[SimConfig, None],
                     orientations: Union[np.ndarray, None],
                     q_range: Tuple[float, float],
                     ):
        """
            Finds the best orientation of the structure that matches the experimental pattern,
             returns orientation, matched peaks and metric values.
            ...
            Parameters
            ----------
                q_real_all : np.ndarray
                    full peak list in the experimental pattern, shape (peaks_real_num_all, ndim) - ndim=1 or 2
                intens_real_all : np.ndarray
                    full peak intensities list in the experimental pattern, shape (peaks_real_num_all, )
                peaks_indices : np.ndarray
                    indices of the subset according to the full peak list, shape (peaks_real_num, )
                q_sim_list : Union[List[np.ndarray], np.ndarray, None]
                    list of the simulated peak arrays (own array for each orientation)
                    each array has shape (peaks_sim_num, 2)
                    OR just np.ndarray - peak list for powder3d pattern
                    if None (sim_list was not prepared in preprocessing) - calculate pattern manually
                intens_sim_list : Union[List[np.ndarray], np.ndarray, None]
                    list of the simulated intensity arrays (own array for each orientation)
                    each array has shape (peaks_sim_num,)
                    OR just np.ndarray - intensities list for powder3d pattern
                    if None (sim_list was not prepared in preprocessing) - calculate pattern manually
                sim_config : Union[SimConfig, None]
                    configuration with q_sim_3d, intens_sim_3d and rec to calculate sim_patterns manually
                orientations : Union[np.ndarray, None]
                    all possible orientation for the structure, shape (orient_num, 3)
                    for powder3d pattern = None
                q_range : Tuple[float, float]
                    Q value limits for xy and z axes
        """

        if q_real_all.ndim == 1:
            raise Exception
            # q_sim_matched, indices_real_matched, metric_sim, metric_sim_150 = self.get_match_metrics(
            #     q_real=q_real_all[peaks_indices],
            #     q_sim=q_sim_list,
            #     intensities_sim=intens_list_sim,
            #     q_range=q_range,
            # )
            # metric_real = self.calculate_real_metric(intens_real_all[peaks_indices], indices_real_matched)
            # if len(q_real_all) != len(peaks_indices):
            #     """ depth of the branch > 0"""
            #     q_sim_matched, indices_real_matched_all, metric_sim_all, metric_sim_150_all = self.get_match_metrics(
            #         q_real=q_real_all,
            #         q_sim=q_sim_list,
            #         intensities_sim=intens_list_sim,
            #         q_range=q_range,
            #     )
            #     metric_real_add = self.calculate_real_metric(intens_real_all, peaks_indices[indices_real_matched])
            # else:
            #     """ depth of the branch = 0"""
            #     indices_real_matched_all = indices_real_matched
            #     metric_sim_all = metric_sim
            #     metric_sim_150_all = metric_sim_150
            #     metric_real_add = metric_real
            # orientation = np.array([0, 0, 0])

        elif q_real_all.ndim == 2:
            or_opt, q_sim_matched, indices_real_matched, metric_sim, metric_sim_150, metric_real = (
                self.get_best_orientation(
                    q_real=q_real_all[peaks_indices],
                    intens_real=intens_real_all[peaks_indices],
                    q_sim_list=q_sim_list,
                    intens_sim_list=intens_sim_list,
                    sim_config=sim_config,
                    orientations=orientations,
                    q_range=q_range,
                )
            )

            if ((metric_sim >= 0.04) &
                    (metric_sim_150 >= 0.05) &
                    (metric_real >= 0.15) &
                    (len(q_sim_matched) > 3)):
                if len(q_real_all) != len(peaks_indices):
                    """ depth of the branch > 0"""
                    if q_sim_list is None:
                        # the pattern was not calculated in preprocessing
                        q_sim, intens_sim = self.calculate_pattern(q_3d=sim_config.q_sim_3d,
                                                                   rec=sim_config.rec,
                                                                   intensity=sim_config.intens_sim_3d,
                                                                   q_range=q_range,
                                                                   orientation=or_opt)
                    else:
                        idx_opt = np.where(np.all(orientations == or_opt, axis=1))[0].item()
                        q_sim, intens_sim = q_sim_list[idx_opt], intens_sim_list[idx_opt]
                    q_sim_matched, indices_real_matched_all, metric_sim_all, metric_sim_150_all = self.get_match_metrics(
                        q_real=q_real_all,
                        q_sim=q_sim,
                        intensities_sim=intens_sim,
                        q_range=q_range,
                    )
                    metric_real_add = self.calculate_real_metric(intens_real_all, peaks_indices[indices_real_matched])
                    sum_metrics = max(metric_sim, metric_sim_all, metric_sim_150_all) + metric_real
                    mult_metrics = max(metric_sim, metric_sim_all, metric_sim_150_all) * metric_real
                    max_metrics = max(metric_sim, metric_sim_all, metric_sim_150_all, metric_real)
                    if ((metric_sim_150_all < 0.08) or
                            (metric_real_add < 0.1) or
                            (max(metric_sim, metric_sim_all, metric_sim_150_all) + metric_real < 0.35) or
                            (sum_metrics < 0.5 and (mult_metrics < 0.03 or max_metrics < 0.2))):
                        metric_sim = -1
                        metric_sim_150 = -1
                        metric_real = -1
                        metric_sim_all = -1
                        metric_sim_150_all = -1
                        metric_real_add = -1

                else:
                    """ depth of the branch == 0"""
                    indices_real_matched_all = indices_real_matched
                    metric_sim_all = metric_sim
                    metric_real_add = metric_real
                    metric_sim_150_all = metric_sim_150
            else:
                indices_real_matched_all = indices_real_matched
                metric_sim = -1
                metric_sim_150 = -1
                metric_real = -1
                metric_sim_all = -1
                metric_sim_150_all = -1
                metric_real_add = -1
                or_opt = None
        else:
            raise ValueError("ndim should be 1 or 2")

        return (or_opt,
                q_sim_matched, indices_real_matched_all, indices_real_matched,
                metric_sim, metric_sim_150, metric_real,
                metric_sim_all, metric_sim_150_all, metric_real_add,)

    def get_best_orientation(self,
                             q_real: np.ndarray,
                             intens_real: np.ndarray,
                             q_sim_list: Union[List[np.ndarray], None],
                             intens_sim_list: Union[List[np.ndarray], None],
                             sim_config: Union[SimConfig, None],
                             orientations: np.ndarray,
                             q_range: Tuple[float, float],
                             ):
        """
            Finds the best orientation of the structure that matches the experimental pattern
            ...
            Parameters
            ----------
                q_real : np.ndarray
                    peak list in the experimental pattern, shape (peaks_real_num, ndim) - ndim=1 or 2
                intens_real : np.ndarray
                    peak intensities list in the experimental pattern, shape (peaks_real_num,)
                q_sim_list : Union[List[np.ndarray], None]
                    list of the simulated peak arrays (own array for each orientation)
                    each array has shape (peaks_sim_num, 2)
                    if None - calculate pattern manually
                intens_sim_list : Union[List[np.ndarray], None]
                    list of the simulated intensity arrays (own array for each orientation)
                    each array has shape (peaks_sim_num,)
                    if None - calculate pattern manually
                sim_config : Union[SimConfig, None]
                    configuration with q_sim_3d, intens_sim_3d and rec to calculate sim_patterns manually
                orientations : np.ndarray
                    allowed orientations
                q_range : Tuple[float, float]
                    Q value limits for xy and z axes
        """
        q_2d_sim_matched_opt = None
        indices_real_matched_opt = None
        metric_sim_opt = -1
        metric_sim_150_opt = -1
        or_opt = None
        metric_real_opt = -1

        for idx, orientation in enumerate(orientations):
            if q_sim_list is None:
                # patterns were not calculated in preprocessing
                q_sim, intens_sim = self.calculate_pattern(q_3d=sim_config.q_sim_3d,
                                                           rec=sim_config.rec,
                                                           intensity=sim_config.intens_sim_3d,
                                                           q_range=q_range,
                                                           orientation=orientation)
            else:
                q_sim = q_sim_list[idx]
                intens_sim = intens_sim_list[idx]
            q_2d_sim_matched, indices_real_matched, metric_sim, metric_sim_150 = self.get_match_metrics(
                q_real=q_real,
                q_sim=q_sim,
                intensities_sim=intens_sim,
                q_range=q_range,
            )
            if metric_sim > metric_sim_opt:
                or_opt = orientation
                q_2d_sim_matched_opt = q_2d_sim_matched
                indices_real_matched_opt = indices_real_matched
                metric_sim_opt = metric_sim
                metric_sim_150_opt = metric_sim_150

        if metric_sim_opt != -1:
            metric_real_opt = self.calculate_real_metric(intens_real, indices_real_matched_opt)

        return (or_opt,
                q_2d_sim_matched_opt, indices_real_matched_opt,
                metric_sim_opt, metric_sim_150_opt, metric_real_opt,)

    @staticmethod
    def calculate_real_metric(intens_real, indices_real_matched):
        """
            Calculates the metric for real intensities.
            ...
            Parameters
            ----------
                intens_real : np.ndarray
                    intensity values for each peak in the experimental pattern, shape (peaks_real_num,)
                indices_real_matched : np.ndarray
                    indices of the matched peaks, shape (peaks_matched_num,)

            Returns
            -------
                metric_real : np.float
                    metric value for the matched experimental peaks
        """
        intens_real = (intens_real / intens_real.max()) ** (1 / 3)
        sum_int_real = intens_real.sum()
        matched_logs_real = intens_real[indices_real_matched]  # (peaks_real_num)
        metric_real = matched_logs_real.sum() / sum_int_real
        return metric_real

    @staticmethod
    def calculate_pattern(q_3d,  # shape - (peaks_num, 3)
                          rec,
                          intensity,
                          q_range,
                          orientation,
                          ):
        R, orientation = rotate_vect(rec, orientation)
        q_3d = q_3d @ R

        q_2d, intensity, _ = GIWAXS.giwaxs_2d(
            q_3d=q_3d,
            intensity=intensity,
            mi=None,
            q_range=q_range,
            wavelength=12_398 / 18_000,
            move_fromMW=True,
        )
        return q_2d.T, intensity

    @staticmethod
    def get_match_metrics(
            q_real: np.ndarray,  # shape (peaks_real_num, ndim) - ndim=1 or 2
            q_sim: np.ndarray,  # shape (peaks_sim_num, ndim) - ndim=1 or 2
            intensities_sim: np.ndarray,  # shape (peaks_sim_num,)
            q_range: Tuple[float, float],
            max_distance: float = 0.05,
    ):
        """
            Matches simulated peaks to experimental peaks, returns matched peaks and metric value.
            ...
            Parameters
            ----------
                q_real : np.ndarray
                    peak list in the experimental pattern, shape (peaks_real_num, ndim) - ndim=1 or 2
                q_sim : np.ndarray
                    peak list in the simulated pattern, shape (peaks_sim_num, ndim) - ndim=1 or 2
                intensities_sim : np.ndarray
                    intensities list in the simulated pattern, shape (peaks_sim_num,)
                q_range : Tuple[float, float]
                    Q value limits for xy and z axes
                max_distance : float
                    maximum distance between experimental and simulated peak, default = 0.05

            Returns
            -------
                q_sim : np.ndarray
                    list of matched simulated peaks, shape (peaks_matched_num, ndim) - ndim=1 or 2
                indices_real : np.ndarray
                    indices of matched peaks in the experimental pattern, shape (peaks_matched_num,)
                matched_metric_sim : np.float
                    metric value for the matched simulated peaks
                matched_metric_sim_150 : np.float
                    metric value for the matched top150 simulated peaks
        """
        ndim = q_real.ndim
        q_range = np.array(q_range)
        if ndim == 1:
            q_range = np.linalg.norm(q_range)
            q_real, q_sim = q_real[:, np.newaxis], q_sim[:, np.newaxis]
            max_distance = np.sqrt(2) * max_distance

        # exclude simulated peaks outside q_range
        q_range_mask = (q_sim <= q_range).all(axis=1)
        q_sim = q_sim[q_range_mask]
        intensities_sim = intensities_sim[q_range_mask]

        # normalize intensities
        intensities_sim = (intensities_sim / intensities_sim.max()) ** (1 / 3)
        sum_int_sim = intensities_sim.sum()
        sum_int_sim_150 = np.sort(intensities_sim)[::-1][:150].sum()

        distances = cdist(q_sim, q_real)  # (peaks_sim_num, peaks_real_num)
        # exclude useless simulated peaks
        indices_sim_mask = np.where((distances < max_distance).sum(axis=1) >= 1)[0]
        distances = distances[indices_sim_mask]
        q_sim = q_sim[indices_sim_mask]
        intensities_sim = intensities_sim[indices_sim_mask]
        # exclude useless real peaks
        indices_real_mask = np.where((distances < max_distance).sum(axis=0) >= 1)[0]
        distances = distances[:, indices_real_mask]

        # Find the closest sym peak to each real peak
        beta = 0.5
        f_intensity = 1 + beta * intensities_sim[:, None]
        cost = distances / f_intensity
        indices_sim, indices_real = linear_sum_assignment(cost)
        dist_indices = np.where(distances[indices_sim, indices_real] < max_distance)[0]
        indices_sim, indices_real = indices_sim[dist_indices], indices_real[dist_indices]

        # Calculate the metric
        matched_logs_sim = intensities_sim[indices_sim]  # (peaks_real_num)
        matched_metric_sim = matched_logs_sim.sum() / sum_int_sim
        matched_metric_sim_150 = matched_logs_sim.sum() / sum_int_sim_150

        if ndim == 1:
            q_sim = q_sim.squeeze(1)

        return q_sim[indices_sim], indices_real_mask[indices_real], matched_metric_sim, matched_metric_sim_150
