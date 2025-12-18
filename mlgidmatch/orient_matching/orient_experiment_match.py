import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from pygidsim.giwaxs_sim import GIWAXS
from mlgidmatch.cif_matching.utils import ExpConfig
from mlgidmatch.orient_matching.utils import SimConfig
from mlgidmatch.preprocess.rotate import rotate_vect


@dataclass
class DataForMatch:
    """
    A class to keep essential data for peak matching for GID patterns with one CIF.

    Attributes
    ----------
    q_real_all : np.ndarray
        Shape (peaks_real_num_all, ndim), ndim = 1 or 2.
        Full input peak list in the experimental pattern.
    intens_real_all : np.ndarray
        Shape (peaks_real_num_all,).
        Intensities corresponding to q_real_all.
    peaks_indices : np.ndarray
        Shape (peaks_real_num,).
        Indices of the subset according to the full peak list q_real_all
        (left for the current iteration).
    q_sim_list : List[np.ndarray] or np.ndarray or None
        List of the simulated peak positions arrays (one array for each orientation).
        Each array has shape (peaks_sim_num, 2).
        OR only one np.ndarray for powder3d pattern.
        If None (sim_list was not prepared in preprocessing), the pattern is calculated manually.
    intens_sim_list : List[np.ndarray] or np.ndarray or None
        List of the simulated intensity arrays (one array for each orientation).
        Each array has shape (peaks_sim_num,).
        OR only one np.ndarray for powder3d pattern.
        If None (sim_list was not prepared in preprocessing), the pattern is calculated manually.
    sim_config : SimConfig or None
        Configuration with q_sim_3d, intens_sim_3d and rec to calculate simulated patterns manually.
    orientations : np.ndarray or None
        Shape (orient_num, 3). All possible orientations for the structure.
        For the powder3d pattern: None.
    q_range : Tuple[float, float]
        Upper limits of q-range (for q_xy, q_z).
    """
    q_real_all: np.ndarray  # (peaks_num_all, ndim), ndim = 1 or 2
    intens_real_all: np.ndarray  # (peaks_num_all,)
    peaks_indices: np.ndarray  # shape (peaks_real_num,)
    q_sim_list: Union[List[np.ndarray], np.ndarray, None]
    intens_sim_list: Union[List[np.ndarray], np.ndarray, None]
    sim_config: Union[SimConfig, None]
    orientations: Union[np.ndarray, None]
    q_range: Tuple[float, float]


class Match_Orient():
    """
    A class to make peak-to-structure matching for GID patterns with CIFs.

    Attributes
    ----------
    config : ExpConfig
        Configuration of the experiment.
    """

    def __init__(self,
                 config: ExpConfig,
                 ):
        self.config = config

    def match(self,
              q_real_all: np.ndarray,  # (peaks_num_all, ndim)
              intens_real_all: np.ndarray,  # (peaks_num_all,)
              probs: np.ndarray,  # (candidates_num,)
              q_range: Tuple[float, float],
              peaks_indices: np.ndarray,  # (peaks_num,)
              candidate_ind: np.ndarray,  # (candidates_num,)
              threshold: float,
              save_metrics: bool,
              ):
        """
        Create peak-to-structure matching for the GID experimental pattern.

        Parameters
        ----------
        q_real_all : np.ndarray
            Shape (peaks_real_num_all, ndim), ndim = 1 or 2.
            Full input peak list in the experimental pattern.
        intens_real_all : np.ndarray
            Shape (peaks_real_num_all,).
            Intensities corresponding to q_real_all.
        probs : np.ndarray
            Probabilities (from neural matching) for the candidate structures.
        q_range : Tuple[float, float]
            Upper limits of q-range (for q_xy, q_z).
        peaks_indices : np.ndarray
            Shape (peaks_real_num,).
            Indices of the subset according to the full peak list q_real_all
            (left for the current iteration).
        candidate_ind : np.ndarray
            Indices of the candidate structures corresponding to self.config.cif_prepr.cifs.
        threshold : float
            Probability threshold.
        save_metrics : bool
            If True - save all metrics in the output.
        """

        valid_indices = candidate_ind[np.where(probs >= threshold)[0]]
        if len(valid_indices) == 0:
            return {}

        data_matched = self.test_sev_cifs(
            cif_indices_list=valid_indices,
            q_real_all=q_real_all,
            intens_real_all=intens_real_all,
            peaks_indices=peaks_indices,
            q_range=q_range,
        )

        (cif_indices_list, orients,
         _q_sim_matched_list_, indices_real_matched_all, indices_real_matched,
         metrics_sim, metrics_sim_150, metrics_real,
         metrics_sim_all, metrics_sim_150_all, metrics_real_add,) = data_matched

        answ_indices = np.nonzero(np.isin(candidate_ind, cif_indices_list))[0]
        data_matched_dict = {}
        for key in range(len(cif_indices_list)):
            entry = {
                'cif': self.config.cif_prepr.cifs[cif_indices_list[key]],
                'orient': orients[key].astype(int),
                'probability': probs[answ_indices[key]],
                'indices_real_matched_all': indices_real_matched_all[key],
                'indices_real_matched': peaks_indices[indices_real_matched[key]],
            }
            if save_metrics:
                entry.update(
                    {
                        'metric_sim': metrics_sim[key],
                        'metric_sim_150': metrics_sim_150[key],
                        'metric_real': metrics_real[key],
                        'metric_sim_all': metrics_sim_all[key],
                        'metric_sim_150_all': metrics_sim_150_all[key],
                        'metric_real_add': metrics_real_add[key],
                    },
                )
            data_matched_dict[str(key)] = entry
        return data_matched_dict

    def test_sev_cifs(self,
                      cif_indices_list: np.ndarray,  # indices of cifs to test
                      q_real_all: np.ndarray,  # (peaks_num_all, ndim) - ndim=1 or 2
                      intens_real_all: np.ndarray,  # (peaks_num_all,)
                      peaks_indices: np.ndarray,  # shape (peaks_real_num,)
                      q_range: Tuple[float, float],
                      ):
        """Match several CIFs in loop."""
        data_matched = [
            self.test_one_cif(
                self._prepare_input(
                    idx,
                    q_real_all,
                    intens_real_all,
                    peaks_indices,
                    q_range,
                ),
            ) for idx in
            cif_indices_list
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

    def _prepare_input(self,
                       idx: int,
                       q_real_all: np.ndarray,  # (peaks_num_all, ndim) - ndim=1 or 2
                       intens_real_all: np.ndarray,  # (peaks_num_all,)
                       peaks_indices: np.ndarray,  # shape (peaks_real_num,)
                       q_range: Tuple[float, float],
                       ):
        is_2d = (q_real_all.ndim == 2)
        cfg = self.config.cif_prepr
        need_sim_config = (is_2d and cfg.all_patterns_q2d is None) or (
                (not is_2d) and cfg.all_patterns_q1d is None)
        return DataForMatch(
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
            ) if need_sim_config else None,
            orientations=self.config.cif_prepr.pattern_3d.orientations[idx],
            q_range=q_range,
        )

    def test_one_cif(self,
                     in_data: DataForMatch, ) -> tuple:
        ndim = in_data.q_real_all.ndim
        if ndim == 1:
            data_matched = self.test_rings(in_data)
        elif ndim == 2:
            data_matched = self.test_segments(in_data)
        else:
            raise ValueError("ndim should be 1 or 2")
        return data_matched

    def test_rings(self,
                   in_data: DataForMatch,
                   ) -> tuple:
        """Make matching for 1D data (for rings)."""
        q_sim_matched, indices_real_matched, metric_sim, metric_sim_150 = self.get_match_metrics(
            q_real=in_data.q_real_all[in_data.peaks_indices],
            q_sim=in_data.q_sim_list,
            intensities_sim=in_data.intens_sim_list,
            q_range=in_data.q_range,
        )
        metric_real = self.calculate_real_metric(in_data.intens_real_all[in_data.peaks_indices], indices_real_matched)
        if len(in_data.q_real_all) != len(in_data.peaks_indices):
            # depth of the branch > 0
            q_sim_matched, indices_real_matched_all, metric_sim_all, metric_sim_150_all = self.get_match_metrics(
                q_real=in_data.q_real_all,
                q_sim=in_data.q_sim_list,
                intensities_sim=in_data.intens_sim_list,
                q_range=in_data.q_range,
            )
            metric_real_add = self.calculate_real_metric(
                in_data.intens_real_all, in_data.peaks_indices[indices_real_matched],
            )
        else:
            # depth of the branch = 0
            indices_real_matched_all = indices_real_matched
            metric_sim_all = metric_sim
            metric_sim_150_all = metric_sim_150
            metric_real_add = metric_real
        orientation = np.array([0, 0, 0])

        return (orientation,
                q_sim_matched, indices_real_matched_all, indices_real_matched,
                metric_sim, metric_sim_150, metric_real,
                metric_sim_all, metric_sim_150_all, metric_real_add,)

    def test_segments(self,
                      in_data: DataForMatch,
                      ) -> tuple:
        """Make matching for 2D data (for segments)."""

        # find the orientation and matched peaks
        or_opt, q_sim_matched, indices_real_matched, metric_sim, metric_sim_150, metric_real = (
            self.get_best_orientation(
                q_real=in_data.q_real_all[in_data.peaks_indices],
                intens_real=in_data.intens_real_all[in_data.peaks_indices],
                q_sim_list=in_data.q_sim_list,
                intens_sim_list=in_data.intens_sim_list,
                sim_config=in_data.sim_config,
                orientations=in_data.orientations,
                q_range=in_data.q_range,
            )
        )

        # expert predefined thresholds
        if ((metric_sim >= 0.04) &
                (metric_sim_150 >= 0.05) &
                (metric_real >= 0.15) &
                (len(q_sim_matched) > 3)):
            if len(in_data.q_real_all) != len(in_data.peaks_indices):
                # depth of the branch > 0
                if in_data.q_sim_list is None:
                    # the pattern was not calculated in preprocessing
                    q_sim, intens_sim = self.calculate_pattern(
                        q_3d=in_data.sim_config.q_sim_3d,
                        rec=in_data.sim_config.rec,
                        intensity=in_data.sim_config.intens_sim_3d,
                        q_range=in_data.q_range,
                        orientation=or_opt,
                    )
                else:
                    idx_opt = np.where(np.all(in_data.orientations == or_opt, axis=1))[0].item()
                    q_sim, intens_sim = in_data.q_sim_list[idx_opt], in_data.intens_sim_list[idx_opt]
                q_sim_matched, indices_real_matched_all, metric_sim_all, metric_sim_150_all = self.get_match_metrics(
                    q_real=in_data.q_real_all,
                    q_sim=q_sim,
                    intensities_sim=intens_sim,
                    q_range=in_data.q_range,
                )
                metric_real_add = self.calculate_real_metric(
                    in_data.intens_real_all, in_data.peaks_indices[indices_real_matched],
                )
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
                # depth of the branch == 0
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
        Find the best orientation for the structure that matches the experimental pattern.

        Parameters
        ----------
        q_real : np.ndarray
            Shape (peaks_real_num, ndim), ndim = 1 or 2.
            Input peak list in the experimental pattern.
        intens_real : np.ndarray
            Shape (peaks_real_num,). Intensities corresponding to q_real.
        q_sim_list : List[np.ndarray] or None
            List of the simulated peak positions arrays (one array for each orientation).
            Each array has shape (peaks_sim_num, 2).
            If None (sim_list was not prepared in preprocessing), the pattern is calculated manually.
        intens_sim_list : List[np.ndarray] or None]
            List of the simulated intensity arrays (one array for each orientation).
            Each array has shape (peaks_sim_num,).
            If None (sim_list was not prepared in preprocessing), the pattern is calculated manually.
        sim_config : SimConfig or None
            Configuration with q_sim_3d, intens_sim_3d and rec to calculate simulated patterns manually.
        orientations : np.ndarray
            Allowed orientations for the structure.
        q_range : Tuple[float, float]
            Upper limits of q-range (for q_xy, q_z).
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
                q_sim, intens_sim = self.calculate_pattern(
                    q_3d=sim_config.q_sim_3d,
                    rec=sim_config.rec,
                    intensity=sim_config.intens_sim_3d,
                    q_range=q_range,
                    orientation=orientation,
                )
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
        Calculates the metric for real (experimental) intensities.

        Parameters
        ----------
        intens_real : np.ndarray
            Shape (peaks_real_num,). Intensities corresponding to the experimental pattern.
        indices_real_matched : np.ndarray
            Shape (peaks_matched_num,). Indices of the matched experimental peaks.

        Returns
        -------
        metric_real : np.float
            Metric value for the matched experimental peaks.
        """

        intens_real = (intens_real / intens_real.max()) ** (1 / 3)
        sum_int_real = intens_real.sum()
        matched_logs_real = intens_real[indices_real_matched]  # (peaks_real_num)
        metric_real = matched_logs_real.sum() / sum_int_real
        return metric_real

    @staticmethod
    def calculate_pattern(q_3d: np.ndarray,  # shape - (peaks_num, 3)
                          rec,  # shape - (3, 3)
                          intensity: np.ndarray,  # shape - (peaks_num,)
                          q_range: Tuple[float, float],
                          orientation: np.ndarray,  # shape - (3,)
                          ):
        """Calculates simulated GID pattern."""
        R = rotate_vect(rec, orientation)
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
            q_real: np.ndarray,  # (peaks_real_num, ndim), ndim=1 or 2
            q_sim: np.ndarray,  # (peaks_sim_num, ndim), ndim=1 or 2
            intensities_sim: np.ndarray,  # (peaks_sim_num,)
            q_range: Tuple[float, float],
            max_distance: float = 0.05,
    ):
        """
        Match simulated peaks to experimental peaks and return matched peaks with metric value.

        Parameters
        ----------
        q_real : np.ndarray
            Shape (peaks_real_num, ndim), ndim = 1 or 2.
            Input peak list in the experimental pattern.
        q_sim : np.ndarray
            Shape (peaks_sim_num, ndim), ndim = 1 or 2.
            Peak list in the simulated pattern.
        intensities_sim : np.ndarray
            Shape (peaks_sim_num,).
            Intensities corresponding to q_sim.
        q_range : Tuple[float, float]
            Upper limits of q-range (for q_xy, q_z).
        max_distance : float
            Maximum distance between experimental and simulated peak.
            Default is 0.05.

        Returns
        -------
        q_sim : np.ndarray
            Shape (peaks_matched_num, ndim), ndim = 1 or 2.
            List of matched simulated peaks.
        indices_real : np.ndarray
            Shape (peaks_matched_num,).
            Indices of matched peaks according to the experimental input peak list q_real.
        matched_metric_sim : float
            Metric value for the matched simulated peaks.
        matched_metric_sim_150 : float
            Metric value for the matched top150 simulated peaks.
        """

        ndim = q_real.ndim
        q_range = np.array(q_range)
        if ndim == 1:
            q_range = np.linalg.norm(q_range)
            q_real, q_sim = q_real[:, np.newaxis], q_sim[:, np.newaxis]
            max_distance = np.sqrt(2) * max_distance

        # exclude simulated peaks outside the q_range
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

        # Find the closest sim peak to each real peak
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
