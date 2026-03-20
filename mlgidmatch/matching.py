import torch
from typing import List, Union, Tuple, Literal
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path

from mlgidmatch.preprocess.cif_preprocess import CifPattern
from mlgidmatch.cif_matching.models.ResNet import IMGClassifier

from mlgidmatch.cif_matching.utils import ExpConfig
from mlgidmatch.cif_matching.cif_experiment_match import Match_CIF
from mlgidmatch.orient_matching.orient_experiment_match import Match_Orient


class Match:
    """
    A class for peak-to-structure matching of GID patterns.

    Attributes
    ----------
    cif_prepr : CifPattern
        A class with preprocessed CIFs.
    model_type : str, optional
        Name of the model. Default is 'ResNet18'.
    model_path : path, optional
        Path to the model weights.
    device : str or torch.device
    """

    def __init__(self,
                 cif_prepr: CifPattern,
                 *,
                 model_type: str = 'ResNet18',
                 model_path: Union[str, None] = None,
                 device='cuda'):
        if model_path is None:
            model_path = Path(__file__).parent / 'cif_matching' / 'models' / 'ResNet18_best_model.pt'
        if model_type == 'ResNet18':
            model = IMGClassifier(input_dim=14, output_dim=1, res=18).eval()
        elif model_type == 'ResNet34':
            model = IMGClassifier(input_dim=14, output_dim=1, res=34).eval()
        elif model_type == 'ResNet50':
            model = IMGClassifier(input_dim=14, output_dim=1, res=50).eval()
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval().to(device)

        self.peaks_type = None
        self.device = device

        self.config = ExpConfig(
            model=model,
            cif_prepr=cif_prepr,
        )
        self.cif_class = Match_CIF(self.config)
        self.orient_class = Match_Orient(self.config)

    def match_all(self,
                  measurements: List[str],
                  peak_list: List[ArrayLike],
                  intensities_real_list: List[ArrayLike],
                  q_range_list: List[Tuple[float, float]],
                  peaks_type: Literal['segments', 'rings'] = 'segments',
                  *,
                  threshold: float = 0.5,
                  candidates_list: Union[List[List[str]], None] = None,
                  save_metrics: bool = False,
                  ):
        """
        Create a list of solutions for the full matching process.

        Parameters
        ----------
        measurements : List[str]
            List of names of the measurements.
        peak_list : List[ArrayLike]
            List of experimental peak positions (one ArrayLike for each measurement).
            Length should be equal len(measurements).
        intensities_real_list : List[ArrayLike]
            List of experimental intensities corresponding to peak_list (one ArrayLike for each measurement).
            Length should be equal len(measurements).
        q_range_list : List[Tuple[float, float]]
            List of upper limits of q-range (for q_xy, q_z).
            Length should be equal len(measurements).
        peaks_type : Literal['segments', 'rings'], optional
            Type of the peaks. Default is 'segments'.
        threshold : float, optional
            Probability threshold to continue the matching process for the candidates. Default is 0.5.
        candidates_list : List[List[str]], optional
            List of lists of candidate structures (one list for each measurement).
            Length should equal len(measurements). If None, the whole dataset from self.config.cif_prepr will be used.
            Default is None.
        save_metrics : bool, optional
            True if you want to save all matching metrics for further analysis. Default is False.
        """
        self.peaks_type = peaks_type
        if self.peaks_type is None:
            raise ValueError('please specify peaks_type')

        assert len(measurements) == len(peak_list) == len(intensities_real_list) == len(q_range_list), \
            (f"lengths are not equal: {len(measurements)}, {len(peak_list)}, {len(intensities_real_list)},"
             f" {len(q_range_list)}")
        if candidates_list:
            assert len(measurements) == len(candidates_list), \
                f"lengths are not equal: {len(measurements)}, {len(candidates_list)}"

        full_data = {key: {} for key in measurements}
        for idx, meas in enumerate(measurements):
            peaks = np.asarray(peak_list[idx], dtype=np.float32)
            intens_real = np.asarray(intensities_real_list[idx], dtype=np.float32)
            q_range = q_range_list[idx]
            full_data[meas]['peaks'] = peaks

            candidates = None
            if candidates_list:
                candidates = candidates_list[idx]

            full_data[meas].update(
                self._build_tree(
                    peaks_all=peaks,
                    intens_real_all=intens_real,
                    q_range=q_range,
                    peaks_indices=np.arange(len(peaks)),
                    candidates=candidates,
                    threshold=threshold,
                    save_metrics=save_metrics,
                    depth=0,
                ),
            )
        return full_data

    def _build_tree(self, peaks_all, intens_real_all, q_range, peaks_indices, candidates, threshold, save_metrics,
                    depth):
        """Build the dictionary with all solutions."""

        if depth >= 3:
            # if three phases have already found
            return {}
        if len(peaks_indices) <= 3:
            # if too low number of peaks left for the matching
            return {}

        if threshold > 0:
            probs = self.match_cifs(
                peaks=peaks_all[peaks_indices],
                q_range=q_range,
                candidates=candidates,
            )
            if sum(probs >= threshold) == 0:
                # if all probabilities are too low
                return {}
        else:
            probs = np.ones(len(self.config.cif_prepr.cifs))

        if self.peaks_type == 'rings':
            peaks_input = np.linalg.norm(peaks_all, axis=-1)
        elif self.peaks_type == 'segments':
            peaks_input = peaks_all
        else:
            raise ValueError("peaks_type should be either 'rings' or 'segments'")
        data_matched = self.match_peaks(
            peaks_all=peaks_input,
            intens_real_all=intens_real_all,
            probs=probs,
            q_range=q_range,
            peaks_indices=peaks_indices,
            candidates=candidates,
            threshold=threshold,
            save_metrics=save_metrics,
        )
        if not data_matched:
            return {}

        for key, branch in data_matched.items():
            if len(branch['indices_real_matched']) == 0:
                continue

            mask = np.zeros(len(peaks_all), dtype=bool)
            mask[peaks_indices] = True
            mask[branch['indices_real_matched']] = False
            new_peaks_indices = np.arange(len(peaks_all))[mask]
            branch.update(
                self._build_tree(
                    peaks_all, intens_real_all, q_range, new_peaks_indices, candidates, threshold, save_metrics,
                    depth=depth + 1,
                ),
            )
        return data_matched

    def match_cifs(self, peaks, q_range, candidates=None):
        """Make Neural Matching for CIFs."""
        # if the list is not provided - use the whole dataset from self.config.cif_prepr
        candidate_ind = np.arange(len(self.config.cif_prepr.cifs))
        if candidates:
            candidate_ind = np.nonzero(np.isin(self.config.cif_prepr.cifs, candidates))[0]

        return self.cif_class.match(
            peak_list=peaks,
            q_range=q_range,
            candidate_ind=candidate_ind,
            batch_size=128,
            device=self.device,
        )

    def match_peaks(self, peaks_all, intens_real_all, probs, q_range, peaks_indices, candidates=None, threshold=0.5,
                    save_metrics=False):
        """Make peak-to-structure matching for candidate structures."""
        # if the list is not provided - use the whole dataset from self.config.cif_prepr
        candidate_ind = np.arange(len(self.config.cif_prepr.cifs))
        if candidates:
            candidate_ind = np.nonzero(np.isin(self.config.cif_prepr.cifs, candidates))[0]

        return self.orient_class.match(
            q_real_all=peaks_all,
            intens_real_all=intens_real_all,
            probs=probs,
            q_range=q_range,
            peaks_indices=peaks_indices,
            candidate_ind=candidate_ind,
            threshold=threshold,
            save_metrics=save_metrics,
        )

    def unique_solutions(self, data_matched: dict):
        """Find unique solutions in the list of solutions."""
        all_solutions = {meas: [] for meas in data_matched.keys()}
        for meas in data_matched.keys():
            peaks_num = len(data_matched[meas]['peaks'])
            seen = set()
            unique = {}
            cur_idx = 0
            for idx, sol in enumerate(
                    self._collect_solutions(
                        tree=data_matched[meas],
                        peaks_num=peaks_num,
                        prev_names=None,
                        path=None,
                    )
            ):
                h = self._make_hashable(sol)
                if h not in seen:
                    seen.add(h)
                    unique[cur_idx] = sol
                    cur_idx += 1
            all_solutions[meas] = unique
        return all_solutions

    def _make_hashable(self, solution):
        return frozenset(el['_unique_name'] for el in solution)

    def _collect_solutions(self, tree, peaks_num, prev_names=None, path=None):
        if prev_names is None:
            prev_names = set()
            path = []

        solutions = []

        for key in tree.keys():
            if not key.isdigit():
                continue

            branch = tree[key]
            cur_name = frozenset((branch['cif'], tuple(branch['orient'])))

            if cur_name in prev_names:
                for item in path:
                    if item['_unique_name'] == cur_name:
                        item['matched_peaks'][branch['indices_real_matched']] = branch['probability']
                        break
                new_path = path
                new_prev = prev_names
            else:
                probabilities = np.zeros(peaks_num)
                probabilities[branch['indices_real_matched_all']] = branch['probability']
                current = {'_unique_name': cur_name,
                           'cif': branch['cif'],
                           'orientation': tuple(branch["orient"]),
                           'matched_peaks': probabilities,
                           }

                new_path = path + [current]
                new_prev = prev_names | {cur_name}

            sub_solutions = self._collect_solutions(
                tree=branch,
                peaks_num=peaks_num,
                prev_names=new_prev,
                path=new_path,
            )

            if sub_solutions:
                solutions.extend(sub_solutions)
            else:
                solutions.append(new_path)

        return solutions
