import torch
from typing import List, Union, Tuple
import numpy as np
from pathlib import Path

from mlgidmatch.preprocess.cif_preprocess import CifPattern
from mlgidmatch.cif_matching.models.ResNet import IMGClassifier

from mlgidmatch.cif_matching.utils import ExpConfig
from mlgidmatch.cif_matching.cif_experiment_match import Match_CIF
from mlgidmatch.orient_matching.orient_experiment_match import Match_Orient


class Match:
    """
    A class for peak-to-structure matching of GID patterns.
    ...
    Attributes
    ----------
    """

    def __init__(self,
                 cif_class: CifPattern,
                 model_path: Union[str, None] = None,
                 device='cuda'):
        if model_path is None:
            model_path = Path(__file__).parent / 'cif_matching' / 'models' / 'ResNet18_newimage_14ch_state99999.pt'
        model = IMGClassifier(input_dim=14, output_dim=1, res=18).eval()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval().to(device)

        self.peaks_type = None
        self.device = device

        self.config = ExpConfig(
            model=model,
            cif_class=cif_class,
        )
        self.cif_class = Match_CIF(self.config)
        self.orient_class = Match_Orient(self.config)

    def match_all(self,
                  measurements: List[str],
                  peak_list: List[np.ndarray],
                  intensities_real_list: List[np.ndarray],
                  q_range_list: List[Tuple[float, float]],
                  threshold: float = 0.5,
                  candidates_list: Union[List[List[str]], None] = None,
                  peaks_type: str = None,  # 'segments' or 'rings'
                  ):
        self.peaks_type = peaks_type
        if self.peaks_type is None:
            raise ValueError('please specify peaks_type')

        full_data = {key: {} for key in measurements}
        for idx, meas in enumerate(measurements):
            peaks = peak_list[idx]
            intens_real = intensities_real_list[idx]
            q_range = q_range_list[idx]
            full_data[meas]['peaks'] = peaks
            if candidates_list is None:
                candidate_indices = np.arange(len(self.config.cif_class.cifs))
            else:
                candidate_indices = \
                    np.nonzero(np.isin(self.config.cif_class.cifs, candidates_list[idx]))[0]

            full_data[meas].update(
                self._build_tree(
                    peaks_all=peaks,
                    intens_real_all=intens_real,
                    q_range=q_range,
                    peaks_indices=np.arange(len(peaks)),
                    candidate_ind=candidate_indices,
                    threshold=threshold,
                    depth=0,
                ),
            )
        return full_data

    def _build_tree(self, peaks_all, intens_real_all, q_range, peaks_indices, candidate_ind, threshold, depth):
        if depth >= 3:
            return {}
        if len(peaks_indices) <= 3:
            return {}

        probs = self.match_cifs(peaks_all[peaks_indices], q_range, candidate_ind)
        if sum(probs >= threshold) == 0:
            return {}

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
            candidate_ind=candidate_ind,
            threshold=threshold,
        )
        if not data_matched:
            return {}

        for key, branch in data_matched.items():
            mask = np.ones(len(peaks_indices), dtype=bool)
            if len(branch['_indices_real_matched']) == 0:
                continue
            mask[branch['_indices_real_matched']] = False
            new_peaks_indices = peaks_indices[mask]
            branch.update(
                self._build_tree(
                    peaks_all, intens_real_all, q_range, new_peaks_indices, candidate_ind, threshold,
                    depth=depth + 1,
                ),
            )
        return data_matched

    def match_cifs(self, peaks, q_range, candidate_ind):
        return self.cif_class.match(
            peak_list=peaks,
            q_range=q_range,
            candidate_ind=candidate_ind,
            batch_size=128,
            device=self.device,
        )

    def match_peaks(self, peaks_all, intens_real_all, probs, q_range, peaks_indices, candidate_ind, threshold):
        return self.orient_class.match(
            peaks_all=peaks_all,
            intens_real_all=intens_real_all,
            probs=probs,
            q_range=q_range,
            peaks_indices=peaks_indices,
            candidate_ind=candidate_ind,
            threshold=threshold,
        )

    def unique_solutions(self, data_matched):
        all_solutions = {meas: [] for meas in data_matched.keys()}
        for meas in data_matched.keys():
            seen = set()
            unique = []
            for sol in self.collect_solutions(data_matched[meas]):
                h = self.make_hashable(sol)
                if h not in seen:
                    seen.add(h)
                    unique.append(sol)
            all_solutions[meas] = unique
        return all_solutions

    def make_hashable(self, solution):
        return frozenset((cif, tuple(orient)) for _, cif, orient in solution)

    def collect_solutions(self, tree, depth=0):
        solutions = []
        for key in tree.keys():
            if not key.isdigit():
                continue
            branch = tree[key]

            current = [(key, branch["cif"], branch["orient"])]
            sub_solutions = self.collect_solutions(branch, depth + 1)
            if sub_solutions:
                for sub in sub_solutions:
                    solutions.append(current + sub)
            else:
                solutions.append(current)

        return solutions


if __name__ == "__main__":
    import pickle
    from mlgidmatch.preprocess.cif_preprocess import *

    with open(
            './data/prepr_cifs.pickle',
            'rb',
    ) as file:
        cif_cl = pickle.load(file)
    match_class = Match(
        # model_path='./cif_matching/models/ResNet18_newimage_14ch_state99999.pt',
        cif_class=cif_cl,
        peaks_type='segments',
        device='cuda',
    )

    from pygidsim.experiment import ExpParameters
    from pygidsim.giwaxs_sim import GIWAXSFromCif

    params = ExpParameters(q_xy_max=2.7, q_z_max=2.7, en=18000)

    path_to_cif_1 = '/data/romodin/gi_matching/dataset/experiment/perovskites/cifs/1_BA2PbI4_n1.cif'
    el_1 = GIWAXSFromCif(path_to_cif_1, params)
    q_2d_1, intensity_1 = el_1.giwaxs.giwaxs_sim(
        orientation=np.array([5., 1., 2.]),
        move_fromMW=True,
    )  # q_2d is array with shape (2, peaks number)
    idx = np.argsort(intensity_1)[-15:]
    q_2d_real = q_2d_1[:, idx]
    intensity_real = intensity_1[idx]

    path_to_cif_2 = '/data/romodin/gi_matching/dataset/experiment/perovskites/cifs/581_BA2FAPb2I7_n2.cif'
    el_2 = GIWAXSFromCif(path_to_cif_2, params)
    q_2d_2, intensity_2 = el_2.giwaxs.giwaxs_sim(
        orientation=np.array([1., 1., 2.]),
        move_fromMW=True,
    )  # q_2d is array with shape (2, peaks number)
    idx = np.argsort(intensity_2)[-15:]
    q_2d_real = np.concatenate((q_2d_real, q_2d_2[:, idx]), axis=1)
    intensity_real = np.concatenate((intensity_real, intensity_2[idx]), axis=0)

    data_matched = match_class.match_all(
        measurements=['Own_Meas'],
        peak_list=[q_2d_real.T],
        intensities_real_list=[intensity_real],
        q_range_list=[(2.7, 2.7)],
        candidates_list=None, )
    print(match_class.unique_solutions(data_matched))
    for key_0 in data_matched['Own_Meas'].keys():
        if not key_0.isdigit():
            continue
        print(data_matched['Own_Meas'][key_0]['cif'] + ' ' + str(data_matched['Own_Meas'][key_0]['orient']))
        for key_1 in data_matched['Own_Meas'][key_0].keys():
            if not key_1.isdigit():
                continue
            print(
                '   ', data_matched['Own_Meas'][key_0][key_1]['cif'],
                data_matched['Own_Meas'][key_0][key_1]['orient'],
            )
            for key_2 in data_matched['Own_Meas'][key_0][key_1].keys():
                if not key_2.isdigit():
                    continue
                print(
                    '   ', data_matched['Own_Meas'][key_0][key_1][key_2]['cif'],
                    data_matched['Own_Meas'][key_0][key_1][key_2]['orient'],
                )
