import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from pygidsim.giwaxs_sim import GIWAXSFromCif, GIWAXS
from pygidsim.experiment import ExpParameters
from pygidsim.q_sim import QPos
from pygidsim.int_sim import Intensity
from pymatgen.core import Structure
import pymatgen.core.surface as surface

from mlgidmatch.preprocess.rotate import rotate_vect
from mlgidmatch.preprocess.directions import get_unique_directions
from mlgidmatch.preprocess.utils import unique, limit_q, limit_int, lorentz_correction_2d

from typing import List
import warnings


class Pattern3d(object):
    def __init__(self,
                 q_3d: List[np.ndarray],  # each array with shape (points_num, 3)
                 rec: np.ndarray,  # (str_num, 3, 3,)
                 intensities: List[np.ndarray],  # each array with shape (str_num, points_num),
                 lengths: List[int],  # (str_num)
                 orientations: List[np.ndarray],  # each array with shape (orient_num, 3),
                 ):
        self.q_3d = q_3d
        self.rec = rec
        self.intensities = intensities
        self.lengths = lengths
        self.orientations = orientations


class CifPattern(object):
    """
        A class to parse CIFs and prepare them for evaluation.
        ...
        Attributes
        ----------
            folder_path : str
                folder with cifs
            cifs : List[str]
                paths to cifs
            params : ExpParameters
            pattern_3d : Pattern3d
                class with all essential data for GID pattern simulation
            background : torch.Tensor, shape - (struct_num, 13, 100, 3)
                peaks coordinates in 2D q-space for 13 basic orientations
                (100 - peaks number, 3 - q_xy, q_z, intensity)
            all_patterns_q : Union[List[List[np.ndarray]]], None], default=None
                peak positions for given cifs and all possible orientations
            all_patterns_int : Union[List[List[np.ndarray]]], None], default=None
                intensities for given cifs and all possible orientations
    """

    def __init__(self, params, folder_path, cifs=None, create_all=False):
        self.params = params
        self.folder_path = folder_path
        self.cifs = cifs

        self.pattern_3d = self.calculate_patterns3d()
        self.background = self.create_background()
        self.all_patterns_q2d = None
        self.all_patterns_int2d = None
        self.all_patterns_q1d = None
        self.all_patterns_int1d = None
        if create_all:
            self.all_patterns_q2d, self.all_patterns_int2d, self.all_patterns_q1d, self.all_patterns_int1d = \
                self.create_all_possible_patterns()

    def calculate_patterns3d(self):
        cif_list = []
        rec_list = []
        q_list = []
        intensities_list = []
        lengths = []
        orientations = []

        if self.cifs is None:
            self.cifs = os.listdir(self.folder_path)
        self.cifs = [os.path.join(self.folder_path, filename) for filename in self.cifs]

        print("Parse CIFs")
        for idx, cif_path in enumerate(self.cifs):
            if os.path.isfile(cif_path) and cif_path.lower().endswith('.cif'):
                cif_list.append(os.path.basename(cif_path))
                el = GIWAXSFromCif(cif_path, self.params).giwaxs
                intensity = Intensity(
                    el.crystal.atoms,
                    el.crystal.atom_positions,
                    el.crystal.occ,
                    el.q_3d,
                    el.mi,
                    el.exp.wavelength,
                    el.exp.ai,
                    el.exp.database,
                ).get_intensities()
                rec_list.append(el.rec)
                q_list.append(el.q_3d)
                intensities_list.append(intensity)
                lengths.append(len(intensity))

                try:
                    pym_structure = Structure.from_file(cif_path)
                    unique_orientations = surface.get_symmetrically_distinct_miller_indices(pym_structure, max_index=5)
                except (ValueError, AttributeError, TypeError):
                    unique_orientations = get_unique_directions(max_index=5)
                orientations.append(np.array(unique_orientations, dtype=np.float32))
            else:
                warnings.warn(f"not CIF {cif_path}")
        rec_list = np.stack(rec_list, axis=2, dtype=np.float32)  # (3, 3, str_num)
        rec_list = np.transpose(rec_list, (2, 0, 1))  # (str_num, 3, 3)
        pattern_3d = Pattern3d(
            q_3d=q_list,
            rec=rec_list,
            intensities=intensities_list,
            lengths=lengths,
            orientations=orientations,
        )
        self.cifs = cif_list
        print("Parsing finished\n")
        return pattern_3d

    def create_background(self, top_peaks=100):
        """Create 13 ideal patterns with 'elementary' orientations from matching_rows"""
        matching_rows = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 1, 0],
             [1, 0, 1],
             [0, 1, 1],
             [1, 1, 1],
             [1, 1, -1],
             [1, -1, 1],
             [1, -1, -1],
             [1, -1, 0],
             [1, 0, -1],
             [0, 1, -1],
             ],
        )

        # q_range_max = np.sqrt(self.params.q_xy_max ** 2 + self.params.q_z_max ** 2)
        background = torch.empty(
            len(self.cifs), len(matching_rows), top_peaks, 3,
        )  # 3 - q2d + intensity
        print("Create background")
        for idx in range(len(self.cifs)):
            q_list = []
            int_list = []
            for row in matching_rows:
                R, orientation = rotate_vect(self.pattern_3d.rec[idx], orientation=row)
                q_3d = self.pattern_3d.q_3d[idx] @ R
                #
                # q_2d, intensity, _ = GIWAXS.giwaxs_2d(
                #     q_3d=q_3d,
                #     intensity=self.pattern_3d.intensities[idx],
                #     mi=None,
                #     q_range=(self.params.q_xy_max, self.params.q_z_max),
                #     wavelength=self.params.wavelength,
                #     move_fromMW=False,
                # )

                q_xy = np.linalg.norm(q_3d[:, :2], axis=1)
                q_z = q_3d[:, -1]
                q_2d = np.concatenate((q_xy[:, np.newaxis], q_z[:, np.newaxis]), axis=1)
                q_2d, intensity = limit_q(
                    q_2d, self.pattern_3d.intensities[idx], (self.params.q_xy_max, self.params.q_z_max),
                )
                # intensity_corr = lorentz_correction_2d(q_2d, intensity)
                q_2d, intensity = unique(q_2d, intensity)
                q_2d, intensity = limit_int(q_2d, intensity, top_peaks=top_peaks)

                q_2d = torch.tensor(q_2d, dtype=torch.float32, device='cpu')
                q_list.append(q_2d)
                int_list.append(torch.tensor(intensity))

            q_2d_pad = pad_sequence(q_list, batch_first=True, padding_value=0.0)
            intensity_pad = pad_sequence(int_list, batch_first=True, padding_value=0.0)
            q_tensor = torch.concat((q_2d_pad, intensity_pad.unsqueeze(-1)), dim=-1)
            if q_tensor.shape[1] < top_peaks:
                q_tensor = torch.cat(
                    (q_tensor,
                     torch.zeros(
                         (len(matching_rows),
                          top_peaks - q_tensor.shape[1],
                          3),
                     )), dim=1,
                )

            background[idx] = q_tensor
        print("Background created\n")
        return background

    def create_all_possible_patterns(self):
        full_q_2d = []
        full_intensity_2d = []
        full_q_1d = []
        full_intensity_1d = []

        print("Create all possible patterns")
        for idx, cif in enumerate(self.cifs):
            q_2d_list = []
            intensity_list = []
            for orientation in self.pattern_3d.orientations[idx]:
                R, orientation = rotate_vect(self.pattern_3d.rec[idx], orientation)
                q_3d = self.pattern_3d.q_3d[idx] @ R

                q_2d, intensity, _ = GIWAXS.giwaxs_2d(
                    q_3d=q_3d,
                    intensity=self.pattern_3d.intensities[idx],
                    mi=None,
                    q_range=(self.params.q_xy_max, self.params.q_z_max),
                    wavelength=self.params.wavelength,
                    move_fromMW=True,
                )

                # remove peaks with very low intensities
                max_peaks = 1000
                intens_norm = (intensity / intensity.max())
                sort_idx = np.where(intens_norm > 0.01)[0]
                if len(sort_idx) < max_peaks:
                    sort_idx = np.argsort(intensity)[::-1][:max_peaks]

                q_2d_list.append(q_2d.T[sort_idx])
                intensity_list.append(intensity[sort_idx])

            full_q_2d.append(q_2d_list)
            full_intensity_2d.append(intensity_list)

            q_1d, int_1d = self.create_powder3d_pattern(idx)
            full_q_1d.append(q_1d)
            full_intensity_1d.append(int_1d)
        print("All patterns created\n")
        return full_q_2d, full_intensity_2d, full_q_1d, full_intensity_1d

    def create_powder3d_pattern(self, idx):
        q_1d = np.linalg.norm(self.pattern_3d.q_3d[idx], axis=-1)
        q_1d, int_1d, _ = GIWAXS.giwaxs_1d(
            q_1d,
            intensity=self.pattern_3d.intensities[idx],
            mi=None,
            wavelength=self.params.wavelength,
        )
        return q_1d, int_1d


if __name__ == '__main__':
    # folder_path = '/home/romodin/Romodos/Packages/mlgidMATCH/mlgidmatch/data/cifs/'
    folder_path = '/data/romodin/gi_matching/forms_cifs/'
    # all_cifs = ['1_BA2PbI4_n1.cif', '5_BA2MAPb2I7_n2.cif', '6_BA2MA2Pb3I10_n3.cif', '576_PEA2PbI4_n1.cif',
    #             '579_PEA2MAPb2I7_n2.cif', '581_BA2FAPb2I7_n2.cif', 'Bn-Br_test.cif', 'hex_2H_S41.cif',
    #             'hex_4H_S41.cif', 'hex_6H_S41.cif']

    params = ExpParameters(q_xy_max=5, q_z_max=5, en=18_000)
    cif_prepr = CifPattern(
        params=params,
        folder_path=folder_path,
        # cifs=all_cifs,
        create_all=True,
    )

    with open('/data/romodin/gi_matching/prepr_cifs.pickle', 'wb') as file:
        pickle.dump(cif_prepr, file)
    print('FINALLY:', len(cif_prepr.cifs), 'structures')
