import sys
import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from pygidsim.experiment import ExpParameters
from pygidsim.giwaxs_sim import GIWAXSFromCif, GIWAXS
from pygidsim.int_sim import Intensity
from pymatgen.core import Structure
import pymatgen.core.surface as surface

from mlgidmatch.preprocess.rotate import rotate_vect
from mlgidmatch.preprocess.directions import get_unique_directions
from mlgidmatch.preprocess.utils import limit_int

from typing import List
import warnings


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Pattern3d(object):
    """A class with all essential data for GID pattern simulation."""

    def __init__(self,
                 q_3d: List[np.ndarray],  # each array with shape (peaks_num, 3)
                 rec: np.ndarray,  # (str_num, 3, 3,)
                 intensities: List[np.ndarray],  # each array with shape (peaks_num, ),
                 lengths: List[int],  # list of peaks_num for each structure
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

    Attributes
    ----------
    params : ExpParameters
        Experimental parameters.
    folder_path : str
        Folder with CIFs.
    cifs : List[str], optional
        CIF files names.
    create_all : bool, optional
        Whether to generate all patterns. Default is False.
    preprocessed_3d : str, optional
        Path to preprocessed class (with calculated 3d patterns).
    pattern_3d : Pattern3d
        Class with all essential data for GID pattern simulation.
    elementary : torch.Tensor
        Shape (struct_num, 13, 100, 3). Peaks coordinates in 2D q-space for 13 basic orientations
        (100 peaks, 3 = q_xy, q_z, intensity).
    all_patterns_q2d : List[List[np.ndarray]] or None
        Peak positions in 2D for given cifs and all possible unique orientations.
        len(self.all_patterns_q2d) == len(self.cifs).
        len(self.all_patterns_q2d[i]) == number of unique orientations for self.cifs[i].
    all_patterns_int2d : List[List[np.ndarray]] or None
        Intensities in 2D for given cifs and all possible orientations.
        len(self.all_patterns_int2d) == len(self.cifs).
        len(self.all_patterns_int2d[i]) == number of unique orientations for self.cifs[i].
    all_patterns_q1d : List[np.ndarray] or None
        Peak positions in 1D for given cifs and all possible unique orientations.
        len(self.all_patterns_q2d) == len(self.cifs).
    all_patterns_int1d : List[List[np.ndarray]] or None
        Intensities in 1D for given cifs.
        len(self.all_patterns_int2d) == len(self.cifs).
    """

    def __init__(self,
                 params: ExpParameters,
                 folder_path: str,
                 cifs: List[str] = None,
                 create_elementary: bool = True,
                 create_all: bool = False,
                 preprocessed_3d: str = None,
                 ):
        self.params = params
        self.folder_path = folder_path
        self.cifs = cifs

        if not preprocessed_3d:
            self.cifs = cifs
            self.pattern_3d = self._calculate_patterns3d()
        else:
            with open(preprocessed_3d, 'rb') as file:
                data = pickle.load(file)
                assert (np.sort(data.cifs) == np.sort(cifs)).all(), "cifs and preprocessed_3d do not match"
                self.cifs = data.cifs
                self.pattern_3d = data.pattern_3d

        self.elementary = None
        if create_elementary:
            self.elementary = self._create_elementary()

        self.all_patterns_q2d = None
        self.all_patterns_int2d = None
        self.all_patterns_q1d = None
        self.all_patterns_int1d = None
        if create_all:
            self.all_patterns_q2d, self.all_patterns_int2d, self.all_patterns_q1d, self.all_patterns_int1d = \
                self._create_all_possible_patterns()

    def _calculate_patterns3d(self):
        """Return the Pattern3d class with calculated patterns in 3D"""

        cif_list = []
        rec_list = []
        q_list = []
        intensities_list = []
        lengths = []
        orientations = []

        if self.cifs is None:
            self.cifs = os.listdir(self.folder_path)
        _cifs = [os.path.join(self.folder_path, filename) for filename in self.cifs]

        for idx, cif_path in enumerate(_cifs):
            if os.path.isfile(cif_path) and cif_path.lower().endswith('.cif'):
                with SuppressPrint():
                    try:
                        el = GIWAXSFromCif(cif_path, self.params).giwaxs
                    except:
                        warnings.warn(f"could not parse {cif_path}")
                        continue
                intensity = Intensity(
                    atoms=el.crystal.atoms,
                    atom_positions=el.crystal.atom_positions,
                    occ=el.crystal.occ,
                    q_3d=el.q_3d,
                    mi=el.mi,
                    wavelength=el.exp.wavelength,
                    ai=el.exp.ai,
                    database=el.exp.database,
                ).get_intensities()
                cif_list.append(os.path.basename(cif_path))
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
        # print("parsing finished\n")
        return pattern_3d

    def _create_elementary(self, top_peaks=100):
        """Create 13 ideal patterns with 'elementary' orientations from matching_rows."""

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

        elementary = torch.empty(len(self.cifs), len(matching_rows), top_peaks, 3)  # qxy, qz, intensity
        for idx in range(len(self.cifs)):
            q_list = []
            int_list = []
            for row in matching_rows:
                R = rotate_vect(self.pattern_3d.rec[idx], orientation=row)
                q_3d = self.pattern_3d.q_3d[idx] @ R

                q_2d, intensity, _ = GIWAXS.giwaxs_2d(
                    q_3d=q_3d,
                    intensity=self.pattern_3d.intensities[idx],
                    mi=None,
                    q_range=(self.params.q_xy_max, self.params.q_z_max),
                    wavelength=self.params.wavelength,
                    move_fromMW=False,
                )
                q_2d, intensity = limit_int(q_2d=q_2d.T, intensity=intensity, top_peaks=top_peaks)

                q_2d = torch.tensor(q_2d, dtype=torch.float32, device='cpu')  # (top_peaks, 2)
                q_list.append(q_2d)
                int_list.append(torch.tensor(intensity))

            q_2d_pad = pad_sequence(q_list, batch_first=True, padding_value=0.0)  # (13, top_peaks, 2)
            intensity_pad = pad_sequence(int_list, batch_first=True, padding_value=0.0)  # (13, top_peaks)
            q_tensor = torch.concat((q_2d_pad, intensity_pad.unsqueeze(-1)), dim=-1)  # (13, top_peaks, 3)

            # add padding if the number of peaks is too low
            if q_tensor.shape[1] < top_peaks:
                q_tensor = torch.cat(
                    (q_tensor,
                     torch.zeros(
                         (len(matching_rows),
                          top_peaks - q_tensor.shape[1],
                          3),
                     )), dim=1,
                )

            elementary[idx] = q_tensor  # (13, top_peaks, 3)
        return elementary

    def _create_all_possible_patterns(self):
        """Create all possible patterns (within the orientations range)."""

        full_q_2d = []
        full_intensity_2d = []
        full_q_1d = []
        full_intensity_1d = []

        for idx, cif in enumerate(self.cifs):
            q_2d_list = []
            intensity_list = []
            for orientation in self.pattern_3d.orientations[idx]:
                R = rotate_vect(rec=self.pattern_3d.rec[idx], orientation=orientation)
                q_3d = self.pattern_3d.q_3d[idx] @ R

                q_2d, intensity, _ = GIWAXS.giwaxs_2d(
                    q_3d=q_3d,
                    intensity=self.pattern_3d.intensities[idx],
                    mi=None,
                    q_range=(self.params.q_xy_max, self.params.q_z_max),
                    wavelength=self.params.wavelength,
                    move_fromMW=True,
                )

                # remove peaks with low intensities
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
        return full_q_2d, full_intensity_2d, full_q_1d, full_intensity_1d

    def create_powder3d_pattern(self, idx):
        q_1d = np.linalg.norm(self.pattern_3d.q_3d[idx], axis=-1)
        q_1d, int_1d, _ = GIWAXS.giwaxs_1d(
            q_1d=q_1d,
            intensity=self.pattern_3d.intensities[idx],
            mi=None,
            wavelength=self.params.wavelength,
        )
        return q_1d, int_1d
