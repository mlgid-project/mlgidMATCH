from typing import cast, Tuple, Sequence
import itertools
from functools import reduce
import numpy as np
import math

from pymatgen.util.coord import in_coord_list
from pymatgen.core import SymmOp


def get_unique_directions(max_index: int) -> np.ndarray:
    """Generate possible crystallographic orientations within max_index."""
    mi_list = _get_all_directions(max_index)
    symm_ops = _get_default_symm_op()
    mi_unique = []
    for idx, mi in enumerate(mi_list):
        denom = abs(reduce(math.gcd, mi))  # type: ignore[arg-type]
        mi = cast("tuple[int, int, int]", tuple(int(idx / denom) for idx in mi))
        if not _is_in_miller_family(mi, mi_unique, symm_ops):
            mi_unique.append(mi)
    return np.array(mi_unique, dtype=np.float32)


def _get_all_directions(max_index: int):
    rng = list(range(-max_index, max_index + 1))[::-1]
    hkl_list = [miller for miller in itertools.product(rng, rng, rng) if any(i != 0 for i in miller)]
    return hkl_list


def _get_default_symm_op(tol=0.1):
    aff_1 = np.eye(4)
    aff_2 = np.eye(4) * -1
    aff_2[-1, -1] = 1

    symm_ops = [
        SymmOp(
            affine_transformation_matrix=aff_1,
            tol=tol,
        ),
        SymmOp(
            affine_transformation_matrix=aff_2,
            tol=tol,
        ),
    ]
    return symm_ops


def _is_in_miller_family(
        miller_index: Tuple[int, int, int],
        miller_list: Sequence[Tuple[int, int, int]],
        symm_ops: list,
) -> bool:
    """
    Function from the pymatgen package (https://github.com/materialsproject/pymatgen)
    to check if the given Miller index belongs to the same family of any index in the provided list.

    Parameters
    ----------
    miller_index : Tuple[int, int, int]
    miller_list : List
        List of Miller indices to compare.
    symm_ops : List
        Symmetry operations for a lattice,
        used to define the indices family.
    """
    return any(in_coord_list(miller_list, op.operate(miller_index)) for op in symm_ops)
