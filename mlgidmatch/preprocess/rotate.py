import numpy as np
from typing import Union, Tuple
from mlgidmatch.preprocess.directions import get_unique_directions


def rotate_vect(rec,
                orientation: Union[str, np.ndarray] = np.array([0., 0., 1.], dtype=np.float32),
                baz: np.ndarray = np.array([0., 0., 1.], dtype=np.float32),
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate crystal

    Parameters
    ----------
        rec : np.ndarray, shape (3, 3)
            reciprocal vectors
        orientation : Union[str, np.ndarray]
            crystallographic orientation, e.g. 'random' or np.array([0., 1., 0.]), default=[001]
        baz : np.ndarray, optional
            basis vector for the default orientation, default=001

    Return
    -------
        R : np.ndarray, shape (3, 3)
            rotation matrix
        orientation : np.ndarray, shape (3, )
    """
    if isinstance(orientation, np.ndarray) and orientation.shape == (3,):
        orientation = orientation / np.linalg.norm(orientation, axis=0)
    else:
        raise TypeError("orientation is not correct - use np.array with size (3,)")

    if np.array_equal(baz, orientation):
        R = np.array(
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]],
        )
        return R, orientation

    orient = orientation @ rec

    v1 = orient / np.linalg.norm(orient, axis=0)
    v2 = baz / np.linalg.norm(baz, axis=0)
    _n = np.cross(v1, v2)
    if np.array_equal(_n, np.array([0., 0., 0.])):
        _n = baz
    n = _n / np.linalg.norm(_n, axis=0)

    cos_phi = v1 @ v2
    sin_phi = np.sqrt(1 - cos_phi ** 2)

    a_1 = np.stack(
        (n[..., 0] ** 2 * (1 - cos_phi) + cos_phi,
         n[..., 0] * n[..., 1] * (1 - cos_phi) + n[..., 2] * sin_phi,
         n[..., 0] * n[..., 2] * (1 - cos_phi) - n[..., 1] * sin_phi),
    )

    a_2 = np.stack(
        (n[..., 0] * n[..., 1] * (1 - cos_phi) - n[..., 2] * sin_phi,
         n[..., 1] ** 2 * (1 - cos_phi) + cos_phi,
         n[..., 1] * n[..., 2] * (1 - cos_phi) + n[..., 0] * sin_phi),
    )

    a_3 = np.stack(
        (n[..., 0] * n[..., 2] * (1 - cos_phi) + n[..., 1] * sin_phi,
         n[..., 1] * n[..., 2] * (1 - cos_phi) - n[..., 0] * sin_phi,
         n[..., 2] ** 2 * (1 - cos_phi) + cos_phi),
    )

    R = np.stack((a_1, a_2, a_3), dtype=np.float32)

    return R, orientation
