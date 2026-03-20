from typing import List, Tuple, Any, Union
from dataclasses import dataclass, field
import torch

from mlgidmatch.preprocess.cif_preprocess import CifPattern


@dataclass
class ExpConfig:
    cif_prepr: CifPattern
    model: Any = None
    settings_dict: dict = field(
        default_factory=lambda: {
            'image_size': 224,
        },
    )


def generate_images(
        q_2d: torch.Tensor,  # (str_num, peaks_num, 2)
        q_range: torch.Tensor,  # (str_num, 2)
        intensities: Union[torch.Tensor, None],  # (str_num, peaks_num)
        settings_dict: dict,
) -> torch.Tensor:
    """
    Generate 'pseudo-experimental' images.

    Parameters
    ----------
    q_2d : torch.Tensor
        Shape (str_num, peaks_num, 2). Tensor with peaks (q_xy, q_z).
        Preferably on CUDA.
    q_range : torch.Tensor
        Shape (str_num, 2). Upper limits of q-range (for q_xy, q_z).
    intensities : torch.Tensor or None
        Shape (str_num, peaks_num). Intensities corresponding to q_2d peaks.
    settings_dict : dict
        Dictionary of settings for the image generation.

    Returns
    -------
    images : torch.Tensor
    """

    image_size = settings_dict.get('image_size', 224)

    if intensities is not None:
        nonzero_mask = (intensities != 0)
        intensities = torch.where(
            nonzero_mask,
            (intensities / intensities.max(axis=1, keepdims=True).values) ** (1 / 3),
            torch.tensor(0.0),
        )
    else:
        nonzero_mask = (q_2d != -1)
        nonzero_mask = nonzero_mask.any(axis=2)
        intensities = nonzero_mask.to(torch.int32)

    images = torch.zeros((q_2d.shape[0], image_size, image_size), device=q_2d.device, dtype=intensities.dtype)

    normalized_points = (q_2d * image_size / q_range.unsqueeze(1).to(q_2d.device)).round().long()
    x, y = normalized_points[..., 0], normalized_points[..., 1]

    x = torch.clamp(x, 0, image_size - 1)
    y = torch.clamp(y, 0, image_size - 1)

    images[torch.arange(q_2d.shape[0]).unsqueeze(1), y, x] = intensities

    return images
