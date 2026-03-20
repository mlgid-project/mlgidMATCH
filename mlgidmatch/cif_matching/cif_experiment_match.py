import torch
import numpy as np
from typing import Tuple

from mlgidmatch.cif_matching.utils import generate_images
from mlgidmatch.cif_matching.utils import ExpConfig


class Match_CIF():
    """
    A class to make Neural Matching for GID patterns with CIFs.

    Attributes
    ----------
    config : TestExpConfig
        Configuration of the test experiment.
    """

    def __init__(self,
                 config: ExpConfig,
                 ):
        self.config = config

    def match(self,
              peak_list: np.ndarray,  # (peaks_num, 2)
              q_range: Tuple[float, float],
              candidate_ind: np.ndarray,
              batch_size: int = 128,
              device='cuda',
              ) -> np.ndarray:
        """
        Return probabilities for the candidate structures.

        Parameters
        ----------
        peak_list : np.ndarray
            Shape (peaks_num, 2). Peak list (q_xy, q_z).
        q_range : Tuple[float, float]
            Upper limits of q-range (for q_xy, q_z).
        candidate_ind : np.ndarray
            Indices of candidate structures (corresponding to self.config.cif_prepr.cifs list).
        batch_size : int
        device : str or torch.device

        Returns
        -------
        probabilities_full : np.ndarray
            Probabilities for the candidate structures.
        """

        q_range = torch.tensor(q_range, dtype=torch.float32).to(device)
        assert (len(candidate_ind) == len(set(candidate_ind))), f"duplicates in {candidate_ind}"
        probabilities_full = np.empty(len(candidate_ind))

        for batch in range(0, len(candidate_ind), batch_size):
            elementary_input = self.config.cif_prepr.elementary[candidate_ind][batch:batch + batch_size].to(device)
            elementary_img = [generate_images(
                q_2d=elementary_input[i, :, :, :2],
                q_range=q_range.unsqueeze(0).repeat((13, 1)),
                intensities=elementary_input[i, :, :, -1],
                settings_dict=self.config.settings_dict,
            ) for i in range(len(elementary_input))]

            elementary_img = torch.stack(elementary_img)  # (cifs_num, 13, 128, 128)

            images = generate_images(
                q_2d=torch.tensor(peak_list, dtype=torch.float32).unsqueeze(0).to(device),
                q_range=q_range.unsqueeze(0),
                intensities=None,
                settings_dict=self.config.settings_dict,
            ).unsqueeze(1)  # (1, 1, 128, 128)
            input_batch = torch.concatenate(
                (images.repeat(len(elementary_img), 1, 1, 1),
                 elementary_img), dim=1,
            )  # (cifs_num, 14, 128, 128)

            self.config.model.eval()
            with torch.no_grad():
                outputs = self.config.model(input_batch).squeeze(1)  # (cifs_num,)
            probabilities = torch.sigmoid(outputs).cpu()  # (cifs_num,)
            probabilities_full[batch:batch + batch_size] = probabilities.numpy()

        return probabilities_full
