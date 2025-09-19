import torch
import numpy as np
from typing import List, Tuple, Union

from mlgidmatch.cif_matching.utils import generate_images
from mlgidmatch.cif_matching.utils import ExpConfig


class Match_CIF():
    """
        A class to match CIFs with the GID patterns.
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
              peak_list: np.ndarray,  # (peaks_num, 2)
              q_range: Tuple[float, float],
              candidate_ind: np.ndarray,
              batch_size: int = 128,
              device='cuda',
              ):
        q_range = torch.tensor(q_range, dtype=torch.float32).to(device)
        assert (len(candidate_ind) == len(set(candidate_ind))), f"duplicates in {candidate_ind}"
        probabilities_full = np.empty(len(candidate_ind))

        for batch in range(0, len(candidate_ind), batch_size):
            background_input = self.config.cif_class.background[candidate_ind][batch:batch + batch_size].to(device)
            background_img = [generate_images(
                q_2d=background_input[i, :, :, :2],
                q_range=q_range.unsqueeze(0).repeat((13, 1)),
                intensities=background_input[i, :, :, -1],
                settings_dict=self.config.settings_dict,
            )
                for i in range(len(background_input))]

            background_img = torch.stack(background_img)  # (cifs_num, 13, 128, 128)

            images = generate_images(
                q_2d=torch.tensor(peak_list, dtype=torch.float32).unsqueeze(0).to(device),
                q_range=q_range.unsqueeze(0),
                intensities=None,
                settings_dict=self.config.settings_dict,
            ).unsqueeze(1)  # torch.Size([1, 1, 128, 128])
            input_batch = torch.concatenate(
                (images.repeat(len(background_img), 1, 1, 1),
                 background_img), dim=1,
            )  # torch.Size([cifs_num, 14, 128, 128])

            self.config.model.eval()
            with torch.no_grad():
                outputs = self.config.model(input_batch).squeeze(1)  # torch.Size([cifs_num])
            probabilities = torch.sigmoid(outputs).cpu()  # torch.Size([cifs_num])
            probabilities_full[batch:batch + batch_size] = probabilities.numpy()

        return probabilities_full
