import pytest
from pygidsim.giwaxs_sim import GIWAXSFromCif
import numpy as np
import torch
from mlgidmatch.cif_matching.cif_experiment_match import Match_CIF
from mlgidmatch.cif_matching.utils import ExpConfig


@pytest.mark.run(order=2)
class TestMatchCif:
    @pytest.fixture
    def q_2d_full(self, cif_files, orientations, exp_parameters):
        q_list = []
        for cif_file, orient in zip(cif_files, orientations):
            el = GIWAXSFromCif(str(cif_file), exp_parameters)
            q_2d, _ = el.giwaxs.giwaxs_sim(
                orientation=orient,
                return_mi=False,
                move_fromMW=True,
            )
            q_list.append(q_2d)
        return np.concatenate(q_list, axis=1).T

    @pytest.mark.parametrize(
        "device", [
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA not available",
                ),
            ),
            torch.device("cpu")
        ],
    )
    def test_match_cif(self, model, cif_class, q_2d_full, exp_parameters, device):
        assert q_2d_full.shape[1] == 2, "Incorrect shape"

        model.eval().to(device)
        config = ExpConfig(
            cif_prepr=cif_class,
            model=model,
        )
        matcher = Match_CIF(config)
        probs = matcher.match(
            peak_list=q_2d_full,
            q_range=(exp_parameters.q_xy_range[1], exp_parameters.q_z_range[1]),
            candidate_ind=np.arange(len(cif_class.cifs)),
            batch_size=128,
            device=device,
        )
        assert len(probs) == q_2d_full.shape[1], "length mismatch"
        assert (probs > 0.5).all(), f"bad predictions: {probs}"
