import pytest
from pygidsim.giwaxs_sim import GIWAXSFromCif
import numpy as np
from mlgidmatch.orient_matching.orient_experiment_match import Match_Orient
from mlgidmatch.cif_matching.utils import ExpConfig


@pytest.mark.run(order=3)
class TestMatchOrient:
    @pytest.fixture
    def config(self, model, cif_class):
        model.eval()
        return ExpConfig(
            cif_prepr=cif_class,
            model=model,
        )

    @pytest.fixture
    def real_pattern(self, cif_files, orientations, exp_parameters):
        q_list = []
        int_list = []
        lengths = []
        for cif_file, orient in zip(cif_files, orientations):
            el = GIWAXSFromCif(str(cif_file), exp_parameters)
            q_2d, intensities = el.giwaxs.giwaxs_sim(
                orientation=orient,
                return_mi=False,
                move_fromMW=True,
            )
            q_list.append(q_2d)
            int_list.append(intensities)
            lengths.append(len(intensities))
        return np.concatenate(q_list, axis=1).T, np.concatenate(int_list, axis=0), lengths

    def test_match_orient(self, config, real_pattern, orientations, exp_parameters):
        q_2d_real, int_real, lengths = real_pattern
        assert q_2d_real.shape[1] == 2, f"Incorrect q_2d_real shape: {q_2d_real.shape[1]}"
        assert q_2d_real.shape[0] == len(int_real) == sum(lengths), \
            f"Incorrect lengths: {q_2d_real.shape[0]}, {len(int_real)}, {sum(lengths)}"

        matcher = Match_Orient(config)
        data_match = matcher.match(
            q_real_all=q_2d_real,
            intens_real_all=int_real,
            probs=np.array([0.95, 0.95]),
            q_range=(exp_parameters.q_xy_range[1], exp_parameters.q_z_range[1]),
            peaks_indices=np.arange(len(q_2d_real)),
            candidate_ind=np.arange(len(config.cif_prepr.cifs)),
            threshold=0.5,
            save_metrics=True,
        )
        assert len(data_match.keys()) == len(config.cif_prepr.cifs)

        len_sum = 0
        for i in range(len(config.cif_prepr.cifs)):
            orient = data_match[str(i)]['orient']
            matched_len = len(data_match[str(i)]['indices_real_matched_all'])
            len_sum += matched_len
            assert matched_len >= min(lengths)
            assert np.any(np.all(orient == orientations, axis=1)), f"bad orientation: {orient}"
        assert len_sum >= sum(lengths), f"low number of matched peaks: {len_sum}, {sum(lengths)}"

        for i in range(len(config.cif_prepr.cifs)):
            len_sum += len(data_match[str(i)]['indices_real_matched_all'])
        assert len_sum >= sum(lengths), f"low number of matched peaks: {len_sum}, {sum(lengths)}"
