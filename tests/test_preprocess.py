import pytest
from pygidsim.giwaxs_sim import GIWAXSFromCif


@pytest.mark.run(order=1)
class TestPreprocess:

    def test_pygidsim(self, cif_files, exp_parameters):

        for cif in cif_files:
            el = GIWAXSFromCif(str(cif), exp_parameters)
            q_2d, intensity_2d = el.giwaxs.giwaxs_sim(
                orientation='random',
                move_fromMW=True,
            )
            assert q_2d.shape[1] == len(intensity_2d), "Shapes mismatch"
            assert len(intensity_2d) > 0, "No intensity data returned"

    def test_preprocess(self, cif_class, cif_files):

        cifs = list(cif_class.cifs or [])
        all_patterns_q2d = cif_class.all_patterns_q2d or []
        all_patterns_int2d = cif_class.all_patterns_int2d or []
        all_patterns_q1d = cif_class.all_patterns_q1d or []
        all_patterns_int1d = cif_class.all_patterns_int1d or []
        orientations = cif_class.pattern_3d.orientations

        cifs_num = len(cif_files)
        assert len(cifs) == cifs_num, "Number of cifs does not match"
        assert len(all_patterns_q2d) == cifs_num, "length of all_patterns_q2d is not correct"
        assert len(all_patterns_int2d) == cifs_num, "length of all_patterns_int2d is not correct"
        assert len(all_patterns_q1d) == cifs_num, "length of all_patterns_q1d is not correct"
        assert len(all_patterns_int1d) == cifs_num, "length of all_patterns_int1d is not correct"

        for q, intens, orients in zip(
                all_patterns_q2d, all_patterns_int2d, orientations,
        ):
            assert len(q) == len(intens) == len(orients), "Shapes mismatch"
        for q, intens in zip(all_patterns_q1d, all_patterns_int1d):
            assert q.shape[0] == len(intens), "Shapes mismatch"
