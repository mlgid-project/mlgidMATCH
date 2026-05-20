import pytest
from pathlib import Path
from pygidsim.experiment import ExpParameters
from mlgidmatch.cif_matching.models.ResNet import IMGClassifier
from mlgidmatch.preprocess.cif_preprocess import CifPattern
import numpy as np
import torch


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def cif_folder(test_data_dir):
    """Return path to the test CIF folder."""
    return test_data_dir.joinpath('test_data')


@pytest.fixture(scope="session")
def cif_files(cif_folder):
    """Return path to the test CIF file."""
    cif_path_1 = cif_folder.joinpath('1_BA2PbI4_n1.cif')
    cif_path_2 = cif_folder.joinpath('Bn-Br_test.cif')
    if not cif_path_1.exists():
        pytest.skip(f"CIF file not found: {cif_path_1}")
    if not cif_path_2.exists():
        pytest.skip(f"CIF file not found: {cif_path_1}")
    return [cif_path_1, cif_path_2]


@pytest.fixture(scope="session")
def exp_parameters():
    """Return a standard ExpParameters instance for testing."""
    return ExpParameters(
        q_xy_max=3.0,
        q_z_max=3.0,
        ai=0.3,
        en=18_000,
    )


@pytest.fixture(scope="session")
def orientations():
    """Return some orientations for testing."""
    return np.array(
        [[0., 0., 1.],
         [1., 0., 1.]],
    )


@pytest.fixture(scope="session")
def cif_class(exp_parameters, cif_folder, cif_files):
    return CifPattern(
        params=exp_parameters,
        folder_path=cif_folder,
        cifs=[cif.name for cif in cif_files],
        create_all=True,
    )


@pytest.fixture
def model(test_data_dir):
    """Return a model."""
    model_path = test_data_dir.parent.joinpath(
        'mlgidmatch',
        'cif_matching',
        'models',
        'ResNet18_best_model.pt',
    )
    model = IMGClassifier(input_dim=14, output_dim=1, res=18).eval()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model
