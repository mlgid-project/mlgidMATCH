from mlgidmatch.preprocess.cif_preprocess import Pattern3d, CifPattern
import pickle
import os
import h5py
import numpy as np
from mlgidmatch.matching import Match


def test_patterns(
        match_class: Match,
        data_folder: str,
        sample_type: str,
        save_folder: str,
        peaks_type='segments',
        save_metrics=False,
):
    h5_file_list = []
    if sample_type == 'perovskites':
        h5_file_list = os.scandir(os.path.join(data_folder, 'perovskites/h5/11_06_2025'))
    elif sample_type == 'perovskites_new':
        h5_file_list = os.scandir(os.path.join(data_folder, 'perovskites/h5/18_09_2025'))
    elif sample_type == 'organic':
        h5_file_list = os.scandir(os.path.join(data_folder, 'organic/h5/27_06_2025'))
    elif sample_type == 'in_situ':
        h5_file_list = os.scandir(os.path.join(data_folder, 'in_situ/h5/01_09_2025'))
    elif sample_type == 'powder':
        h5_file_list = os.scandir(os.path.join(data_folder, 'powder3d/h5/04_07_2025'))
    h5_file_list = [h5_file for h5_file in h5_file_list if (h5_file.is_file() and h5_file.name.endswith('.h5'))]
    assert len(h5_file_list) > 0
    for h5_file in h5_file_list:
        print(h5_file.name)
        with h5py.File(h5_file, 'r') as f:
            for entry in f.keys():
                q_xy_max = f[entry]['data']['q_xy'][()][-1]
                q_z_max = f[entry]['data']['q_z'][()][-1]
                measurements = []
                peak_list = []
                intens_real_list = []
                q_range_list = []
                for meas in f[entry]['data']['analysis'].keys():
                    measurements.append(meas)
                    try:
                        xy_fitted = f[entry]['data']['analysis'][meas]['fitted_peaks']['q_xy'][()]
                        z_fitted = f[entry]['data']['analysis'][meas]['fitted_peaks']['q_z'][()]
                    except:
                        xy_fitted = f[entry]['data']['analysis'][meas]['fitted_peaks']['peaks_qxy'][()]
                        z_fitted = f[entry]['data']['analysis'][meas]['fitted_peaks']['peaks_qz'][()]
                    try:
                        intens_real = f[entry]['data']['analysis'][meas]['fitted_peaks']['amplitude'][()]
                    except:
                        intens_real = f[entry]['data']['analysis'][meas]['fitted_peaks']['amplitudes'][()]
                    try:
                        types = f[entry]['data']['analysis'][meas]['fitted_peaks']['type'][()]
                        if peaks_type == 'segments':
                            xy_fitted = xy_fitted[np.where(types == 2)]
                            z_fitted = z_fitted[np.where(types == 2)]
                            intens_real = intens_real[np.where(types == 2)]
                        elif peaks_type == 'rings':
                            xy_fitted = xy_fitted[np.where(types == 1)]
                            z_fitted = z_fitted[np.where(types == 1)]
                            intens_real = intens_real[np.where(types == 1)]
                    except:
                        is_ring = f[entry]['data']['analysis'][meas]['fitted_peaks']['is_ring'][()]
                        if peaks_type == 'segments':
                            xy_fitted = xy_fitted[~is_ring]
                            z_fitted = z_fitted[~is_ring]
                            intens_real = intens_real[~is_ring]
                        elif peaks_type == 'rings':
                            xy_fitted = xy_fitted[is_ring]
                            z_fitted = z_fitted[is_ring]
                            intens_real = intens_real[is_ring]

                    q_2d_real = np.stack((xy_fitted, z_fitted)).T
                    peak_list.append(q_2d_real)
                    intens_real_list.append(intens_real)
                    q_range_list.append(q_range_list)

                data_matched = match_class.match_all(
                    measurements=measurements,
                    peak_list=peak_list,
                    intensities_real_list=intens_real_list,
                    q_range_list=[(q_xy_max, q_z_max) for _ in measurements],
                    threshold=0.5,
                    candidates_list=None,
                    peaks_type=peaks_type,
                    save_metrics=save_metrics,
                )

                save_folder_full = os.path.join(save_folder, f'{sample_type}')
                if not os.path.exists(save_folder_full):
                    os.mkdir(save_folder_full)
                save_folder_full = os.path.join(save_folder_full, f'{os.path.basename(h5_file).split(".")[0]}')
                if not os.path.exists(save_folder_full):
                    os.mkdir(save_folder_full)
                with h5py.File(
                        os.path.join(save_folder_full, f'{peaks_type}_results_all.h5'), 'a',
                ) as h5_results:
                    save_dict_to_hdf5({entry: data_matched}, h5_results)


def save_dict_to_hdf5(dic, group):
    for key, item in dic.items():
        if isinstance(item, dict):
            subgroup = group.create_group(key)
            save_dict_to_hdf5(item, subgroup)
        else:
            group.create_dataset(key, data=item)


if __name__ == "__main__":
    data_folder = '/data/romodin/gi_matching/dataset/experiment/'
    # with open(os.path.join(data_folder, 'prepr_cifs.pickle'), 'rb') as file:  # _old_version
    #     cif_cl = pickle.load(file)

    with open(os.path.join(data_folder, 'prepr_cifs_LC.pickle'), 'rb') as file:  # _old_version
        cif_cl = pickle.load(file)

    model_path = '/home/romodin/Romodos/Packages/mlgidMATCH/mlgidmatch/cif_matching/models/ResNet18_actual_LC_best_state_dict.pt'
    # model_path = '/home/romodin/Romodos/Packages/mlgidMATCH/mlgidmatch/cif_matching/models/ResNet18_actual_state74999.pt'
    # model_path = '/home/romodin/Romodos/Packages/mlgidMATCH/mlgidmatch/cif_matching/models/ResNet18_newimage_14ch_state99999.pt'
    match_class = Match(
        cif_class=cif_cl,
        model_path=model_path,
        device='cuda',
    )

    save_folder = '/home/romodin/Romodos/Packages/Romodin_GIWAXS/gi_matching/Results/desy_output/cifs/ResNet18_actual_LC/val/experiment/'
    # save_folder = '/home/romodin/Romodos/Packages/Romodin_GIWAXS/gi_matching/Results/desy_output/cifs/ResNet18_actual/val/experiment_74999/'
    # save_folder = '/home/romodin/Romodos/Packages/Romodin_GIWAXS/gi_matching/Results/desy_output/cifs/ResNet18_newimage/val/experiment/'
    from datetime import datetime

    start = datetime.now()
    test_patterns(
        match_class=match_class,
        data_folder=data_folder,
        sample_type='perovskites',
        save_folder=save_folder,
        peaks_type='segments',
        save_metrics=True,
    )
    print('TRUE:', datetime.now() - start)

    start = datetime.now()
    test_patterns(
        match_class=match_class,
        data_folder=data_folder,
        sample_type='perovskites',
        save_folder=save_folder,
        peaks_type='segments',
        save_metrics=False,
    )
    print("FALSE:", datetime.now() - start)

    # unique_solutions = match_class.unique_solutions(data_matched)
