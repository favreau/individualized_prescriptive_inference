import argparse
from datetime import datetime
import copy
import os

import joblib
from nilearn import image
import nibabel as nib
import nimfa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import torch
from tqdm import trange, tqdm
from xgboost import XGBClassifier


class ground_truth_splits:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.seed = 42

    def make_kfold_splits(self, whole_ground_truth):
        whole_ground_truth = whole_ground_truth.reset_index(drop=True)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for kf_count, (train_index, test_index) in enumerate(kf.split(whole_ground_truth)):
            ground_truth_train = whole_ground_truth.iloc[train_index].reset_index(drop=True)
            ground_truth_test = whole_ground_truth.iloc[test_index].reset_index(drop=True)

            ground_truth_train = ground_truth_train.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            train_dict = {col: ground_truth_train[col] for col in ground_truth_train.columns}
            test_dict = {col: ground_truth_test[col] for col in ground_truth_test.columns}

            yield pd.DataFrame(train_dict), pd.DataFrame(test_dict), kf_count


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def proportion_lesioned(functional_territory, lesion):
    functional_territory_flat = functional_territory.flatten()
    lesion_flat = lesion.flatten()
    return np.sum(functional_territory_flat * lesion_flat) / np.sum(functional_territory_flat)

def generate_ground_truth_minimal(lesionpath, discopath, functional_parcellation_path, disco_thresh=0.5):
    """Minimal version that only uses functional parcellation (no vascular atlases)"""
    allfiles, alllesionpaths = [], []
    dirs = [x[0] for x in os.walk(lesionpath)]
    files = [x[2] for x in os.walk(lesionpath)]
    for i in range(len(dirs)):
        for file in files[i]:
            if 'nii' in file and not file.startswith('input'):
                allfiles.append(file)
                alllesionpaths.append(os.path.join(dirs[i], file))

    # Extract MNI dimensions for reshaping between 3D and vector representation.
    mni_dim = nib.load(alllesionpaths[0]).get_fdata().shape

    functional_parcellation = nib.load(functional_parcellation_path).get_fdata()
    func_masks = []
    for func in range(1, int(np.max(functional_parcellation) + 1)):
        func_masks.append((functional_parcellation == func).astype(int))

    gt_df = pd.DataFrame()
    for i in tqdm(range(len(allfiles)), desc="Processing lesions"):
        input_representation_list, image_data = [], {}

        if lesionpath and os.path.exists(alllesionpaths[i]):
            orig_lesion = nib.load(alllesionpaths[i]).get_fdata()
            lesion_functions_lesioned = np.zeros(len(func_masks))
            input_representation_list.append('lesion')
            image_data['lesion'] = {}
        else:
            orig_lesion = False

        if discopath and os.path.exists(os.path.join(discopath, allfiles[i])):
            orig_disconnectome = nib.load(os.path.join(discopath, allfiles[i])).get_fdata()
            disco_functions_lesioned = np.zeros(len(func_masks))
            input_representation_list.append('disconnectome')
            image_data['disconnectome'] = {}
        else:
            orig_disconnectome = False

        for j, func_mask in enumerate(func_masks):
            if not isinstance(orig_lesion, bool):
                lesion_functions_lesioned[j] = proportion_lesioned(func_mask, orig_lesion)
            if not isinstance(orig_disconnectome, bool):
                disco_functions_lesioned[j] = proportion_lesioned(func_mask, orig_disconnectome)

        for input_representation in input_representation_list:
            inds = np.where(orig_lesion) if input_representation == 'lesion' else np.where(
                orig_disconnectome > disco_thresh)
            image_data[input_representation]['vol'] = orig_lesion.sum() if input_representation == 'lesion' else (
                        orig_disconnectome > disco_thresh).sum()

            image_data[input_representation]['centroid_x'] = inds[0].mean()
            image_data[input_representation]['centroid_y'] = inds[1].mean()
            image_data[input_representation]['centroid_z'] = inds[2].mean()

            representation = orig_lesion if input_representation == 'lesion' else orig_disconnectome

            dice_scores = {
                'dice_functional_territories': np.zeros(len(func_masks))
            }

            for j, territory in enumerate(func_masks):
                dice_scores['dice_functional_territories'][j] = dice_coef(territory, representation)

            image_data[input_representation]['dice_scores'] = dice_scores

        results_dict = {'filename': allfiles[i]}
        if not isinstance(orig_lesion, bool):
            results_dict['lesion_vol'] = image_data['lesion']['vol']
            results_dict['lesion_centroid_x'] = image_data['lesion']['centroid_x']
            results_dict['lesion_centroid_y'] = image_data['lesion']['centroid_y']
            results_dict['lesion_centroid_z'] = image_data['lesion']['centroid_z']
            results_dict['lesion_functional_territories_dice'] = [
                image_data['lesion']['dice_scores']['dice_functional_territories']]
            results_dict["lesion_functional_distribution"] = [lesion_functions_lesioned]
        if not isinstance(orig_disconnectome, bool):
            results_dict['disco_thresh'] = disco_thresh
            results_dict['disco_vol'] = image_data['disconnectome']['vol']
            results_dict['disco_centroid_x'] = image_data['disconnectome']['centroid_x']
            results_dict['disco_centroid_y'] = image_data['disconnectome']['centroid_y']
            results_dict['disco_centroid_z'] = image_data['disconnectome']['centroid_z']
            results_dict['disco_functional_territories_dice'] = [
                image_data['disconnectome']['dice_scores']['dice_functional_territories']]
            results_dict["disco_functional_distribution"] = [disco_functions_lesioned]

        gt_df = pd.concat([gt_df, pd.DataFrame(results_dict, index=[i])])

    return gt_df


class ground_truth_preprocess:
    def __init__(self, n_splits, dims, input_types, lesionpath, discopath, savepath, icvpath, mni_dim=[91, 109, 91], device="cuda:0"):
        self.n_splits = n_splits
        self.dims = dims
        self.icvpath = icvpath
        self.lesionpath = lesionpath
        self.discopath = discopath
        self.savepath = savepath
        self.mni_dim = mni_dim
        self.input_types = input_types
        self.device = device
        self.seed = 42

    def intracranial_volume_mask(self, arr):
        mask = nib.load(self.icvpath).get_fdata().flatten().astype(bool)
        if len(mask) == arr.shape[1]:
            return arr[:, mask]
        else:
            raise ValueError("mask different dimension to array")

    def intracranial_volume_unmask(self, arr):
        mask = nib.load(self.icvpath).get_fdata().flatten().astype(bool)
        mask_inds = np.where(mask)[0]
        unmasked = np.zeros([arr.shape[0], mask.shape[0]])
        for i in range(len(mask_inds)):
            unmasked[:, mask_inds[i]] = arr[:, i]
        return unmasked.reshape(arr.shape[0], self.mni_dim[0], self.mni_dim[1], self.mni_dim[2])

    def reduce_nimfa_nmf(self, train, test, dim, input_type):
        arr_all = np.zeros((len(train), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2]))
        for i in range(len(train)):
            filename = train['filename'].iloc[i]
            img = nib.load(
                os.path.join(self.lesionpath, filename)).get_fdata() if input_type == 'lesion' else nib.load(
                os.path.join(self.discopath, filename)).get_fdata()
            arr_all[i, :] = img.flatten()

        nmf = nimfa.Nmf(V=self.intracranial_volume_mask(arr_all), seed='random_vcol', rank=dim)
        nmf_fitted = nmf()

        arr_test = np.zeros((len(test), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2]))
        for i in range(len(test)):
            filename = test['filename'].iloc[i]
            img = nib.load(
                os.path.join(self.lesionpath, filename)).get_fdata() if input_type == 'lesion' else nib.load(
                os.path.join(self.discopath, filename)).get_fdata()
            arr_test[i, :] = img.flatten()

        return np.array(nmf_fitted.fit.W), np.array(np.matmul(self.intracranial_volume_mask(arr_test), np.linalg.pinv(nmf_fitted.fit.H))), np.array(nmf_fitted.fit.H)

    def reduce_pca(self, train, test, dim, input_type):
        arr_all = np.zeros((len(train), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2]))
        for i in range(len(train)):
            filename = train['filename'].iloc[i]
            img = nib.load(
                os.path.join(self.lesionpath, filename)).get_fdata() if input_type == 'lesion' else nib.load(
                os.path.join(self.discopath, filename)).get_fdata()
            arr_all[i, :] = img.flatten()

        pca = PCA(n_components=dim)
        pca.fit(self.intracranial_volume_mask(arr_all))

        arr_test = np.zeros((len(test), self.mni_dim[0] * self.mni_dim[1] * self.mni_dim[2]))
        for i in range(len(test)):
            filename = test['filename'].iloc[i]
            img = nib.load(
                os.path.join(self.lesionpath, filename)).get_fdata() if input_type == 'lesion' else nib.load(
                os.path.join(self.discopath, filename)).get_fdata()
            arr_test[i, :] = img.flatten()

        return pca.transform(self.intracranial_volume_mask(arr_all)), pca.transform(self.intracranial_volume_mask(arr_test)), pca


    def reduce(self, train, test, k, dim, nmf=True, pca=True):

        N_training = len(train)
        train_dict = {col: train[col] for col in train.columns}
        test_dict = {col: test[col] for col in test.columns}

        for input_type in self.input_types:
            sub_savepath = os.path.join(self.savepath, f"K{k}_dim{dim}_{input_type}_N{N_training}")
            if os.path.exists(sub_savepath):
                now = datetime.now()
                sub_savepath = os.path.join(sub_savepath, "forcompletion_" + now.strftime("%H_%M_%S__%d_%m_%Y"))
                os.makedirs(sub_savepath)
            else:
                os.makedirs(sub_savepath)

            if nmf:
                sub_savepath_nmf = os.path.join(sub_savepath, "nmf")
                if not os.path.exists(sub_savepath_nmf):
                    os.makedirs(sub_savepath_nmf)

                print(f"Running NMF for {input_type} with {dim} components...")
                train_embeddings, test_embeddings, archetypes = self.reduce_nimfa_nmf(train, test, dim, input_type)

                train_dict = {**train_dict, **{f"nmf_{input_type}_{dim}_K{k}": list(train_embeddings)}}
                test_dict = {**test_dict, **{f"nmf_{input_type}_{dim}_K{k}": list(test_embeddings)}}

            if pca:
                sub_savepath_pca = os.path.join(sub_savepath, "pca")
                if not os.path.exists(sub_savepath_pca):
                    os.makedirs(sub_savepath_pca)

                print(f"Running PCA for {input_type} with {dim} components...")
                train_embeddings, test_embeddings, pca_fitted = self.reduce_pca(train, test, dim, input_type)
                joblib.dump(pca_fitted.explained_variance_, os.path.join(sub_savepath_pca, f"pca_explained_variance_K{k}_{input_type}_dim{dim}_N{N_training}.joblib"))
                joblib.dump(pca_fitted.explained_variance_ratio_, os.path.join(sub_savepath_pca, f"pca_explained_variance_ratio_K{k}_{input_type}_dim{dim}_N{N_training}.joblib"))

                train_dict = {**train_dict, **{f"pca_{input_type}_{dim}_K{k}": list(train_embeddings)}}
                test_dict = {**test_dict, **{f"pca_{input_type}_{dim}_K{k}": list(test_embeddings)}}

        return pd.DataFrame(train_dict), pd.DataFrame(test_dict)


def command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lesionpath", type=str, default="", help="Path to lesion mask nii files")
    parser.add_argument("--discopath", type=str, default="", help="Path to disconnectome nii files")
    parser.add_argument("--savepath", type=str, default="", help="Path to save results")
    parser.add_argument("--kfolds", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--run_nmf", type=bool, default=True, help="Run non-negative matrix factorisation")
    parser.add_argument("--run_pca", type=bool, default=True, help="Run principal component analysis")
    parser.add_argument("--latent_components", type=int, nargs='+', default=[50], help="Number of latent components")

    args = parser.parse_args()
    paths = (args.lesionpath, args.discopath, args.savepath)
    reductions = (args.run_nmf, args.run_pca)
    return (paths, reductions, args.kfolds, args.latent_components)

def run(parameters):
    paths, reductions, kfolds, latent_components = parameters
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lesionpath, discopath, savepath = paths
    run_nmf, run_pca = reductions

    if not savepath:
        savepath = os.path.join(os.getcwd(), "results")
        print(f"Saving results to {savepath}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if not lesionpath and not discopath:
        raise ValueError("Please provide a path to the lesion masks and/or disconnectomes")
    atlases_path = os.path.join(os.getcwd(), "atlases")

    functional_parcellation_path = os.path.join(atlases_path, "functional_parcellation_2mm.nii.gz")

    if os.path.exists(os.path.join(savepath, 'whole_ground_truth.pkl')):
        print("Loading existing whole_ground_truth.pkl...")
        whole_ground_truth = pd.read_pickle(os.path.join(savepath, 'whole_ground_truth.pkl'))
    else:
        print("Generating ground truth data...")
        whole_ground_truth = generate_ground_truth_minimal(lesionpath=lesionpath,
                                                           discopath=discopath,
                                                           functional_parcellation_path=functional_parcellation_path)

        whole_ground_truth.to_pickle(os.path.join(savepath, 'whole_ground_truth.pkl'))
        print(f"Saved whole_ground_truth.pkl with {len(whole_ground_truth)} samples")

    input_types = []
    if lesionpath: input_types.append(["lesion"])
    if discopath: input_types.append(["disco"])
    
    gt_generator = ground_truth_splits(n_splits=kfolds)
    for ground_truth_train, ground_truth_test, kf_count in gt_generator.make_kfold_splits(whole_ground_truth):
        print(f"\n=== Processing fold {kf_count+1}/{kfolds} ===")
        ground_truth_train.to_pickle(os.path.join(savepath, f"train_split_{kf_count}.pkl"))
        ground_truth_test.to_pickle(os.path.join(savepath, f"test_split_{kf_count}.pkl"))

        for dim in latent_components:
            for input_type in input_types:
                gt_generator = ground_truth_preprocess(n_splits=kfolds, dims=int(dim), input_types=input_type,
                                                       lesionpath=lesionpath, discopath=discopath, savepath=savepath,
                                                       icvpath=os.path.join(atlases_path, "icv_mask_2mm.nii.gz"),
                                                       device=device)

                train_reduced, test_reduced = gt_generator.reduce(ground_truth_train, ground_truth_test,
                                                                  k=kf_count, dim=int(dim),
                                                                  nmf=run_nmf, pca=run_pca)
                train_reduced.to_pickle(os.path.join(savepath, f"train_{kf_count}_dim_{dim}_{input_type}.pkl"))
                test_reduced.to_pickle(os.path.join(savepath, f"test_{kf_count}_dim_{dim}_{input_type}.pkl"))
                print(f"Saved train/test files for fold {kf_count}, dim {dim}, input {input_type}")

    print("\n=== Representation pipeline complete! ===")

if __name__ == "__main__":
    parameters = command_line_options()
    run(parameters)

