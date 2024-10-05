from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KDTree
import torch
import os
from tqdm import tqdm
from pqdm.processes import pqdm
import pickle
import time
import faiss
import json
import argparse
import pprint
from uqocp.utils.load_data import load_is2re_data
from uqocp.utils.uncertainty_evaluation import log_likelihood
from scipy.optimize import minimize
pp = pprint.PrettyPrinter(indent=4)


class ConformalPrediction:
    """
    Performs quantile regression on score functions to obtain the estimated qhat
        on calibration data and apply to test data during prediction.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, residuals_calib, heurestic_uncertainty_calib) -> None:
        # score function
        scores = abs(residuals_calib / heurestic_uncertainty_calib)
        scores = np.array(scores)

        n = len(residuals_calib)
        qhat = torch.quantile(
            torch.from_numpy(scores), np.ceil((n + 1) * (1 - self.alpha)) / n
        )
        qhat_value = np.float64(qhat.numpy())
        self.qhat = qhat_value
        pass

    def predict(self, heurestic_uncertainty_test):
        cp_uncertainty_test = heurestic_uncertainty_test * self.qhat
        return cp_uncertainty_test, self.qhat

class FlexibleNLL:
    """
    Minimizes the NLL over the validation using a flexible non-linear fit
    """
    def __init__(self):
        self.p0 = 0
        self.p1 = 1
    
    def fit(self, residuals_calib, heurestic_uncertainty_calib) -> None:
        unc = np.array(heurestic_uncertainty_calib)
        err = np.array(residuals_calib)
        p_init = [self.p0, self.p1]

        # drop all infinities
        inf_unc_indices = np.isinf(unc)
        inf_err_indices = np.isinf(err)
        if np.sum(inf_unc_indices) > 0 or np.sum(inf_err_indices) > 0:
            print(f"dropping {np.sum(inf_unc_indices)} unc infinities and {np.sum(inf_err_indices)} err infinities")
            both_inf_indices = inf_unc_indices | inf_err_indices
            unc = unc[~both_inf_indices]
            err = err[~both_inf_indices]

        result = minimize(
            flexible_objective,
            p_init,
            args=(unc, err),
            method="Nelder-Mead",
            options={
                "maxiter": 200,
            },
        )

        self.p0 = result.x[0]
        self.p1 = result.x[1]

    def predict(self, heurestic_uncertainty_test):
        calibrated_unc = self.p0 + ((self.p1 ** 2) * heurestic_uncertainty_test)
        calibrated_unc = np.clip(calibrated_unc, 0.0001, None)
        return calibrated_unc, (self.p0, self.p1)

def flexible_objective(
        ps,
        unc,
        err,
    ):
        calibrated_unc = np.zeros_like(unc)
        ps = np.array(ps, dtype="float64")
        unc = np.array(unc, dtype="float64")
        err = np.array(err, dtype="float64")

        calibrated_unc += ps[0] + ((ps[1] ** 2) * unc)
        calibrated_unc = np.clip(calibrated_unc, 0.0001, None)
        costs = log_likelihood(calibrated_unc, err)['average_log_likelihood']
        cost = np.sum(costs)
        print(f"cost: {cost}, params: {ps}")
        return cost
    
class ConformalPredictionLatentNpz:
    def __init__(
        self,
        alpha=0.1,
        num_nearest_neighbors=10,
        debug=False,
        save_dir=None,
        max_clip_multiplier=None,
        nlist_div=100,
        per_atom=None,
        index_constructor=None,
        gpu=False,
        load_index_dir=None,
        load_cp_dir=None,
        ncores=32,
        train_on_n_npz_files=10,
        add_remaining_n_npz_files=10,
        save_histos=None,
        use_std_scaler=True,
        which_frame_latents=-1,
        atoms_to_calib_and_test="both",
        calib_file_type="vasp",
        test_file_type="vasp",
        fit_method="ConformalPrediction"
    ) -> None:
        # set parameters
        self.alpha = alpha
        self.num_nearest_neighbors = num_nearest_neighbors
        self.debug = debug
        self.debug_limit = 1000
        self.max_clip_multiplier = max_clip_multiplier
        self.nlist_div = nlist_div
        self.save_dir = save_dir
        self.per_atom = per_atom
        self.index_constructor = index_constructor
        self.gpu = gpu
        self.load_index_dir = load_index_dir 
        self.load_cp_dir = load_cp_dir
        self.ncores = ncores
        self.train_on_n_npz_files = train_on_n_npz_files
        self.add_remaining_n_npz_files = add_remaining_n_npz_files
        self.data_storage = {}
        self.save_histos = save_histos
        self.use_std_scaler = use_std_scaler
        self.which_frame_latents = which_frame_latents
        self.atoms_to_calib_and_test = atoms_to_calib_and_test
        self.calib_file_type = calib_file_type
        self.test_file_type = test_file_type
        self.fit_method = fit_method

    def fit(self, train_dir, calib_dir):
        if self.load_index_dir is not None and os.path.exists(os.path.join(self.load_index_dir, "std_scaler.pkl")) and os.path.exists(os.path.join(self.load_index_dir, "faiss.index")):
            print(f"load_index_dir detected, loading index and standard scaler from {self.load_index_dir}")
            start = time.time()
            self.load_index(self.load_index_dir)
            self.load_std_scaler(self.load_index_dir)
            print(f"loaded index and standard scaler fit ({time.time()-start} seconds)")
        else:
            self.fit_index(train_dir)
            if self.save_dir is not None:
                self.save_index(self.save_dir)
                self.save_std_scaler(self.save_dir)

        if self.load_cp_dir is not None and os.path.exists(os.path.join(self.load_cp_dir, "cp_model.pkl")) and os.path.exists(os.path.join(self.load_cp_dir, "local_vars.json")):
            print(f"load_cp_dir detected, loading cp_model and local variables from {self.load_cp_dir}")
            start = time.time()
            self.load_cp_model(self.load_cp_dir)
            self.load_local_vars(self.load_cp_dir)
            print(f"loaded cp_model and local variables fit ({time.time()-start} seconds)")
            return self.predict(calib_dir)
        else:
            res = self.calibrate_cp(calib_dir)
            if self.save_dir:
                self.save_cp_model(self.save_dir)
                self.save_local_vars(self.save_dir)
            return res

    def fit_index(self, train_dir):
        # compute the latent representations and residuals for train and test data
        start = time.time()
        npz_file_paths = get_npz_file_paths(train_dir)
        train_npz_file_paths = npz_file_paths[:self.train_on_n_npz_files]
        add_npz_file_paths = npz_file_paths[self.train_on_n_npz_files:]
        train_X = prepare_latentNerror_from_npz(train_npz_file_paths, train_X_only=True, n_jobs=self.ncores, per_atom_x_only=self.per_atom)
        print(f"fit: loaded train data ({time.time()-start} seconds) ({len(train_X)} elements)", flush=True)
        start = time.time()
        if self.debug:
            train_X = train_X[:self.debug_limit]
        # fitting the faiss distance calculator
        self.std_scaler = StandardScaler().fit(train_X)
        if self.use_std_scaler:
            train_X_std_scaled = self.std_scaler.transform(train_X)
        else:
            train_X_std_scaled = train_X
        print(f"fit: initialized standard scaler ({time.time() - start} seconds)")
        start = time.time()
        dimension = train_X_std_scaled.shape[1]
        if self.index_constructor is not None:
            print(f"faiss index factory initializing {self.index_constructor}", flush=True)
            self.index = faiss.index_factory(dimension, self.index_constructor)
        else:
            nlist = int(train_X_std_scaled.shape[0]/self.nlist_div)
            quantizer = faiss.IndexFlatL2(dimension)
            print(f"initializing index with {nlist} clusters", flush=True)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        if self.gpu:
            print(f"using faiss with {faiss.get_num_gpus()} gpus", flush=True)
            index_ivf = faiss.extract_index_ivf(self.index)
            clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dimension))
            index_ivf.clustering_index = clustering_index
        print(f"fit: initialized index ({time.time() - start} seconds), dimension: {dimension}", flush=True)
        start = time.time()
        self.index.train(train_X_std_scaled)
        print(f"fit: trained index ({time.time() - start} seconds)", flush=True)
        start = time.time()
        self.index.add(train_X_std_scaled)
        print(f"fit: added index ({time.time() - start} seconds)", flush=True)
        self.add_to_index(add_npz_file_paths)
        if self.save_histos is not None:
            self.histogram_limit = {"mean": None, "max": None, "sum": None, "int_0": None, "int_-1": None, "n": None, "y": None}
            self.store_instance("train_X_s2ef", X=train_X[:min(len(train_X), 100000)], y=None, sids=None, fids=None, aids=None)
            self.plot_distance_histograms("train_X_s2ef", which_atoms="y", save_name=self.save_histos)

    def add_to_index(self, add_npz_file_paths):
        start = time.time()
        if self.add_remaining_n_npz_files is not None:
            for i, npz_file_path in tqdm(enumerate(add_npz_file_paths), "adding remaining files to index", total=self.add_remaining_n_npz_files):
                if i >= self.add_remaining_n_npz_files:
                    break
                add_X = prepare_latentNerror_from_npz([npz_file_path], train_X_only=True, n_jobs=1, per_atom_x_only=self.per_atom)
                if self.use_std_scaler:
                    add_X_std_scaled = self.std_scaler.transform(add_X)
                else:
                    add_X_std_scaled = add_X
                self.index.add(add_X_std_scaled)
            print(f"fit: added remaining files to index ({time.time() - start} seconds)", flush=True)

    def calibrate_cp(self, calib_dir):
        #calibrate
        start = time.time()
        if self.calib_file_type == "vasp":
            calib_X, calib_y, calib_sids, calib_fids, calib_aids = prepare_latents_from_trajs_and_vasp(calib_dir, per_atom=self.per_atom, n_jobs=self.ncores, which_frame_latents=self.which_frame_latents, atoms_to_calib_and_test=self.atoms_to_calib_and_test)
        elif self.calib_file_type == "npz":
            calib_X, calib_y, calib_sids, calib_fids, calib_aids = prepare_latentNerror_from_npz(get_npz_file_paths(calib_dir), train_X_only=False, debug=False, n_jobs=self.ncores, per_atom_x_only=self.per_atom)
        else:
            raise ValueError(f"calib_file_type must be vasp/npz, got {self.calib_file_type}")
        if self.debug:
            calib_X = calib_X[:self.debug_limit]
            calib_y = calib_y[:self.debug_limit]
        print(f"fit: loaded calibration data ({time.time()-start} seconds) ({len(calib_y)} elements)", flush=True)
        start = time.time()
        # calculating the distance metric for test+calib
        calib_dist = self.calc_dist(calib_X)
        print(f"fit: calculated calibration distance ({time.time()-start} seconds)", flush=True)
        start = time.time()
        # conformal prediction for expected confidence level as (1-alpha)100%
        if self.fit_method.lower() == "conformalprediction" or self.fit_method.lower() == "cp":
            self.model_cp = ConformalPrediction(alpha=self.alpha)
        elif self.fit_method.lower() == "flexiblenll" or self.fit_method.lower() == "nll":
            self.model_cp = FlexibleNLL()
        else:
            raise ValueError(f"fit_method must be conformalprediction/cp or flexiblenll/nll, got {self.fit_method}")
        print(f"fit: initialized conformal prediction model ({time.time()-start} seconds)", flush=True)
        start = time.time()
        if self.per_atom is not None:
            calib_y, calib_dist = self.get_per_atom_dist(calib_y, calib_dist, calib_sids, calib_fids, calib_aids)
        self.model_cp.fit(calib_y, calib_dist)
        self.calib_max = float(np.max(calib_y))
        print(f"fit: fit conformal prediction model ({time.time()-start} seconds), w/ calibration max err: {self.calib_max}", flush=True)
        if self.save_histos:
            self.store_instance("calib_is2re", X=calib_X, y=calib_y, sids=calib_sids, fids=calib_fids, aids=calib_aids)
            self.histogram_limit = {"mean": None, "max": None, "sum": None, "int_0": None, "int_-1": None, "n": None, "y": None}
            if self.per_atom is not None:
                self.plot_distance_histograms("calib_is2re", which_atoms="int_0", save_name=self.save_histos)
                self.plot_distance_histograms("calib_is2re", which_atoms="int_-1", save_name=self.save_histos)
                self.plot_distance_histograms("calib_is2re", which_atoms="mean", save_name=self.save_histos)
                self.plot_distance_histograms("calib_is2re", which_atoms="max", save_name=self.save_histos)
                self.plot_distance_histograms("calib_is2re", which_atoms="sum", save_name=self.save_histos)
            else:
                self.plot_distance_histograms("calib_is2re", which_atoms="n", save_name=self.save_histos)
        return self._predict(calib_dist, calib_y)

    def predict(self, test_dir):
        start = time.time()
        # calculating the distance metric for test+calib
        if self.test_file_type == "vasp":
            test_X, test_y, test_sids, test_fids, test_aids = prepare_latents_from_trajs_and_vasp(test_dir, per_atom=self.per_atom, n_jobs=self.ncores, which_frame_latents=self.which_frame_latents, atoms_to_calib_and_test=self.atoms_to_calib_and_test)
        elif self.test_file_type == "npz":
            test_X, test_y, test_sids, test_fids, test_aids = prepare_latentNerror_from_npz(get_npz_file_paths(test_dir), train_X_only=False, debug=False, n_jobs=self.ncores, per_atom_x_only=self.per_atom)
        else:
            raise ValueError(f"test_file_type must be vasp/npz, got {self.test_file_type}")
        print(f"predict: loaded test data ({time.time()-start} seconds) (test: {len(test_y)} elements)", flush=True)
        start = time.time()
        if self.debug:
            test_X = test_X[:self.debug_limit]
            test_y = test_y[:self.debug_limit]
        # calculating the distance metric for test+calib
        test_dist = self.calc_dist(test_X)
        print(f"predict: calculated test distance ({time.time()-start} seconds), (test: {len(test_dist)} elements, {np.sum(np.isinf(test_dist))} are inf)", flush=True)
        if self.per_atom is not None:
            test_y, test_dist = self.get_per_atom_dist(test_y, test_dist, test_sids, test_fids, test_aids)
        if self.save_histos:
            self.store_instance("test_is2re", X=test_X, y=test_y, sids=test_sids, fids=test_fids, aids=test_aids)
            if self.per_atom is not None:
                self.plot_distance_histograms("test_is2re", which_atoms="int_0", save_name=self.save_histos)
                self.plot_distance_histograms("test_is2re", which_atoms="int_-1", save_name=self.save_histos)
                self.plot_distance_histograms("test_is2re", which_atoms="mean", save_name=self.save_histos)
                self.plot_distance_histograms("test_is2re", which_atoms="max", save_name=self.save_histos)
                self.plot_distance_histograms("test_is2re", which_atoms="sum", save_name=self.save_histos)
            else:
                self.plot_distance_histograms("test_is2re", which_atoms="n", save_name=self.save_histos)
        return self._predict(test_dist, test_y)
    
    def _predict(self, test_dist, test_y):
        start = time.time()
        # conformal prediction for expected confidence level as (1-alpha)100%
        test_uncertainty, qhat = self.model_cp.predict(test_dist)
        print(f"predict: predicted uncertainty ({time.time()-start} seconds)", flush=True)
        if self.max_clip_multiplier is not None:
            test_uncertainty = np.clip(test_uncertainty, 0, self.calib_max*self.max_clip_multiplier)
            print(f"predict: clipped uncertainty (multiplier: {self.max_clip_multiplier}) (calib max: {self.calib_max}) (max unc: {self.calib_max*self.max_clip_multiplier})")
        else:
            print(f"predict: did not clip uncertainty (multiplier: {self.max_clip_multiplier})")
        res = {
            "err": test_y,
            "unc": test_uncertainty,
            "alpha": self.alpha,
        }
        print(f"number of nans: {np.count_nonzero(np.isnan(test_uncertainty))}/{len(test_uncertainty)}")
        return res

    def calc_dist(self, calib_X):
        """
        Returns the distances of calibration data to training data
        """
        if self.use_std_scaler:
            calib_X_std_scaled = self.std_scaler.transform(calib_X)
        else:
            calib_X_std_scaled = calib_X
        distances, neighbors = self.index.search(calib_X_std_scaled, self.num_nearest_neighbors)
        return distances.mean(axis=1)
    
    def get_per_atom_dist(self, calib_y, calib_dist, calib_sids, calib_fids, calib_aids):
        if self.per_atom.startswith("max_"):
            max_n = int(self.per_atom.replace("max_", ""))

        dist_dict = {}
        for sid, fid, aid, dist, y in tqdm(zip(calib_sids, calib_fids, calib_aids, calib_dist, calib_y), "preparing per-atom data", total=len(calib_y)):
            if f"{sid}_{fid}" not in dist_dict:
                dist_dict[f"{sid}_{fid}"] = ([], y)
            if dist != np.inf:
                dist_dict[f"{sid}_{fid}"][0].append(dist)
        for k, v in tqdm(dist_dict.items(), f"collecting {self.per_atom}s", total=len(dist_dict)):
            if self.per_atom == "max":
                dist_dict[k] = (np.max(v[0]), v[1])
            elif self.per_atom == "mean":
                dist_dict[k] = (np.mean(v[0]), v[1])
            elif self.per_atom == "sum":
                dist_dict[k] = (np.sum(v[0]), v[1])
            elif self.per_atom.startswith("max_"):
                sorted_args = np.argsort(v[0])
                sorted_v0 = np.array(v[0])[sorted_args]
                top_n = sorted_v0[-min(max_n, len(v[0])):]
                dist_dict[k] = (np.mean(top_n), v[1])
            else:
                raise ValueError(f"per_atom must be max/mean/n, got {self.per_atom}")
        
        _dist = []
        _y = []
        sid_fid = []
        for k, v in tqdm(dist_dict.items(), "retrieving per-atom data", total=len(dist_dict)):
            _dist.append(v[0])
            _y.append(v[1])
            sid_fid.append(k)
        return np.array(_y), np.array(_dist)
    
    def save_cp_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "cp_model.pkl"), "wb") as f:
            pickle.dump(self.model_cp, f)
        print(f"saved cp model to {save_dir}", flush=True)

    def save_std_scaler(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "std_scaler.pkl"), "wb") as f:
            pickle.dump(self.std_scaler, f)
        print(f"saved std scaler to {save_dir}", flush=True)

    def save_index(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        print(f"saved index to {save_dir}", flush=True)

    def save_local_vars(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "local_vars.json"), "w") as f:
            json.dump({
                "alpha": self.alpha,
                "num_nearest_neighbors": self.num_nearest_neighbors,
                "calib_max": self.calib_max,
            }, f)
        print(f"saved local vars to {save_dir}", flush=True)

    def save_fit(self, save_dir):
        self.save_local_vars(save_dir)
        self.save_std_scaler(save_dir)
        self.save_cp_model(save_dir)
        self.save_index(save_dir)

    def load_cp_model(self, load_dir):
        with open(os.path.join(load_dir, "cp_model.pkl"), "rb") as f:
            self.model_cp = pickle.load(f)
        print(f"loaded cp model from {load_dir}", flush=True)

    def load_std_scaler(self, load_dir):
        with open(os.path.join(load_dir, "std_scaler.pkl"), "rb") as f:
            self.std_scaler = pickle.load(f)
        print(f"loaded std scaler from {load_dir}", flush=True)

    def load_index(self, load_dir):
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        print(f"loaded index from {load_dir}", flush=True)

    def load_local_vars(self, load_dir):
        with open(os.path.join(load_dir, "local_vars.json"), "r") as f:
            local_vars = json.load(f)
        for k, v in local_vars.items():
            setattr(self, k, v)
        print(f"loaded local vars from {load_dir}", flush=True)

    def load_fit(self, load_dir):
        self.load_index(load_dir)
        self.load_cp_model(load_dir)
        self.load_std_scaler(load_dir)
        self.load_local_vars(load_dir)

    def store_instance(self, key, X, y, sids, fids, aids):
        self.data_storage[key] = {
            "X": X,
            "y": y,
            "sids": sids,
            "fids": fids,
            "aids": aids,
        }

    def plot_distance_histograms(self, key, bins=100, which_atoms="y", save_name=""):
        import matplotlib.pyplot as plt
        dist = self.calc_dist(self.data_storage[key]["X"])
        if not which_atoms == "y":
            chunks = [i for i in range(len(self.data_storage[key]["aids"])) if self.data_storage[key]["aids"][i] == 0][1:]
            if which_atoms.startswith("int_"):
                index = int(which_atoms.replace("int_", ""))
                dist = [dist_list[index] for dist_list in np.split(dist, chunks)]
            elif which_atoms == "mean":
                dist = [np.mean(dist_list) for dist_list in np.split(dist, chunks)]
            elif which_atoms == "max":
                dist = [np.max(dist_list) for dist_list in np.split(dist, chunks)]
            elif which_atoms == "sum":
                dist = [np.sum(dist_list) for dist_list in np.split(dist, chunks)]
        else:
            dist = dist.tolist()

        if self.histogram_limit[which_atoms] is not None:
            upper_limit = self.histogram_limit[which_atoms]
        else:
            dist = np.array(dist)
            # upper_limit = np.max(dist[dist < 1e6])
            upper_limit = np.percentile(dist[dist < 1e6], 99)*1.3
            self.histogram_limit[which_atoms] = upper_limit

        dist = np.array(dist).tolist()
        plt.figure()
        plt.hist(np.clip(dist + [upper_limit, 0], None, upper_limit), bins=bins)
        plt.title(f"{which_atoms} {key} distances")
        plt.savefig(os.path.join(self.save_dir, f"{save_name}_hist_{key}_{which_atoms}_dist.png"))

        if self.data_storage[key]["y"] is not None:
            err = self.data_storage[key]["y"]
            fig, axs = plt.subplots(1, 1, figsize=(7, 6))
            hb = axs.hexbin(
                np.clip(dist, None, upper_limit),
                np.clip(err, None, 2),
                gridsize=100,
                mincnt=1,
                extent=[0, upper_limit, 0, 2],
                bins="log",
            )
            axs.plot([0, upper_limit],[0, 2], c="k", label="parity")
            axs.set_xlabel(f"Distance")
            axs.set_ylabel(f"Measured Error (eV)")
            axs.set_title(f"{which_atoms} {key} distances vs. error")
            cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
            fig.colorbar(hb, cax=cbar_ax)
            fig.savefig(os.path.join(self.save_dir, f"{save_name}_scatter_{key}_{which_atoms}_dist_err.png"), dpi=600)
        


def get_npz_file_paths(npz_dir_list):
    npz_file_paths = []
    for npz_dir in npz_dir_list:
        for file in os.listdir(npz_dir):
            if file.endswith(".npz"):
                npz_file = os.path.join(npz_dir, file)
                npz_file_paths.append(npz_file)
    return npz_file_paths

def prepare_latentNerror_from_npz(npz_file_paths, train_X_only=False, debug=False, n_jobs=8, per_atom_x_only=True):
    """
    Return the latent representations and respective residuals for data list.
    """
    # get the latent representations
    if train_X_only:
        tqdm_str = "loading train_X_only npz files"
    else:
        tqdm_str = "loading npz files"

    sids = []
    fids = []
    aids = []
    latents = []
    errors = []
    if debug:
        results = [_prepare_latent_from_npz_parallel_helper((file, train_X_only, per_atom_x_only)) for file in npz_file_paths]
    else:
        results = pqdm([(file, train_X_only, per_atom_x_only) for file in npz_file_paths], _prepare_latent_from_npz_parallel_helper, n_jobs=min(n_jobs, len(npz_file_paths)), desc=tqdm_str)
    for _latents, _errors, _sids, _fids, _aids in results:
        latents.extend(_latents)
        if not train_X_only:
            errors.extend(_errors)
            sids.extend(_sids)
            fids.extend(_fids)
            aids.extend(_aids)

    latents = np.array(latents)

    if train_X_only:
        return latents
    else:
        errors = np.array(errors)
        return latents, errors, sids, fids, aids
    
def _prepare_latent_from_npz_parallel_helper(file_and_bool):
    file, train_X_only, per_atom_x_only = file_and_bool
    with np.load(file, allow_pickle=True) as data:
        if "latents" in data.keys():
            key = "latents"
        else:
            key = "predictions"
        if train_X_only:
            _latents = data[key].astype(np.float32)
            _errors = None
            _sids = None
            _fids = None
            _aids = None
            if per_atom_x_only is None:
                _latents = [np.mean(lat_arr, axis=0) for lat_arr in np.split(_latents, data["chunk_idx"])]
        else:
            _latents = data[key].astype(np.float32)
            _errors = data["errors"].astype(np.float32)
            _sids = data["sids"]
            _fids = data["fids"]
            _aids = data["aids"]
            if per_atom_x_only is None:
                chunk_idx = []
                for i, sid in enumerate(_sids):
                    if i != 0 and sid != _sids[i-1]:
                        chunk_idx.append(i)
                unique_sids = list(dict.fromkeys(_sids))
                assert len(unique_sids) == len(chunk_idx) + 1, f"{len(unique_sids)} == {len(chunk_idx) + 1}"
                zeros = [err_arr[0] - np.mean(err_arr) for err_arr in np.split(_errors, chunk_idx)]
                assert np.sum(zeros) < 0.1, f"{np.sum(zeros)} < 0.1"
                _latents = [np.mean(lat_arr, axis=0) for lat_arr in np.split(_latents, chunk_idx)]
                _errors = [np.mean(err_arr) for err_arr in np.split(_errors, chunk_idx)]
                _sids = [sid[0] for sid in np.split(_sids, chunk_idx)]
                _fids = [fid[0] for fid in np.split(_fids, chunk_idx)]
                _aids = [aid[0] for aid in np.split(_aids, chunk_idx)]
    return _latents, _errors, _sids, _fids, _aids
    

def prepare_latents_from_trajs_and_vasp(
    ml_traj_and_vasp_sp_dir_list,
    per_atom=False,
    n_jobs=32,
    which_frame_latents=-1,
    atoms_to_calib_and_test="both",
):
    per_sys_latents = []
    per_sys_errors = []
    per_sys_sids = []
    for ml_traj_and_vasp_sp_dir in ml_traj_and_vasp_sp_dir_list:
        latent_trajs_dir = os.path.join(ml_traj_and_vasp_sp_dir, "trajs")
        vasp_sp_dir = os.path.join(ml_traj_and_vasp_sp_dir, "vasp_results/vasp_dirs")

        assert os.path.exists(latent_trajs_dir), f"{latent_trajs_dir} does not exist"
        assert os.path.exists(vasp_sp_dir), f"{vasp_sp_dir} does not exist"

        print(f"loading latent trajs from {latent_trajs_dir} and vasp sp from {vasp_sp_dir}")
        _latents, _errors, _sids = load_is2re_data(
            latent_trajs_dir=latent_trajs_dir,
            vasp_sp_dir=vasp_sp_dir,
            n_jobs=n_jobs,
            debug=False,
            which_frame_latents=which_frame_latents,
            return_sids=True,
            atoms_to_calib_and_test=atoms_to_calib_and_test
        )
        per_sys_latents.extend(_latents)
        per_sys_errors.extend(_errors)
        per_sys_sids.extend(_sids)

        latents = []
        errors = []
        sids = []
        fids = []
        aids = []
        for latent_sys, err, sid in tqdm(zip(per_sys_latents, per_sys_errors, per_sys_sids), "gathering per-atom data", total=len(per_sys_latents)):
            if not per_atom:
                latents.append(np.mean(latent_sys, axis=0))
                errors.append(err)
                sids.append(sid)
                fids.append(-1)
                aids.append(-1)
            else:
                for aid, a_latent in enumerate(latent_sys):
                    latents.append(a_latent)
                    errors.append(err)
                    sids.append(sid)
                    fids.append(-1)
                    aids.append(aid)

        latents = np.array(latents, dtype=np.float32)
        errors = np.array(errors, dtype=np.float32)
        aids = np.array(aids)
        fids = np.array(fids)
        sids = np.array(sids)
        return latents, errors, sids, fids, aids


def get_parser():
    parser = argparse.ArgumentParser(description="collect latent npz files")
    parser.add_argument(
        "--train_dir",
        type=str,
        nargs="+",
        default=["/private/home/jmu/storage/inference/latent_s2ef_equiformer/source_npz_files/few_files/train/train_train"],
        help="list of training data directories to fit the index on, should contain .npz files with latent representations from training dataset"
    )
    parser.add_argument(
        "--calib_dir",
        type=str,
        nargs="+",
        default=["/private/home/jmu/storage/trajectory_results/oc20data_oc20models_1equiformer/is2re_val_id"],
        help="list of calibration data directories to run calibration on, should contain a ./trajs (containing ml trajs) and ./vasp_results/vasp_dirs (containing dft results on final frames) directory"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        nargs="+",
        default=["/private/home/jmu/storage/trajectory_results/oc20data_oc20models_1equiformer/is2re_val_ood"],
        help="list of test data directories to run predictions on, should contain a ./trajs (containing ml trajs) and ./vasp_results/vasp_dirs (containing dft results on final frames) directory"
    )
    parser.add_argument(
        "--per_atom",
        type=str,
        default="sum",
        help="whether to collect per-atom data (max/mean/n) \n\
            max=calculate all the per-atom distance, and take the max distance\n\
            mean=calculate all the per-atom distances, and take the mean distance\n\
            sum=calculate all the per-atom distances, and take the sum of the distances\n\
            n=not per-atom, instead take the mean of the latents and compute the distance from that)"
    )
    parser.add_argument(
        "--debug",
        type=str,
        default="n",
        help="whether to run in debug mode (y/n), this limits the data to 1000 elements to speed up execution"
    )
    parser.add_argument(
        "--num_nearest_neighbors",
        type=int,
        default=1,
        help="number of nearest neighbors to sample when calculating distance, distance is computed as the mean of these neighbor distance"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.33,
        help="confidence level for conformal prediction"
    )
    parser.add_argument(
        "--save_npz",
        type=str,
        default="results_%.npz",
        help="filename for the results npz file (%% will be replaced with the confidence level), will be saved in the save_dir path"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="relative directory path to which the index, model files, and results npz file will be saved (%% will be replaced with the confidence level)"
    )
    parser.add_argument(
        "--load_index_dir",
        type=str,
        default=None,
        help="relative directory path from which to load the index and standard scaler"
    )
    parser.add_argument(
        "--load_cp_dir",
        type=str,
        default=None,
        help="relative directory path from which to load the cp model and local variables"
    )
    parser.add_argument(
        "--max_clip_multiplier",
        type=float,
        default=2,
        help="multiplier for clipping the uncertainty, uncertainties predictions will be clipped above max_clip_multiplier*(max error in the calibration set), if None, no clipping will be performed"
    )
    parser.add_argument(
        "--nlist_div",
        type=int,
        default=1000,
        help="determines the number of clusters in the faiss index. The number of clusters will be calculated as len(train_X)/nlist_div (overriden by index_constructor if not None)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="n",
        help="whether to use gpu for faiss (y/n)"
    )
    parser.add_argument(
        "--index_constructor",
        type=str,
        default=None,
        help="faiss index constructor, if None, will use IVFFlat with nlist=nlist_div, otherwise will use the specified index constructor (see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)"
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=8,
        help="number of cores to use for parallel processing"
    )
    parser.add_argument(
        "--train_on_n_npz_files",
        type=int,
        default=20,
        help="number of npz files to train on, the rest will be added to the index after training is complete"
    )
    parser.add_argument(
        "--add_remaining_n_npz_files",
        type=int,
        default=0,
        help="number of npz files to add to the index after training is complete"
    )
    parser.add_argument(
        "--save_histos",
        type=str,
        default="n",
        help="save histograms of distances and errors to compare distribution of distances (n/int_#/mean/max)"
    )
    parser.add_argument(
        "--use_std_scaler",
        type=str,
        default="y",
        help="whether or not to use std_scaler to normalize latents (y/n)"
    )
    parser.add_argument(
        "--which_frame_latents",
        type=int,
        default=-1,
        help="which frame to use for latents (int)"
    )
    parser.add_argument(
        "--atoms_to_calib_and_test",
        type=str,
        default="both",
        help="which atoms to load for calibration and testing, options: [\n\
            both, (default)\n\
            adsorbate,\n\
            surface,\n\
            ]"
    )
    parser.add_argument(
        "--calib_file_type",
        type=str,
        default="npz",
        help="what kind of files to load for calibration, options: [vasp, npz]"
    )
    parser.add_argument(
        "--test_file_type",
        type=str,
        default="npz",
        help="what kind of files to load for prediction, options: [vasp, npz]"
    )
    parser.add_argument(
        "--alternate_train_dir",
        type=str,
        nargs="+",
        default=None,
        help="list of alternate train directories to examine distribution if plot_histos"
    )
    parser.add_argument(
        "--fit_method",
        type=str,
        default="flexibleNLL",
        help="which method to use for fitting (ConformalPrediction/flexibleNLL)"
    )
    
    return parser

def main(config):
    pp.pprint(config)
    debug = False
    if config["debug"] == "y":
        debug = True
    print("starting conformal prediction processing", flush=True)
    confidence = int((1-config["alpha"]) * 100)

    save_id_npz = config["save_npz"].replace("%", f"{confidence}%").replace(".npz", "_id.npz")
    save_ood_npz = config["save_npz"].replace("%", f"{confidence}%").replace(".npz", "_ood.npz")
    save_dir = None
    if config["save_dir"] is not None:
        save_dir = config["save_dir"].replace("%", f"{confidence}%")
        if debug:
            save_dir = f"{save_dir}_debug"
        save_id_npz = os.path.join(save_dir, save_id_npz)
        save_ood_npz = os.path.join(save_dir, save_ood_npz)
        if debug:
            save_id_npz = save_id_npz.replace(".npz", "_debug.npz")
            save_ood_npz = save_ood_npz.replace(".npz", "_debug.npz")
        os.makedirs(save_dir, exist_ok=True)

    if config["per_atom"] == "max":
        per_atom = "max"
    elif config["per_atom"] == "mean":
        per_atom = "mean"
    elif config["per_atom"] == "sum":
        per_atom = "sum"
    elif config["per_atom"].startswith("max_"):
        per_atom = config["per_atom"]
    elif config["per_atom"] == "n":
        per_atom = None
    else:
        raise ValueError(f"per_atom must be max/mean/n, got {config['per_atom']}")

    if config["save_histos"] == "n":
        save_histos = None
    else:
        save_histos = config["save_histos"]

    cpl = ConformalPredictionLatentNpz(
        alpha=config["alpha"],
        num_nearest_neighbors=config["num_nearest_neighbors"],
        debug=debug,
        save_dir=save_dir,
        max_clip_multiplier=config["max_clip_multiplier"],
        nlist_div=config["nlist_div"],
        per_atom=per_atom,
        index_constructor=config["index_constructor"],
        gpu=config["gpu"] == "y",
        load_index_dir=config["load_index_dir"],
        load_cp_dir=config["load_cp_dir"],
        ncores=config["ncores"],
        train_on_n_npz_files=config["train_on_n_npz_files"],
        add_remaining_n_npz_files=config["add_remaining_n_npz_files"],
        save_histos=save_histos,
        use_std_scaler=config["use_std_scaler"] == "y",
        which_frame_latents=config["which_frame_latents"],
        atoms_to_calib_and_test=config["atoms_to_calib_and_test"],
        calib_file_type=config["calib_file_type"],
        test_file_type=config["test_file_type"],
        fit_method=config["fit_method"]
    )
    id_res = cpl.fit(config["train_dir"], config["calib_dir"])
    np.savez_compressed(save_id_npz, **id_res)

    ood_res = cpl.predict(config["test_dir"])
    np.savez_compressed(save_ood_npz, **ood_res)

    if config["alternate_train_dir"] is not None and save_histos is not None:
        train_X = prepare_latentNerror_from_npz(get_npz_file_paths(config["alternate_train_dir"]), train_X_only=True, n_jobs=cpl.ncores, per_atom_x_only=cpl.per_atom)
        cpl.store_instance("train_X_s2ef_alternate", X=train_X[:min(len(train_X), 100000)], y=None, sids=None, fids=None, aids=None)
        cpl.plot_distance_histograms("train_X_s2ef_alternate", which_atoms="y", save_name=cpl.save_histos)
        print("done saving histograms")

    print(f"max error: {np.max(ood_res['err'])}", flush=True)
    print(f"mean error: {np.mean(ood_res['err'])}", flush=True)
    print(f"min error: {np.min(ood_res['err'])}", flush=True)
    print(f"max uncertainty: {np.max(ood_res['unc'])}", flush=True)
    print(f"mean uncertainty: {np.mean(ood_res['unc'])}", flush=True)
    print(f"min uncertainty: {np.min(ood_res['unc'])}", flush=True)
    print("done")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)

# TODO
# implement a partial fit method so that it can be fit across many npz files
# implement a way to yield one npz file contents at a time so that the whole dataset doesn't take up all the memory