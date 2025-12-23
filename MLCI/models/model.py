import pandas as pd
import os, json
import numpy as np
from astropy.table import Table, vstack
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pprint import pprint
from scipy.special import erf
from astropy.cosmology import Planck18 as cosmo
from ..data_processing.preprocess import observed_data
from scipy.stats import gaussian_kde


# cosmolog label used in model
#simulation_labels = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C11', 'C13', 'C14']
#simulation_labels = ['C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
sim_catalog = pd.read_csv('./simulation_paras.csv')
output_para_name = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
simulation_labels = [f'C{i+3}' for i in range(13)]
simulation_paras, prob_lable = [], []


# the flux limt is set 5*1e-14 erg / s / cm^2 https://arxiv.org/pdf/2401.17274
D0 = cosmo.luminosity_distance(z=0.001).to('cm').value
flux_limit = 5*1e-14
sigma_logL = np.log10(flux_limit*(4*np.pi*D0**2)) - np.log10(0.2*flux_limit*(4*np.pi*D0**2))  #对数光度误差

# 每个 z 下的极限光度（用于与目标 L 比较）
def L_lim_z(z):
    D_L = cosmo.luminosity_distance(z).to('cm').value  # 输出单位：cm
    L_lim = 4 * np.pi * D_L**2 * flux_limit
    return L_lim

z_bins = np.linspace(0, 1.4, 15)
def sigma_L_fitting(observed_data, deg=2):
    z_points, median_points, sigma_points, peak_sigmas = [], [], [], []
    for i in range(len(z_bins)-1):
        subset = observed_data[(observed_data['z'] >= z_bins[i]) & (observed_data['z'] < z_bins[i+1])]
        valid_mask = ~np.isnan(subset['L'])
        subset = subset[valid_mask]
        q16, q50, q84 = np.percentile(subset['L'], [16, 50, 84])
        L_kde = gaussian_kde(subset['L'])
        L_x = np.linspace(np.min(subset['L']), np.max(subset['L']), 1000)
        L_density = L_kde(L_x)
        L_peak = L_x[np.argmax(L_density)]
        logL_lim = np.log10(L_lim_z((z_bins[i]+z_bins[i+1])/2))
        low_lum_subset = subset[subset['L']<=logL_lim] # under the flux limit
        if len(low_lum_subset) <= 0:
            continue
        low_limt = np.percentile(low_lum_subset['L'], [16])
        sigma_eff = logL_lim - low_limt
        print(f"sigma of z={z_bins[i]:.1f}-{z_bins[i+1]:.1f}:", sigma_eff)
        z_points.append((z_bins[i]+z_bins[i+1])/2)
        median_points.append(q50)
        sigma_points.append(sigma_eff)
        peak_sigmas.append((L_peak - logL_lim)/2)
    median_coeffs= np.polyfit(z_points, median_points, deg)
    sigma_coeffs = np.polyfit(z_points, sigma_points, deg)
    peak_sigma_coeffs = np.polyfit(z_points, peak_sigmas, deg)
    return median_coeffs, sigma_coeffs, peak_sigma_coeffs
median_coeffs, sigma_coeffs, peak_sigma_coeffs =  sigma_L_fitting(observed_data())


def selection_function(logL, z, threshold=1e-3):
    z = np.clip(1e-3, 5, z)
    logL_lim = np.log10(L_lim_z(z))
    #logL_lim = np.polyval(median_coeffs, z)
    sigma = np.polyval(sigma_coeffs, z)
    #sigma = np.polyval(peak_sigma_coeffs, z)
    #sigma = np.polyval(median_coeffs, z) - logL_lim # peak - limted flux
    detect_prob = 0.5*(1 - erf((logL_lim - logL)/(np.sqrt(2) * sigma)))
    #detect_prob = 0.5*(1 - erf((logL_lim - logL)/(np.sqrt(2) * sigma_logL)))
    detect_prob = np.clip(detect_prob, threshold, 1.0) 
    return detect_prob


def pre_selection(table_data, set_var=False):
    global simulation_paras, prob_lable
    # slecte a subset, updata simulation parameters and labels
    selected_subset = []
    if not set_var:
        for label in simulation_labels:
            selected_subset.append(table_data[table_data['label'] == label])
        table_data = vstack(selected_subset) 
    prob_lable = [f'prob_{lable}' for lable in simulation_labels]
    simulation_paras = []
    for label in simulation_labels:
        para = sim_catalog[sim_catalog['name'] == label][output_para_name]
        simulation_paras.append(para)
    simulation_paras = np.vstack(simulation_paras)
    print('train cosmology:', simulation_labels)
    print('simulation_paras', simulation_paras)
    print('prob_lable', prob_lable)
    return table_data


def transform(table_data, column_rangs, ramdom_seed=42): 
    #=====
    # calculated the selection function
    #===selection effect
    L500 = table_data['L'].data
    z = table_data['z'].data
    detect_prob = selection_function(L500, z)
    table_data['detect_prob'] = np.clip(1e-5, 1, detect_prob)
    pandas_table = table_data.to_pandas()
    for col_name, col_range in column_rangs.items():
        pandas_table[col_name+'_norm'] = (pandas_table[col_name]-col_range[0])/(col_range[1]-col_range[0])
    # shuffling the table
    np.random.seed(ramdom_seed)
    shuffled_pandas_table = pandas_table.sample(frac=1).reset_index(drop=True)
    return shuffled_pandas_table


def data_split(table, frac=0.9, ramdom_seed=42):
    # input an Astropy Table
    np.random.seed(ramdom_seed)
    # random index
    if frac > 1.0:
        num_test = int(frac)
        table = table[np.random.permutation(len(table))]
        indices = np.arange(len(table))
        simulation_list = np.unique(table['label'])
        train_idx, valid_idx = [], []
        for label in simulation_list:
            label_indices = indices[table['label']==label]
            split_idx = num_test
            valid_idx.extend(label_indices[:split_idx])
            train_idx.extend(label_indices[split_idx:])
    else:
        indices = np.arange(len(table))
        np.random.shuffle(indices)
        split_idx = int(len(table) * frac)
        train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    # split table
    print(f"Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
    train_table = table[train_idx]    
    valid_table = table[valid_idx]   
    return train_table, valid_table


# save the config of model
def save_config_as_json(filename, config_dict):
    with open(filename, 'w') as json_file:
        json.dump(config_dict, json_file, indent=2)

# load the config of model
def load_config_from_json(filename):
    with open(filename, 'r') as json_file:
        config_dict = json.load(json_file)
    pprint(config_dict)
    return config_dict

def save_output_csv(point_value, point_weight, detect_prob, prob, z, L=-1, true_label=-1, output_name=f'./results/predict.csv'):
    point_value = pd.DataFrame(data=point_value, columns=output_para_name)
    point_weight = pd.DataFrame(data=point_weight, columns=['weight'])
    detect_prob = pd.DataFrame(data=detect_prob, columns=['detect_prob'])
    z = pd.DataFrame(data=z, columns=['z'])
    true_label = pd.DataFrame(data=true_label, columns=['label'])
    L = pd.DataFrame(data=L, columns=['L'])
    results = pd.concat([point_value, point_weight, detect_prob, z, prob, true_label, L], axis=1)
    results.to_csv(output_name, index=False)
    return results


class RandomForest:
    """
    Initialize, one shold set the wavelength_min, wavelength_max, output_label, and the name of model.
    """
    def __init__(self):
        #==== data selecting/augmentation setting
        self.config = { 'model_name': 'RandomForest',
                        'input_column_rangs': {},
                        'p_threshold': 0.0,
                        'bin_num': -1,
                        'split_fraction': 0.1,
                        'n_estimators':100,
                        'min_samples_leaf':100*100,
                        'criterion':'gini',
                        'class_weight':'balanced',
                        'n_jobs': 12,
                        'max_depth': 10,
                        'weight':True,
                        'sample_num':100,
                        'max_num':1e4,
                        'simulation_labels':simulation_labels,
                        'noise_level': [1/3, 1/20],
                        'agn_feedback': 0.0,
                        }
        self.train_data, self.test_data = [], []
        self.ramdom_seed = 42

        
    def training(self):
        if not os.path.exists(f"models/{self.config['model_name']}"):
            os.mkdir(f"models/{self.config['model_name']}")
        input_column_rangs = self.config['input_column_rangs']
        self.train_data = transform(self.train_data, input_column_rangs)
        input_column_rangs = self.config['input_column_rangs']
        col_names = list(input_column_rangs.keys())
        col_names = [x + '_norm' for x in col_names]
        x_train = self.train_data[col_names].values
        y_train = self.train_data['label'].values
        print('Num of x_train {}, number of y_train {}'.format(len(x_train), len(y_train)))
        rf_classifier = RandomForestClassifier( n_estimators= self.config['n_estimators'],
                                                class_weight = self.config['class_weight'],
                                                criterion=self.config['criterion'], 
                                                random_state=self.ramdom_seed, 
                                                max_depth=self.config['max_depth'], 
                                                min_samples_leaf=int(self.config['min_samples_leaf']),
                                                max_features= len(input_column_rangs)-1, # 'sqrt' ,
                                                n_jobs=self.config['n_jobs'],
                                                )
        rf_classifier.fit(x_train, y_train)
        # save model and config
        output_file = f"models/{self.config['model_name']}/{self.config['model_name']}.pkl"
        joblib.dump(rf_classifier, output_file, compress=True)
        # save the configration
        self.config['cosmology_label'], self.config['cosmology_para'] = simulation_labels, simulation_paras.tolist()
        filename = f"models/{self.config['model_name']}/{self.config['model_name']}.json"
        save_config_as_json(filename, self.config)
        print('training finished!')


    def probability(self, show_acc=True):
        # input fetures
        input_column_rangs = self.config['input_column_rangs']
        self.test_data = transform(self.test_data, input_column_rangs)
        col_names = list(input_column_rangs.keys())
        col_names = [x + '_norm' for x in col_names]
        x_test = self.test_data[col_names].values
        # load model
        output_file = f"models/{self.config['model_name']}/{self.config['model_name']}.pkl"
        loaded_model = joblib.load(output_file)
        probs = loaded_model.predict_proba(x_test)
        probs = pd.DataFrame(data=probs, columns=loaded_model.classes_)
        # calculate accuracy
        if show_acc:
            y_test = self.test_data['label'].values
            y_pred = loaded_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)
        # reshape as a dictionary
        probs_dict = {}
        for label in simulation_labels:
            probs_dict[label] = probs[:][label]
            self.test_data[f'prob_{label}'] = probs[:][label]
        return probs_dict

    def points(self, output_name='', show_acc=False, confusion_matrix=None):
        json_filename = f"models/{self.config['model_name']}/{self.config['model_name']}.json"
        self.config.update(load_config_from_json(json_filename)) # load configuration
        L500 = self.test_data['L'].value.copy()
        probs_dict = self.probability(show_acc)
        prob = self.test_data[prob_lable]
        prob_std = np.std(prob.values, axis=-1)
        if self.config['weight']:
            point_weight = len(simulation_labels) * prob_std**2
        else:
            point_weight = np.ones_like(prob_std)
        
        #===selection effect
        detect_prob = self.test_data['detect_prob'].values
        z = self.test_data['z'].values
        point_weight = point_weight/detect_prob
        #=== threshold
        point_weight[np.max(prob.values, axis=-1) < self.config['p_threshold']] = 0
        if confusion_matrix is None:
            point_value = np.matmul(prob.values, simulation_paras)
        else:
            new_pro = np.matmul(prob.values, confusion_matrix.T)
            norm_new_pro = new_pro / np.sum(new_pro, axis=1, keepdims=True)
            point_value = np.matmul(norm_new_pro, simulation_paras)
        # =====
        if output_name == '':
            output_name = f"./results/{self.config['model_name']}_predict.csv"
        if show_acc:
            results = save_output_csv(point_value, point_weight, detect_prob, prob, z, L=L500, true_label=self.test_data['label'].values, output_name=output_name)
        else:
            results = save_output_csv(point_value, point_weight, detect_prob, prob, z, L=L500, true_label=['None',]*len(point_value), output_name=output_name)
        return results

    def get_model(self):
        json_filename = f"models/{self.config['model_name']}/{self.config['model_name']}.json"
        self.config.update(load_config_from_json(json_filename)) # load configuration
        output_file =  f"models/{self.config['model_name']}/{self.config['model_name']}.pkl"
        loaded_model = joblib.load(output_file)
        return loaded_model



class Naive_Bayesian:
    """
    Initialize, one shold set the wavelength_min, wavelength_max, output_label, and the name of model.
    """
    def __init__(self):
        self.config = { 'model_name': 'Naive_Bayesian',
                        'input_column_rangs': {},
                        'p_threshold': 0.0,
                        'weight':True,
                        'bin_num': 5,
                        'split_fraction': 0.1
                        }
        self.train_data, self.test_data = [], []
    
    def calculate_bin_edges(self):
        # updata and normalize the training data
        input_column_rangs = self.config['input_column_rangs']
        self.train_data = transform(self.train_data, input_column_rangs)
        bin_edges = {}
        bin_num = self.config['bin_num']
        edge_percentages = np.linspace(0, 100, bin_num + 1)

        col_names = list(input_column_rangs.keys())
        col_names = [x + '_norm' for x in col_names]
        for col_name  in col_names:
            data = self.train_data[col_name].values
            # finding the 0%, 10%, 20% ... 90%, 100% points as edges
            bin_edges[col_name] = [np.percentile(data, percentage) for percentage in edge_percentages]
            bin_edges[col_name][-1] = bin_edges[col_name][-1] + 1e-3
        self.config['edge'] = bin_edges


    def counting_each_simualtion_each_bin(self, label):
        # 使用numpy.histogramdd函数进行多维网格分bin
        if label == 'all':
            #data_set = self.train_data
            data_set = [self.train_data[self.train_data['label']==lab] for lab in simulation_labels]
            data_set = pd.concat(data_set)
        else:
            data_set = self.train_data[self.train_data['label']==label]
            
        bin_edges = self.config['edge']
        bins = [val for key, val in bin_edges.items()]

        input_column_rangs = self.config['input_column_rangs']
        col_names = list(input_column_rangs.keys())
        col_names = [x + '_norm' for x in col_names]
        values = data_set[col_names].values

        hist, edges = np.histogramdd(values, bins=bins)
        edge_labels = []
        for i in range(len(col_names)):
            edges_0, edges_1 = edges[i][0:-1], edges[i][1:]
            edge = np.array([f"{a}:{b}" for a, b in zip(edges_0, edges_1)])
            edge_labels.append(edge)
        # Flatten the grid
        points_edge = np.meshgrid(*edge_labels, indexing='ij')
        cols = {}
        for i in range(len(col_names)):
            cols[col_names[i]+'_edge'] = points_edge[i].flatten()
        cols['num'] = hist.flatten()
        num_counting = pd.DataFrame(cols)
        return num_counting
        
    
    def calculate_likelihood(self):
        # counting the numbers on each simulation
        if not os.path.exists(f"models/{self.config['model_name']}"):
            os.mkdir(f"models/{self.config['model_name']}")
        self.calculate_bin_edges()
        num_count_all = self.counting_each_simualtion_each_bin('all')
        num_count_all.to_csv(f"models/{self.config['model_name']}/likelihood_all.csv")
        num_counts = {}
        for label in simulation_labels:
            num_counts[label] = self.counting_each_simualtion_each_bin(label)
        # calculate the likelihood
        for key, value in num_counts.items():
            llh = num_counts[key]['num']/num_count_all['num']
            llh = np.nan_to_num(llh)
            num_counts[key]['likelihood'] = llh
            num_counts[key].to_csv(f"models/{self.config['model_name']}/likelihood_{key}.csv")
        # save the model configration
        self.config['cosmology_label'], self.config['cosmology_para'] = simulation_labels, simulation_paras.tolist()
        filename = f"models/{self.config['model_name']}/{self.config['model_name']}.json"
        save_config_as_json(filename, self.config)
    training = calculate_likelihood

    def posterior(self):
        json_filename = f"models/{self.config['model_name']}/{self.config['model_name']}.json"
        self.config.update(load_config_from_json(json_filename))
        input_column_rangs = self.config['input_column_rangs']
        self.test_data = transform(self.test_data, input_column_rangs)
        # if there is a flat prior, the posterior is equal to likelihood
        # finding the corresbonding bins index
        bin_edges = self.config['edge']
        col_names = list(input_column_rangs.keys())
        col_names = [x + '_norm' for x in col_names]
        x_indexs = {}
        for col_name in col_names:
            edge = bin_edges[col_name]
            x_point = self.test_data[col_name].values
            array_index = np.digitize(x_point, edge).reshape(-1,1) - 1 # array_index = bin_index - 1 
            print(col_name, 'out_range', sum((array_index>self.config['bin_num']-1) + (array_index<0)))
            array_index = np.clip(array_index, 0, self.config['bin_num']-1)
            x_indexs[col_name] = array_index
        # flatten the indexs
        bin_num = [len(value)-1 for key, value in  bin_edges.items()] # the bin number in each dim
        muti_dim_indexs = [value for key, value in x_indexs.items()]
        muti_dim_indexs = np.concatenate(muti_dim_indexs, axis=-1)
        flatten_idnexs = np.ravel_multi_index(np.transpose(muti_dim_indexs), bin_num)
        # get the likelihood, there are 13 likelihood for 13 cosmology parameters (exclude C1 and C2)
        llh = {}
        for label in simulation_labels:
            llh_function = pd.read_csv(f"models/{self.config['model_name']}/likelihood_{label}.csv")
            llh[label] = llh_function.iloc[flatten_idnexs]['likelihood'].values
            self.test_data[f'prob_{label}'] = llh[label]
        return llh

    def points(self, output_name='', show_acc=False, test=False, confusion_matrix=None):
        L500 = self.test_data['L'].value.copy()
        llh = self.posterior()
        prob = self.test_data[prob_lable]
        prob_std = np.std(prob.values, axis=-1)
        if self.config['weight']:
            point_weight = len(simulation_labels) * prob_std**2
        else:
            point_weight = np.ones_like(prob_std)
        detect_prob = self.test_data['detect_prob'].values
        z = self.test_data['z'].values
        point_weight = point_weight/detect_prob
        point_weight[np.max(prob.values, axis=-1) < self.config['p_threshold']] = 0
        if confusion_matrix is None:
            point_value = np.matmul(prob.values, simulation_paras)
        else:
            new_pro = np.matmul(prob.values, confusion_matrix.T)
            norm_new_pro = new_pro / np.sum(new_pro, axis=1, keepdims=True)
            point_value = np.matmul(norm_new_pro, simulation_paras)
        if output_name == '':
            output_name = f"./results/{self.config['model_name']}_predict.csv"
        if show_acc:
            results = save_output_csv(point_value, point_weight, detect_prob, prob, z, L=L500, true_label=self.test_data['label'].values, output_name=output_name)
        else:
            results = save_output_csv(point_value, point_weight, detect_prob, prob, z, L=L500, true_label=['None',]*len(point_value), output_name=output_name)
        return results
