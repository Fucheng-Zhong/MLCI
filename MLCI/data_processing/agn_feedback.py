from . import preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

proportions = {'R_corr':0.2, 'Mgas':0.5, 'L':1.0, 'T':0.5}

def agn_corrected_data(delta='median', bin_num=20, proportional=False, delta_prop=1):
    simulated_data = preprocess.simulated_data()
    simulated_c8a1 = simulated_data[simulated_data['label'] == 'C8a1']
    simulated_c8a2 = simulated_data[simulated_data['label'] == 'C8a2']
    simulated_c8   = simulated_data[simulated_data['label'] == 'C8']
    cols = {'R_corr':[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1]}
    bins_edge = np.linspace(13, 15, bin_num+1)
    delta_plus = {}
    delta_minus = {}
    delta_plus['M_TOL'] = delta_minus['M_TOL'] = (bins_edge[0:-1] + bins_edge[1:])/2
    for key, value in cols.items():
        delta_plus[key] = np.zeros(len(bins_edge)-1)
        delta_minus[key] = np.zeros(len(bins_edge)-1)
        
    for i in range(len(bins_edge)-1):
        subset_c8a1 =  simulated_c8a1[(simulated_c8a1['M_TOL'] >= bins_edge[i]) & (simulated_c8a1['M_TOL'] < bins_edge[i+1])]
        subset_c8a2 =  simulated_c8a2[(simulated_c8a2['M_TOL'] >= bins_edge[i]) & (simulated_c8a2['M_TOL'] < bins_edge[i+1])]
        subset_c8 =  simulated_c8[(simulated_c8['M_TOL'] >= bins_edge[i]) & (simulated_c8['M_TOL'] < bins_edge[i+1])]
        for key, value in cols.items():
            if isinstance(delta, float):
                if proportional:
                    var_list = proportions.keys()
                    if key in var_list:
                        temp_delta = proportions[key]*delta
                else:
                    temp_delta = delta
                delta_plus[key][i] = temp_delta
                delta_minus[key][i] = -temp_delta
            elif len(subset_c8)==0 or len(subset_c8a1)==0 or len(subset_c8a2)==0:
                delta_plus[key][i] = delta_minus[key][i] = 0
            elif delta == 'median':
                delta_plus[key][i] = delta_prop*(np.median(subset_c8[key]) - np.median(subset_c8a2[key])) 
                delta_minus[key][i] = delta_prop*(np.median(subset_c8[key]) - np.median(subset_c8a1[key]))
            elif delta == 'mean':
                delta_plus[key][i] = delta_prop*(np.mean(subset_c8[key]) - np.mean(subset_c8a2[key])) 
                delta_minus[key][i] = delta_prop*(np.mean(subset_c8[key]) - np.mean(subset_c8a1[key]))

    exclude_set = ['HR', 'C8a1', 'C8a2']
    mask = ~np.isin(simulated_data['label'], exclude_set)
    simulated_data_plus = simulated_data[mask].copy()
    simulated_data_minus = simulated_data[mask].copy()
    for i in range(len(bins_edge)-1):
        mask = (simulated_data_plus['M_TOL']  >= bins_edge[i]) & (simulated_data_plus['M_TOL']  < bins_edge[i+1])
        for key, value in cols.items():
            simulated_data_plus[key][mask] = simulated_data_plus[key][mask] + delta_plus[key][i]
            simulated_data_minus[key][mask] = simulated_data_minus[key][mask] + delta_minus[key][i]
    return simulated_data_plus, simulated_data_minus, delta_plus, delta_minus



def plot_agn_feedback(delta_minus, delta_plus):
    cols = {'R_corr':[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1]}
    fig, axs = plt.subplots(1, len(cols), figsize=(len(delta_minus)*4, 3))
    i = 0
    for key, value in cols.items():
        ax = axs[i]
        ax.scatter(delta_minus['M_TOL'], delta_minus[key],  marker = 'o', s = 5, lw = 2.0,label="$\delta_-$")
        ax.scatter(delta_plus['M_TOL'], delta_plus[key], marker = 'o', s = 5, lw = 2.0, label="$\delta_+$")
        ax.set_xlabel("Log10(M_TOL/M_SUN)")
        ax.set_ylim(-0.1, 0.1)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_title(key)
        i= i + 1
    plt.legend()
    plt.savefig('./figures/agn_feedback_delta.pdf')


def plot_total_mass(simulated_data):
    simulated_c8a1 = simulated_data[simulated_data['label'] == 'C8a1']
    simulated_c8a2 = simulated_data[simulated_data['label'] == 'C8a2']
    simulated_c8   = simulated_data[simulated_data['label'] == 'C8']

    cols = {'R_corr':[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]}
    bins_edge = np.linspace(13, 15, 41)
    col_name = 'M_TOL'
    sns.histplot(simulated_c8a1[col_name].data, bins=bins_edge, kde=False, color='skyblue', edgecolor='black', stat='density', label='c8a1')
    sns.histplot(simulated_c8a2[col_name].data, bins=bins_edge, kde=False, color='green', edgecolor='black', stat='density', label='c8a2')
    sns.histplot(simulated_c8[col_name].data, bins=bins_edge, kde=False, color='yellow', edgecolor='black', stat='density', label='c8')
    plt.xlabel("Log10(M_TOL/M_SUN)")
    plt.legend()
    plt.savefig('./figures/agn_feedback_total_mass.pdf')