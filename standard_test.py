from MLCI.data_processing import preprocess, boosting, agn_feedback
from MLCI.visualization import comparision, plot, features_importance, comparision_h0weight
from MLCI.models import model
from plot import plot_distribution
import pandas as pd
import numpy as np
from astropy.table import Table

mode = 'RF'
agn = ''
R_type = 'R_corr'
delta = 0.1
model_name = f'{mode}_{R_type}_{agn}_{delta:.1f}'
cols = {R_type:[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]}
print(f"model_name: {model_name} \n", cols)
vars_name = list(cols.keys()) # estimate value
if agn == 'plus':
    simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(delta)
    simulated_data = simulated_data_plus
elif agn == 'minus':
    simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(delta)
    simulated_data = simulated_data_minus
elif agn == '':
    simulated_data=preprocess.simulated_data()
exclude_set = ['HR', 'C1', 'C2', 'C8a1', 'C8a2']
mask = ~np.isin(simulated_data['label'], exclude_set)
simulated_data = simulated_data[mask]
observed_data = preprocess.observed_data()
observed_data = preprocess.filtering(observed_data, cols)
simulated_data= preprocess.filtering(simulated_data, cols)
# boosting data
observed_boost_info, observed_boost_sample = boosting.boosting(observed_data, vars_name, num=100, sampler='multinorm')
simulated_boost_info, simulated_boost_sample = boosting.boosting(simulated_data, vars_name, num=100, sampler='multinorm', max_num=1e4)
boosting.show_boosting_results(observed_boost_info, vars_name)
boosting.show_error_dist(observed_boost_info, vars_name)
boosting.show_boosting_results(simulated_boost_info, vars_name)
boosting.show_error_dist(simulated_boost_info, vars_name)
# show after boosting distribute
observed_boost_sample = preprocess.filtering(observed_boost_sample, cols)
simulated_boost_sample= preprocess.filtering(simulated_boost_sample, cols)
eFEDS = observed_boost_sample[observed_boost_sample['label'] == 'eFEDS']
eRASS1 = observed_boost_sample[observed_boost_sample['label'] == 'eRASS1']
data = [eFEDS, eRASS1, simulated_boost_sample]
print(len(eFEDS), len(eRASS1))
labels = ['eFEDS', 'eRASS1', 'Simualtions']
plot_distribution(data, labels, cols, fname=f'{model_name}')
# model traing 
if mode == 'RF':
    classifier = model.RandomForest()
elif mode == 'NB':
    classifier = model.Naive_Bayesian()
simulated_boost_sample = model.pre_selection(simulated_boost_sample)
train_data, valid_data = model.data_split(simulated_boost_sample)
classifier.train_data = train_data
classifier.config['input_column_rangs'] = cols
classifier.config['model_name'] = model_name
if mode == 'RF':
    classifier.training()
    pass
elif mode == 'NB':
    classifier.calculate_likelihood()
# validations
classifier.test_data = valid_data
results = classifier.points(output_name=f'mix_simulations', show_acc=True)
ture_label = classifier.test_data['label'].values
plot.plot_corner(results, ref_point='C8', test_set='Mix', fname=f'mix_simulations', show_cos_name=False)
plot.plot_confusion_matrix(ture_label, results, title=classifier.config['model_name'])
plot.plot_detection_prob(observed_data, fname=model_name)
# prediction
classifier.test_data = observed_boost_sample
classifier.config['weight'] = True
classifier.config['p_threshold'] = 0.0
results = classifier.points(output_name=f"observation")
fname = f"{classifier.config['model_name']}_observation"
plot.plot_corner(results, ref_point='C8', test_set='observed_samples', fname=fname, show_cos_name=False)
plot.plot_corner(results, ref_point='Planck2018', smooth=1.0, test_set='observed_samples', fname=fname+'_updata', show_cos_name=False, show_pdf=False,show_datapoint=False)

redshift_bin = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
results = pd.read_csv(f"./results/{classifier.config['model_name']}_observation.csv")
for i in range(len(redshift_bin)-1):
    subset = results[(results['z'] >= redshift_bin[i]) & (results['z'] < redshift_bin[i+1])]
    print(f"redshift: {redshift_bin[i]} - {redshift_bin[i+1]}, num={len(subset)}")
    fname = f"{model_name}_z={redshift_bin[i]:.1f}_{redshift_bin[i+1]:.1f}"
    plot.plot_corner(subset, ref_point='C8', test_set='boost_sample', fname=fname, show_cos_name=False)

z_0, z_1 = 0.0, 0.8
subset = results[(results['z'] >= z_0) & (results['z'] < z_1)]
catalog_names ={f"{model_name}_z={z_0:.1f}_{z_1:.1f}": subset,}
comparision.RF_smooth = 0.8
comparision.compare(catalog_names, fname=f"{model_name}_z={z_0:.1f}_{z_1:.1f}", is_csv=False)

# plot the cosmological parameters vs redshift
bins_cosmology = []
redshift_bin = np.array([0.0, 0.2, 0.4, 0.6,  0.8, 1.0])
results = catalog = pd.read_csv(f"./results/{model_name}_observation.csv")
for i in range(len(redshift_bin)-1):
    subset = results[(results['z'] >= redshift_bin[i]) & (results['z'] < redshift_bin[i+1])]
    print(f"redshift: {redshift_bin[i]} - {redshift_bin[i+1]}, num={len(subset)}")
    fname = f"RF_{model_name}_z={redshift_bin[i]:.1f}_{redshift_bin[i+1]:.1f}"
    cosmology_para = plot.get_expectation_1sigam(subset, model_name, model_name)
    cosmology_para['z'] = (redshift_bin[i] + redshift_bin[i+1])/2
    bins_cosmology.append(cosmology_para)

bins_cosmology = Table(bins_cosmology)
aveg_cosmology = plot.get_expectation_1sigam(results, model_name, model_name)
plot.plot_cosmology_paras_vs_z(bins_cosmology, aveg_cosmology, fname=model_name)

# print the final values of cosmologcal parameters in the redshift ranges [0.0, 0.8]
para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
results = pd.read_csv(f"./results/{model_name}_observation.csv")
subset = results[(results['z'] >= 0.0) & (results['z'] < 0.8)]
cosmology_para = plot.get_expectation_1sigam(results, model_name, model_name)
for para_name, label in zip(para_names, label_names):
    y = cosmology_para[f'{para_name}_medians']
    yerr_lower = cosmology_para[f'{para_name}_lower_errors']
    yerr_upper = cosmology_para[f'{para_name}_upper_errors']
    print(f"${label.strip('$')}={y:.3f}^{{{yerr_upper:.3f}}}_{{{yerr_lower:.3f}}}$")

# features importance
impur_importance = features_importance.get_impurity_importance(model_name, column_rangs=cols)
permu_importances = features_importance.get_permutation_importance(model_name, valid_data, column_rangs=cols)
#=== normalize the permu_importances
norm = np.sum(permu_importances['permutation_importance'])
permu_importances['permutation_importance'] = permu_importances['permutation_importance']/norm
permu_importances['permutation_importance_std'] = permu_importances['permutation_importance_std']/norm

importances = impur_importance.merge(permu_importances, on='Feature', how='left')
importances.to_csv(f'./results/{model_name}_feature_importance.csv')
importances = pd.read_csv(f'./results/{model_name}_feature_importance.csv')
features_importance.plot(importances)