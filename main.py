#import preprocess, boosting, comparision, model, plot, features_importance, agn_feedback
from MLCI.data_processing import preprocess, boosting, agn_feedback
from MLCI.visualization import comparision, plot, features_importance, comparision_h0weight
from MLCI.models import model
import pandas as pd
import numpy as np
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde
import yaml, os, time
from datetime import datetime

def exclude_out(xlabel, ylabel, data, percent=5):
    x = data[xlabel].data
    y = data[ylabel].data
    xy = np.vstack([x, y]) # 计算 2D 核密度估计值
    print(xy.shape)
    kde = gaussian_kde(xy)
    z = kde(xy)  # 每个点的密度值
    # seaborn 的 kdeplot 默认是密度累积分布等值线，可用百分位估计
    z_threshold = np.percentile(z, percent) 
    mask_inside = z >= z_threshold  # 位于高密度区域内的点
    data_slectded = data[mask_inside]
    return data_slectded

def test_configuration(model_name, config, training):
    output_txt = ''
    start_time = time.time()
    output_line = f"Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    output_txt = output_txt + output_line + '\n'
    print(output_line)
    print(config)
    config['model_name'] = model_name
    #cols = {'R_corr':[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]}
    cols = config['cols']
    #======
    print(f"model_name: {config['model_name']} \n", cols)
    vars_name = list(cols.keys())
    if not os.path.exists(f"models/{config['model_name']}"):
        os.mkdir(f"models/{config['model_name']}")
    if not os.path.exists(f"figures/{config['model_name']}"):
        os.mkdir(f"figures/{config['model_name']}")
    if not os.path.exists(f"results/{config['model_name']}"):
        os.mkdir(f"results/{config['model_name']}")
    
    # using different feedback
    if isinstance(config['agn_feedback'], float):
        if config['agn_feedback'] > 0.0:
            simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(abs(config['agn_feedback']), proportional=config['proportional'])
            simulated_data = simulated_data_plus
        elif config['agn_feedback'] < 0.0:
            simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(abs(config['agn_feedback']), proportional=config['proportional'])
            simulated_data = simulated_data_minus
        else: # using different noise
            simulated_data=preprocess.simulated_data(fac1=config['noise_level'][0], fac2=config['noise_level'][1])

    elif config['agn_feedback'] == '+delta':
        simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(delta_prop=config['delta_prop'])
        simulated_data = simulated_data_plus
    elif config['agn_feedback'] == '-delta':
        simulated_data_plus, simulated_data_minus, delta_plus, delta_minus = agn_feedback.agn_corrected_data(delta_prop=config['delta_prop'])
        simulated_data = simulated_data_minus
    

    exclude_set = ['HR', 'C8a1', 'C8a2']
    mask = ~np.isin(simulated_data['label'], exclude_set)
    simulated_data = simulated_data[mask]
    observed_data = preprocess.observed_data()
    observed_data = preprocess.filtering(observed_data, cols)
    simulated_data= preprocess.filtering(simulated_data, cols)
    # === exclude outlier
    if config['exclude_outlier']:
        xlabel, ylabel = 'T', 'R_corr'
        observed_data = exclude_out(xlabel, ylabel, observed_data, percent=5)
        '''
        new_simulated_data = []
        for key in [f'C{i+3}' for i in range(13)]:
            temp_subset = simulated_data[simulated_data['label'] == key]
            temp_subset = exclude_out(xlabel, ylabel, temp_subset, percent=5)
            new_simulated_data.append(temp_subset)
        simulated_data = vstack(new_simulated_data)
        '''
    # boosting data
    observed_boost_info, observed_boost_sample = boosting.boosting(observed_data, vars_name, num=config['sample_num'], sampler='multinorm')
    simulated_boost_info, simulated_boost_sample = boosting.boosting(simulated_data, vars_name, num=config['sample_num'], sampler='multinorm', max_num=config['max_num'])
    boosting.show_boosting_results(observed_boost_info, vars_name, fname=f"./figures/{config['model_name']}/observed_boosting.pdf")
    boosting.show_error_dist(observed_boost_info, vars_name, fname=f"./figures/{config['model_name']}/observed_boosting_error_dis.pdf")
    boosting.show_boosting_results(simulated_boost_info, vars_name, fname=f"./figures/{config['model_name']}/simulated_boosting.pdf")
    boosting.show_error_dist(simulated_boost_info, vars_name, fname=f"./figures/{config['model_name']}/simulated_boosting_error_dis.pdf")
    # show after boosting distribute
    observed_boost_sample = preprocess.filtering(observed_boost_sample, cols)
    simulated_boost_sample= preprocess.filtering(simulated_boost_sample, cols)
    eFEDS = observed_boost_sample[observed_boost_sample['label'] == 'eFEDS']
    eRASS1 = observed_boost_sample[observed_boost_sample['label'] == 'eRASS1']
    data = [eFEDS, eRASS1, simulated_boost_sample]
    labels = ['eFEDS', 'eRASS1', 'Simualtions']
    output_line = f"Dataset_num: {labels} {len(eFEDS)}-{len(eRASS1)}-{len(simulated_boost_sample)}"
    print(output_line)
    output_txt = output_txt + output_line +'\n'
    plot.plot_distribution(data, labels, cols, fname=f"figures/{config['model_name']}/features_dis.pdf")
    # model traing 
    if config['mode'] == 'RF':
        model.simulation_labels = config['simulations']
        classifier = model.RandomForest()
        classifier.config['min_samples_leaf'] = config['leaf_size']
        classifier.config['max_depth'] = config['max_depth']
        classifier.config['sample_num'] = config['sample_num']
        classifier.config['max_num'] = config['max_num']
        classifier.config['noise_level'] = config['noise_level']
        classifier.config['agn_feedback'] = config['agn_feedback']
    elif config['mode'] == 'NB':
        classifier = model.Naive_Bayesian()
    simulated_boost_sample = model.pre_selection(simulated_boost_sample)
    train_data, valid_data = model.data_split(simulated_boost_sample)
    classifier.train_data = train_data
    classifier.config['input_column_rangs'] = cols
    classifier.config['model_name'] = model_name
    if training:
        classifier.training()
    # validations
    classifier.test_data = valid_data
    results = classifier.points(output_name=f"./results/{config['model_name']}/mix_simulations.csv", show_acc=True)
    ture_label = classifier.test_data['label'].values
    plot.plot_corner(results, ref_point='C8', test_set='Mix', fname=f"figures/{config['model_name']}/mix_simulations.pdf", show_cos_name=False)
    confusion_matrix = plot.plot_confusion_matrix(results, title=classifier.config['model_name'], fname=f"figures/{config['model_name']}/CM.pdf")
    plot.plot_detection_prob([observed_data], labels=['eFEDS+DR1'], colors=['red'], fname=f"figures/{config['model_name']}/detection_prob.pdf")
    
    # prediction and show contours
    if config['boost_real']:
        classifier.test_data = observed_boost_sample
    else:
        classifier.test_data = observed_data
    classifier.config['weight'] = True
    classifier.config['p_threshold'] = 0.0
    results = classifier.points(output_name=f"./results/{config['model_name']}/observation.csv", confusion_matrix=confusion_matrix)
    plot.plot_corner(results, ref_point='C8', test_set='observed_samples', fname=f"figures/{config['model_name']}/observation.pdf", show_cos_name=False)
    plot.plot_corner(results, ref_point='Planck2018', smooth=1.0, test_set='observed_samples', fname=f'figures/{config['model_name']}/observation_Planck2018.pdf', show_cos_name=False, show_pdf=False,show_datapoint=False)
    
    redshift_bin = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
    results = pd.read_csv(f"./results/{config['model_name']}/observation.csv")
    for i in range(len(redshift_bin)-1):
        subset = results[(results['z'] >= redshift_bin[i]) & (results['z'] < redshift_bin[i+1])]
        if len(subset) <= config['sample_num']:
            continue
        print(f"redshift: {redshift_bin[i]} - {redshift_bin[i+1]}, num={len(subset)}")
        fname = f"figures/{config['model_name']}/observation_z={redshift_bin[i]:.1f}_{redshift_bin[i+1]:.1f}.pdf"
        plot.plot_corner(subset, ref_point='Planck2018', test_set='observed_samples', fname=fname, show_cos_name=False)

    #compare with other survey
    z_0, z_1 = 0.1, 0.8
    subset = results[(results['z'] >= z_0) & (results['z'] < z_1)]
    catalog_names ={f"{model_name}_z={z_0:.1f}_{z_1:.1f}": subset,}
    comparision.RF_smooth = 0.6
    comparision.compare(catalog_names, fname=f"./figures/{model_name}/compare_z={z_0:.1f}_{z_1:.1f}.pdf", is_csv=False)
    plot.plot_cosmology_paras_vs_z(model_name, z_bins_edge=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fname=f'./figures/{model_name}/Cosmological_Parameters_vs_z.pdf')
    # print the final values of cosmologcal parameters in the redshift ranges [0.1, 0.8]
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    results = pd.read_csv(f"./results/{model_name}/observation.csv")
    subset = results[(results['z'] >= z_0) & (results['z'] < z_1)]
    cosmology_para = plot.get_expectation_1sigam(results, model_name, model_name)
    for para_name, label in zip(para_names, label_names):
        y = cosmology_para[f'{para_name}_medians']
        yerr_lower = cosmology_para[f'{para_name}_lower_errors']
        yerr_upper = cosmology_para[f'{para_name}_upper_errors']
        output_line = f"redshift {z_0:.1f}-{z_1:.1f}: ${label.strip('$')}={y:.3f}^{{{yerr_upper:.3f}}}_{{{yerr_lower:.3f}}}$"
        output_txt = output_txt + output_line + '\n'
        print(output_line)

    # features importance
    if config['mode'] == 'RF':
        impur_importance = features_importance.get_impurity_importance(model_name, column_rangs=cols)
        permu_importances = features_importance.get_permutation_importance(model_name, valid_data, column_rangs=cols)
        #=== normalize the permu_importances
        norm = np.sum(permu_importances['permutation_importance'])
        permu_importances['permutation_importance'] = permu_importances['permutation_importance']/norm
        permu_importances['permutation_importance_std'] = permu_importances['permutation_importance_std']/norm
        importances = impur_importance.merge(permu_importances, on='Feature', how='left')
        importances.to_csv(f'./results/{model_name}/feature_importance.csv')
        importances = pd.read_csv(f'./results/{model_name}/feature_importance.csv')
        features_importance.plot(f'./figures/{model_name}/feature_importance.pdf', importances)

    output_line = f"End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    output_txt = output_txt + output_line + '\n'
    print(output_line)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    output_line = f"Elapsed time: {elapsed_time:.3f} min"
    print(output_line)
    output_txt = output_txt + output_line + '\n'
    # save as a txt 
    with open(f"./results/{model_name}/output.txt", "w") as f:
        f.write(output_txt + "\n")

import re
def extract_last_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


if __name__ == "__main__":
    file_name = './config/default_test_para.yaml'
    with open(file_name, 'r') as file:
        test_set = yaml.safe_load(file)

    for model_name, config in test_set.items():
        print('Ready testing:', model_name, config)
        test_num = extract_last_number(model_name)
        if test_num >= 25:
            test_configuration(model_name, config, training=False)