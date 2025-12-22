import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy.table import Table 
import getdist.plots as gplot
from getdist import plots, MCSamples
import corner
from matplotlib.ticker import FormatStrFormatter
#sns.set_theme(style="ticks", palette="deep")
#sns.set_theme(style="darkgrid", palette="bright")
#sns.set_theme(style="ticks", palette="bright")
sns.set_theme(style="ticks", palette="colorblind")

import matplotlib.pyplot as plt

colors = plt.cm.tab10.colors      # 10 种鲜艳的分类颜色
colors = plt.cm.Set1.colors       # 9 种强烈对比的颜色
colors = plt.cm.Dark2.colors      # 偏暗但对比度高

RF_smooth = 0.7
alpha_filled = 0.8
linewidth_contour = 1.2

def read_samples(fanme, col_names, names, labels, is_catalog=False):
    if is_catalog:
        catalog = fanme
    else:
        catalog = pd.read_csv(fanme)
    var1, var2 = catalog[col_names[0]].values, catalog[col_names[1]].values
    if col_names[2] == 'None':
        weights = np.ones_like(catalog[col_names[1]].values)
    else:
        weights = catalog[col_names[2]].values
    samples = MCSamples(samples=[var1,var2], weights=weights, names=names, labels=labels, settings={'smooth_scale_2D':RF_smooth, 'smooth_scale_1D':RF_smooth})
    return samples

# example of code see: 
#1 https://cosmologist.info/cosmomc/readme_python.html
#2 https://cosmologist.info/cosmomc/readme_planck.html
#3 https://cosmologist.info/cosmomc/readme_gui.html
# Chains dir download from link: 
#1 https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters#File_formats
#2 https://www.cosmos.esa.int/web/planck/pla



# 当前目录 # Need to add the absolute location
cwd = os.getcwd()
chain_dir = os.path.join(cwd, 'survey/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3/')

def read_plank_data(names, labels):
    g = gplot.getSinglePlotter(chain_dir=chain_dir)
    #samples = g.sample_analyser.samples_for_root('base_plikHM_TTTEEE_lowl_lowE')
    samples = g.sample_analyser.samples_for_root('base_plikHM_TTTEEE_lowl_lowE_lensing')
    p = samples.getParams()
    data_points = []
    # parameters names listed in file: base_plikHM_TTTEEE_lowl_lowE.paramnames
    for name in names:
        if name == 'omega_m':
            data_points.append(p.omegam)
        elif name == 'sigma_8':
            data_points.append(p.sigma8)
        elif name == 'omega_b': # Note: Only approximate consideration
            data_points.append(p.omegabh2/(p.H0/100)**2)
        elif name == 'h0':
            data_points.append(p.H0/100)
    samples = MCSamples(samples=data_points, weights=samples.weights, names=names, labels=labels, settings={'smooth_scale_2D':0.5})
    return samples

# https://arxiv.org/pdf/2503.19442
# Fig. 10
def read_KiDS_Legacy(names, labels):
    fname = os.path.join(cwd, 'survey/KiDS_Legacy_fiducial_COSEBIs_nautilus_B_massdep.txt')
    data = pd.read_csv(fname, delim_whitespace=True)
    weights = np.exp(data['log_weight'].values)
    data_points = []
    for name in names:
        if name == 'omega_m':
            data_points.append(data['OMEGA_M'].values)
        elif name == 'sigma_8':
            data_points.append(data['SIGMA_8'].values)
        elif name == 'omega_b': # No this data
            pass
        elif name == 'h0':
            data_points.append(data['h0'].values)
    samples = MCSamples(samples=data_points, weights=weights, names=names, labels=labels, settings={'smooth_scale_2D':0.5})
    return samples

import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
import os

def read_desi_bao_bbn_acoustic(cosmo_params=['omega_m', 'h0'], paras_label=['\Omega_m','h_0']):
    # 你的 chain 目录
    chain_dir = r'd:\OneDrive\桌面\eROSTA\survey\desi-bao-all_schoneberg2024-bbn_planck2018-thetastar-fixed-marg-nnu'
    # 参数名（从文件头部提取）
    param_names = ['ombh2', 'omch2', 'theta_s_100', 'As', 'h0', 'omega_m', 
                'omegamh2', 'omegal', 'zrei', 'YHe', 'Y_p', 'DHBBN', 
                'A', 'clamp', 'age', 'rdrag', 'zdrag', 'H0rdrag',
                'chi2__BAO', 'chi2__bbn', 'chi2__thetastar', 
                'minuslogprior', 'minuslogprior__0', 'chi2',
                'chi2__desi_bao', 'chi2__bbn_like', 'chi2__thetastar_like']
    # 读取所有 chain
    all_data = []
    for i in range(1, 5):
        data = np.loadtxt(os.path.join(chain_dir, f'chain.{i}.txt'))
        all_data.append(data)
        print(f"DESI Chain {i}: {data.shape[0]} 样本")

    combined = np.vstack(all_data)
    print(f"DESI 总样本数: {combined.shape[0]}")
    # 更简单的方法：直接创建只包含这些参数的新样本对象
    # 提取参数数据
    param_indices = [param_names.index(p) for p in cosmo_params]
    selected_data = combined[:, [2 + idx for idx in param_indices]]
    
    # 对于 'h' 参数，除以 100
    for i, param in enumerate(cosmo_params):
        if param == 'h0':
            selected_data[:, i] = selected_data[:, i] / 100.0
            print(f"已将 H0 转换为 h (H0/100)")
    
    # 创建样本对象
    samples = MCSamples(
        samples=selected_data,
        weights=combined[:, 0],
        loglikes=-combined[:, 1],
        names=cosmo_params,
        labels=paras_label
    )
    return samples



def lable_sim_points(sim_catalog, col_names, ax):
    cos_labels = sim_catalog['name'].values
    for j, label in enumerate(cos_labels):
        x = sim_catalog[sim_catalog['name'] == label][col_names[0]]
        y = sim_catalog[sim_catalog['name'] == label][col_names[1]]
        ax.text(x, y, label, fontsize=8, ha='right', va='top', color='black',alpha=1.0)

def add_surveys(survey_cat, col_names, ax):
    for i, row in survey_cat.iterrows():
        if pd.isna(row[col_names[0]]) or pd.isna(row[col_names[1]]): #missing values
            continue
        xpoint, ypoint = row[col_names[0]].split('|'), row[col_names[1]].split('|')
        x, y = [float(xpoint[0])], [float(ypoint[0])]
        xerr = ([float(xpoint[2])], [float(xpoint[1])]) 
        yerr = ([float(ypoint[2])], [float(ypoint[1])])
        survey_name = row['survey'].split('-', 1)[-1]
        scater = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="", capsize=5, label=survey_name, lw=2.0)
    legends = ax.legend()
    return legends

def updata_legend(legends2, legends1, ax, loc='upper right'):
    #handles1, legend_text1, colors1 = legends1.legendHandles, [text.get_text() for text in legends1.get_texts()], [text.get_color() for text in legends1.get_texts()]
    handles1, legend_text1, colors1 = legends1.legend_handles, [text.get_text() for text in legends1.get_texts()], [text.get_color() for text in legends1.get_texts()]
    print(handles1, legend_text1, colors1)
    #handles1, legend_text1 = ax.get_legend_handles_labels()
    #handles2, legend_text2, colors2 = legends2.legendHandles, [text.get_text() for text in legends2.get_texts()], [text.get_color() for text in legends2.get_texts()]
    handles2, legend_text2, colors2 = legends2.legend_handles, [text.get_text() for text in legends2.get_texts()], [text.get_color() for text in legends2.get_texts()]
    handles, legend_text, colors = handles1+handles2, legend_text1+legend_text2, colors1+colors2
    new_legend = ax.legend(handles=handles, labels=legend_text, loc=loc, framealpha=0.1, prop={'size': 8, 'weight': 'bold'})
    for text, color in zip(new_legend.get_texts(), colors):
        text.set_color(color)  # Set legend text color
    return handles, legend_text


import matplotlib.lines as mlines
def add_corner(catalogs, label_names, ax, smooth, lims, color="blue"):
    key = list(catalogs.keys())[0]
    samples, weights = np.array(catalogs[key][label_names]), catalogs[key]['weight']
    range = [(lims[0], lims[1]), (lims[2], lims[3])]
    corner.hist2d(
                    samples[:,0], samples[:,1],
                    weights=weights,
                    ax=ax,
                    color=color,
                    plot_contours=True,
                    no_fill_contours = True,
                    #fill_contours=False,
                    alpha=0.01,
                    levels=(0.68, 0.95),
                    smooth = smooth,
                    range = range,
                    bins=95,
                    plot_datapoints=False,
                    contourf_kwargs={'alpha':0.01, 'colors':'skyblue'}, 
                    contour_kwargs={'colors':color},  
                )
    

def compare(catalog_names, fname='./figures/test_compare.pdf', is_csv=False, truncation=False):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
    #=== read eFEDS and eRASS1 catalog
    catalogs = {}
    for name, fn in catalog_names.items():
        if is_csv:
            catalogs[name] = pd.read_csv(fn)
        else:
            catalogs[name] = fn
        #=== truncation
        if truncation:
            omegam_max = 0.35
            width = 0.10
            masks = (catalogs[name]['Omega'] >= omegam_max)
            catalogs[name]['weight'][masks] = np.exp(-((catalogs[name]['weight'][masks]-omegam_max)/width)**2)
    #=== other catalogs
    extra_surveys = pd.read_csv('./survey/more_surveys.csv')
    #===== Omega_m & sigma8
    names,labels = ['omega_m','sigma_8'],['\Omega_m','\sigma_8']
    sim_catalog = pd.read_csv('./simulation_paras.csv')
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'Sigm8', 'None'], names, labels)
    #samples_DES_1Y = read_samples('survey/DES 1Y.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_DES_Y3 = read_samples('survey/DES_Y3_kids1000.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_kid1000 = read_samples('survey/kid1000.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_kid1000_2x3pt = read_samples('survey/kid1000+2x3pt.csv', ['omega_m', 'sigma_8', 'weights'], names,labels)
    samples_RF = [read_samples(catalog, ['Omega', 'Sigm8', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_Plank = read_plank_data(names, labels)
    samples_KiDS_Legacy = read_KiDS_Legacy(names, labels)
    samples_list = [samples_Plank, samples_DES_Y3, samples_kid1000, samples_kid1000_2x3pt, samples_KiDS_Legacy] + samples_RF
    samples_legends = ['Plank2018', 'DES Y3+KiDS-1000 ', 'KiDS-1000', 'KiDS-1000+2x3pt', 'KiDS-Legacy'] + samples_RF_label
    color_text = False
    ax = axs[0]
    lims=[0.1, 0.6, 0.5, 1.1]
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims, contour_levels=[0.68,0.95])
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0], names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'Sigm8'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'sigma8'], ax)
    updata_legend(legends1, legends2, ax)

    #==== Omega_m & h0
    names, labels = ['omega_m','h0'],['\Omega_m','h0']
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'Hubble', 'None'], names, labels)
    samples_RF = [read_samples(catalog, ['Omega', 'Hubble', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_Plank = read_plank_data(names, labels)
    samples_list = [samples_Plank] + samples_RF
    samples_legends = ['Plank2018'] + samples_RF_label

    ax = axs[1]
    lims = [0.1, 0.60, 0.55, 0.80]
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims)
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0],names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'Hubble'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'h0'], ax)
    updata_legend(legends1, legends2, ax)

    #===== Omega_m & Omega_b
    names,labels = ['omega_m','omega_b'],['\Omega_m','\Omega_b']
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'OmegaB', 'None'], names, labels)
    samples_RF = [read_samples(catalog, ['Omega', 'OmegaB', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_Plank = read_plank_data(names, labels)
    samples_list = [samples_Plank] + samples_RF
    samples_legends = ['Plank2018'] + samples_RF_label

    ax = axs[2]
    lims = [0.12, 0.45, 0.03, 0.065]
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims)
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0],names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'OmegaB'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'omegab'], ax)
    updata_legend(legends1, legends2, ax, loc='upper left')
    fig.savefig(fname, bbox_inches='tight')
    print('##### finish!!!! #####')



def new_compare(catalog_names, fname='./figures/test_compare.pdf', is_csv=False, add_contour=False):
    
    #=== read eFEDS and eRASS1 catalog
    catalogs = {}
    for name, fn in catalog_names.items():
        if is_csv:
            catalogs[name] = pd.read_csv(fn)
        else:
            catalogs[name] = fn

    #=== other catalogs
    extra_surveys = pd.read_csv('./survey/more_surveys.csv')
    #===== Omega_m & sigma8
    names,labels = ['omega_m','sigma_8'],['\Omega_m','\sigma_8']
    sim_catalog = pd.read_csv('./simulation_paras.csv')
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'Sigm8', 'None'], names, labels)
    #samples_DES_1Y = read_samples('survey/DES 1Y.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_DES_Y3 = read_samples('survey/DES_Y3_kids1000.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_kid1000 = read_samples('survey/kid1000.csv', ['omega_m', 'sigma_8', 'weights'], names, labels)
    samples_kid1000_2x3pt = read_samples('survey/kid1000+2x3pt.csv', ['omega_m', 'sigma_8', 'weights'], names,labels)
    samples_RF = [read_samples(catalog, ['Omega', 'Sigm8', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_Plank = read_plank_data(names, labels)
    samples_KiDS_Legacy = read_KiDS_Legacy(names, labels)
    samples_list = [samples_kid1000, samples_DES_Y3, samples_KiDS_Legacy, samples_kid1000_2x3pt, samples_Plank, ]
    samples_legends = ['KiDS-1000', 'DES Y3+KiDS-1000 ', 'KiDS-Legacy', 'KiDS-1000+2x3pt', 'Plank2018']
    color_list = ['#CC78BC', '#029E73', '#D55E00',"#949494", '#0173B2']
    if add_contour:
        samples_list = samples_list + samples_RF
        samples_legends = samples_legends +  + samples_RF_label
    color_text = False
    RF_color = "black"
    fig, axs = plt.subplots(1,3, figsize=(18, 5), dpi=160)
    ax = axs[0]
    lims=[0.1, 0.6, 0.5, 1.1]
    
    add_corner(catalogs,  ['Omega', 'Sigm8'], ax, smooth=3*RF_smooth, lims=lims, color=RF_color)
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims, contour_levels=[0.68,0.95], colors=color_list,  contour_colors=color_list)
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0], names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'Sigm8'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'sigma8'], ax)
    handles, legend_text = updata_legend(legends1, legends2, ax)
    
    blue_line = mlines.Line2D([], [], color=RF_color, alpha=1.0, label='this work')
    handles.append(blue_line)
    legend_text.append('this work')
    ax.legend(handles=handles, labels=legend_text, loc='upper right', framealpha=0.0, prop={'size': 8, 'weight': 'bold'})
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='both', labelsize=12)

    #==== Omega_m & h0
    names, labels = ['omega_m','h0'],['\Omega_m','h_0']
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'Hubble', 'None'], names, labels)
    samples_RF = [read_samples(catalog, ['Omega', 'Hubble', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_desi = read_desi_bao_bbn_acoustic()
    samples_Plank = read_plank_data(names, labels)
    samples_list = [samples_desi, samples_Plank]
    samples_legends = ['DESI DR1', 'Plank2018']
    color_list = ['#C44E52', '#0173B2' ]
    if add_contour:
        samples_list = samples_list + samples_RF
        samples_legends = samples_legends + samples_RF_label

    ax = axs[1]
    lims = [0.1, 0.60, 0.55, 0.80]
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims, contour_levels=[0.68,0.95], colors=color_list,  contour_colors=color_list)
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0],names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'Hubble'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'h0'], ax)
    handles, legend_text = updata_legend(legends1, legends2, ax)
    add_corner(catalogs,  ['Omega', 'Hubble'], ax, smooth=1.8, lims=lims, color=RF_color)
    blue_line = mlines.Line2D([], [], color=RF_color, alpha=1.0, label='this work')
    handles.append(blue_line)
    legend_text.append('this work')
    ax.legend(handles=handles, labels=legend_text, loc='upper right', framealpha=0.0, prop={'size': 10, 'weight': 'bold'},)
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='both', labelsize=12)

    #===== Omega_m & Omega_b
    names,labels = ['omega_m','omega_b'],['\Omega_m','\Omega_b']
    samples_sim = read_samples('simulation_paras.csv', ['Omega', 'OmegaB', 'None'], names, labels)
    samples_RF = [read_samples(catalog, ['Omega', 'OmegaB', 'weight'], names,labels, is_catalog=True) for name, catalog in catalogs.items()]
    samples_RF_label = [name for name, catalog in catalogs.items()]
    samples_Plank = read_plank_data(names, labels)
    samples_list = [samples_Plank]
    samples_legends = ['Plank2018']
    if add_contour:
        samples_list = samples_list + samples_RF
        samples_legends = samples_legends +  + samples_RF_label
    ax = axs[2]
    lims = [0.12, 0.45, 0.03, 0.065]
    g = plots.get_single_plotter()
    g.settings.alpha_filled_add = alpha_filled
    g.settings.linewidth_contour = linewidth_contour
    g.plot_2d(samples_list, names, ax=ax, filled=True, lims=lims, contour_levels=[0.68,0.95])
    legends1 = g.add_legend(samples_legends, ax=ax, colored_text=color_text, legend_loc='upper right')
    g.plot_2d_scatter(samples_sim, names[0],names[1], ax=ax, color='blue', scatter_size=30, lims=lims)
    lable_sim_points(sim_catalog, ['Omega', 'OmegaB'], ax)
    legends2 = add_surveys(extra_surveys, ['omegam', 'omegab'], ax)
    handles, legend_text = updata_legend(legends1, legends2, ax)
    add_corner(catalogs,  ['Omega', 'OmegaB'], ax, smooth=2*RF_smooth, lims=lims, color=RF_color)
    blue_line = mlines.Line2D([], [], color=RF_color, alpha=1.0, label='this work')
    handles.append(blue_line)
    legend_text.append('this work')
    ax.legend(handles=handles, labels=legend_text, loc='upper left', framealpha=0.0, prop={'size': 12, 'weight': 'bold'})
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.savefig(fname, bbox_inches='tight')
    print('##### finish!!!! #####')
