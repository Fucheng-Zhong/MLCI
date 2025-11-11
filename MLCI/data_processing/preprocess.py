import numpy as np
from astropy.table import Table, vstack
import pandas as pd
from astropy.cosmology import Planck18 as cosmo  
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.cosmology import Planck18


def cosmology(h0, Om0):
    H0 = h0*100* u.km / u.s / u.Mpc
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    return cosmo

fiducial_cosmolgy = cosmology(h0=0.7, Om0=0.3)
observation_error = {'z_e':0.05, 'z_E':0.05}
z_error = 0.05

'''
eFEDS/eROSTA DR1 clusters
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/661/A7#/browse
https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_cosmology_v1.1.html
https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/
https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_primary_v3.2.html
'''

'''
finding the 0.5% and 99.5% points of the column
'''
def filter_paras(column_rangs, simualtion, low_limit=0.01, upper_limit=99.9):
    data = simualtion[simualtion['label']!='HR']
    for key in column_rangs.keys():
        #column_rangs[key] = [np.min(simualtion[key]), np.max(simualtion[key])]
        column_rangs[key] = [np.percentile(data[key].data, low_limit), np.percentile(data[key].data, upper_limit)] 
    return column_rangs
'''
select the observation
1. Within the filters' ranges
2. No missing values
3. Has error measurements
'''

def filtering(data, col_rangs):
    # The nan/inf error will be replaced by the median number error
    # check the range
    print(f'Total: {len(data)}')
    for col_name, value_range in col_rangs.items():
        mask = (data[col_name] >= value_range[0]) & (data[col_name] <= value_range[1])
        data = data[mask]
        print(f'{col_name} in range: {len(data)}')

    #remove no error case
    for col_name, value_range in col_rangs.items():
            if f'{col_name}_E' in data.keys() and f'{col_name}_e' in data.keys():
                err_plus_name = f'{col_name}_E'
                err_plus_mask = (data[err_plus_name]>0) & np.isfinite(data[err_plus_name])
                err_minus_name = f'{col_name}_e'
                err_minus_mask = (data[err_minus_name]>0) & np.isfinite(data[err_minus_name])
                data = data[err_plus_mask & err_minus_mask]
    print(f'Total after filtering: {len(data)}')
    return data



def observed_data():
    table1 = Table([], names=[])
    table2 = Table([], names=[])
    # best-fit cosmology in eROSITA, see https://arxiv.org/pdf/2402.08458
    cosmo_bestfit = cosmology(Planck18.H0.value/100, 0.29)
    #=== eFEDS
    eFEDS_cat2 = Table.read('./Data/J_A+A_661_A7/J_A+A_661_A7_table2.dat.fits', 1)
    table1['ID'] = eFEDS_cat2['ID']
    table1['RA'] = eFEDS_cat2['RAdeg']
    table1['DEC'] = eFEDS_cat2['DEdeg']
    z = eFEDS_cat2['z']
    cosmo_bestfit_rho_z = cosmo_bestfit.critical_density(z).to(u.Msun/u.kpc**3)
    R500 = 3/(4*np.pi)*(eFEDS_cat2['Mgas500'].data*1e12*u.Msun/cosmo_bestfit_rho_z)**(1/3)
    table1['R'] = np.log10(R500.data)
    R500_lower = 3/(4*np.pi)*((eFEDS_cat2['Mgas500'].data-eFEDS_cat2['e_Mgas500'].data)*1e12*u.Msun/cosmo_bestfit_rho_z)**(1/3)
    table1['R_e'] =  table1['R'].data - np.log10(R500_lower.data)
    R500_upper = 3/(4*np.pi)*((eFEDS_cat2['Mgas500'].data+eFEDS_cat2['E_Mgas500'].data)*1e12*u.Msun/cosmo_bestfit_rho_z)**(1/3)
    table1['R_E'] =  np.log10(R500_upper.data) - table1['R'].data
    table1['Mgas'] = np.log10(eFEDS_cat2['Mgas500'].data*1e12)
    table1['Mgas_e'] = -np.log10(1 - eFEDS_cat2['e_Mgas500'].data/eFEDS_cat2['Mgas500'].data)
    table1['Mgas_E'] = np.log10(1+eFEDS_cat2['E_Mgas500'].data/eFEDS_cat2['Mgas500'].data)
    table1['L'] = np.log10(eFEDS_cat2['Lbol500'].data)+42 # 10**42 erg s-1
    table1['L_e'] = -np.log10(1-eFEDS_cat2['e_Lbol500'].data/eFEDS_cat2['Lbol500'].data)
    table1['L_E'] = np.log10(1+eFEDS_cat2['E_Lbol500'].data/eFEDS_cat2['Lbol500'].data)
    table1['T'] = np.log10(eFEDS_cat2['T500'].data)
    table1['T_e'] = table1['T'] - np.log10(eFEDS_cat2['T500'].data - eFEDS_cat2['e_T500'].data)
    table1['T_E'] = np.log10(eFEDS_cat2['T500'].data + eFEDS_cat2['E_T500'].data) - table1['T']
    table1['z'] = z
    table1['VDIS'] = -1
    table1['VDIS_e'] = table1['VDIS_E'] = -1
    table1['label'] = 'eFEDS'
    #=== DR1
    erass1 = Table.read('./Data/erass1cl_primary_v3.2.fits',1)
    table2['ID'] = erass1['DETUID']
    table2['RA'] = erass1['RA']
    table2['DEC'] = erass1['DEC']
    table2['R'] = np.log10(erass1['R500'].data)
    table2['R_e'] = np.log10(erass1['R500'].data)-np.log10(erass1['R500_L'].data)
    table2['R_E'] = np.log10(erass1['R500_H'].data)-np.log10(erass1['R500'].data)
    table2['Mgas'] = np.log10(erass1['MGAS500'].data*1e11)
    table2['Mgas_e'] = np.log10(erass1['MGAS500'].data)-np.log10(erass1['MGAS500_L'].data)
    table2['Mgas_E'] = np.log10(erass1['MGAS500_H'].data)-np.log10(erass1['MGAS500'].data)
    table2['L'] = np.log10(erass1['Lbol500'].data)+42 # 10**42 erg s-1
    table2['L_e'] = np.log10(erass1['Lbol500'].data)-np.log10(erass1['Lbol500_L'].data)
    table2['L_E'] = np.log10(erass1['Lbol500_H'].data)-np.log10(erass1['Lbol500'].data)
    table2['T'] = np.log10(erass1['KT'].data)
    table2['T_e'] = table2['T'] - np.log10(erass1['KT_L'].data)
    table2['T_E'] = np.log10(erass1['KT_H'].data) - table2['T']
    table2['z'] = erass1['BEST_Z']
    # see https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.210K/abstract
    # see https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/688/A210#/browse
    # and https://cdsarc.cds.unistra.fr/ftp/J/A+A/688/A210/ReadMe
    erass1_optical_properties = Table.read('./Data/J_A+A_688_A210/J_A+A_688_A210_tablee1.dat.gz.fits')
    #table2['VDIS'] = erass1_optical_properties['vdispBoot'].data
    table2['VDIS'] = np.log10(erass1_optical_properties['vdispBoot'].data)
    table2['VDIS'].unit = "Log(VDIS / km/s)"
    #table2['VDIS_e'] = table2['VDIS_E'] = erass1_optical_properties['e_vdispBoot'].data
    table2['VDIS_e'] = table2['VDIS_E'] = np.log10(1+erass1_optical_properties['e_vdispBoot'].data/erass1_optical_properties['vdispBoot'].data)
    table2['Nmemb'] = erass1_optical_properties['Nmemb']
    table2['label'] = 'eRASS1'
    table = vstack([table1, table2])
    bestfit_rho_z = cosmo_bestfit.critical_density(table['z'].data)/fiducial_cosmolgy.critical_density(table['z'].data)
    table['R_corr'] = table['R'].data + np.log10(bestfit_rho_z)/3
    table['R_corr_e'] = table['R_e'].data #+ np.log10(bestfit_rho_z)/3
    table['R_corr_E'] = table['R_E'].data #+ np.log10(bestfit_rho_z)/3
    table['R_mass'] = table['R'].data
    table['R_mass'].unit =  "Log10(R / kpc)"
    table['R_mass_e'] = table['R_e'].data
    table['R_mass_E'] = table['R_E'].data

    table['R'].unit = "Log10(R/kpc)"
    table['Mgas'].unit = "Log10(M/M_sun)"
    table['L'].unit = "Log10(L/erg s-1)"
    table['T'].unit = "Log10(T/keV)"
    table['z'].unit = "z"
    table['VDIS'].unit = "Log(VDIS/km/s)"
    # best-fit values
    table['OmegaB'] = 0.29
    table['Hubble'] = Planck18.H0
    table['z_E'] = table['z_e'] = z_error
    return table


def simulated_data(fac1=1/3, fac2=1/20):
    table1 = Table([], names=[])
    mr_simulation = read_15mr_simulation()
    table1['id'] = np.arange(len(mr_simulation))
    table1['R'] = np.log10(mr_simulation['R500_kpc'].data)
    table1['R'].unit =  "Log10(R / kpc)"
    table1['R_E'] = fac1*(np.log10(mr_simulation['R200_kpc'].data) - np.log10(mr_simulation['R500_kpc'].data))
    table1['R_e'] = fac2*(np.log10(mr_simulation['R500_kpc'].data) - np.log10(mr_simulation['R2500_kpc'].data))
    table1['Mgas'] = np.log10(mr_simulation['Mgas500'].data)+12
    table1['Mgas'].unit = "Log10(M / M_sun)"
    table1['Mgas_E'] = fac1*(np.log10(mr_simulation['Mgas200'].data) - np.log10(mr_simulation['Mgas500'].data))
    table1['Mgas_e'] = fac2*(np.log10(mr_simulation['Mgas500'].data) - np.log10(mr_simulation['Mgas2500'].data))
    table1['L'] = np.log10(mr_simulation['Lbol500'].data)+42
    table1['L'].unit = "Log10(L / erg s-1)"
    table1['L_E'] = fac1*(np.log10(mr_simulation['Lbol200'].data) - np.log10(mr_simulation['Lbol500'].data))
    table1['L_e'] = fac2*(np.log10(mr_simulation['Lbol500'].data) - np.log10(mr_simulation['Lbol2500'].data))
    table1['T'] = np.log10(mr_simulation['T500'].data)
    table1['T'].unit = "Log10(T / keV)"
    table1['T_E'] = np.abs(fac1*(np.log10(mr_simulation['T200'].data) - np.log10(mr_simulation['T500'].data)))
    table1['T_e'] = np.abs(fac2*(np.log10(mr_simulation['T500'].data) - np.log10(mr_simulation['T2500'].data)))
    table1['z'] = mr_simulation['z'].data
    table1['Msta'] = np.log10(mr_simulation['Msta500'].data)+12
    table1['Msta_E'] = fac1*(np.log10(mr_simulation['Msta200'].data) - np.log10(mr_simulation['Msta500'].data))
    table1['Msta_e'] = fac2*(np.log10(mr_simulation['Msta500'].data) - np.log10(mr_simulation['Msta2500'].data))
    table1['M_TOL'] = np.log10(mr_simulation['M_R500'])+12 #total mass at R=500C
    table1['M_TOL_E'] = fac1*(np.log10(mr_simulation['M_R200']) - np.log10(mr_simulation['M_R500']))
    table1['M_TOL_e'] = fac2*(np.log10(mr_simulation['M_R500']) - np.log10(mr_simulation['M_R2500']))
    table1['M_RVIR'] = np.log10(mr_simulation['M_RVIR'])+12 # total virial mass
    table1['VDIS'] = np.log10(mr_simulation['SIGMAV'].data)
    table1['VDIS_E'] = table1['VDIS_e'] = np.log10(1+0.3)
    table1['VDIS'].unit = "Log(VDIS / km/s)"
    table1['rho_z'] = mr_simulation['rho_z']
    table1['label'] = mr_simulation['label']
    table1['Omega'] =  mr_simulation['Omega'] 
    table1['OmegaB'] =  mr_simulation['OmegaB']
    table1['Sigm8'] =  mr_simulation['Sigm8']
    table1['Hubble'] =  mr_simulation['Hubble']
   
    table2 = Table([], names=[])
    hr_simulation = pd.read_csv('./Data/Box2b_hr/cluster.csv')
    hr_simulation = hr_simulation[hr_simulation['z'] <= 1.0]
    # note that here we show the large box (Box2b)
    h0 = table1[table1['label']==['C8']]['Hubble'][0]
    table2['id'] = len(table1)+np.arange(len(hr_simulation))
    table2['R'] = np.log10(hr_simulation['r500c[kpc/h]']*h0)
    table2['R'].unit =  "Log10(R / kpc)"
    table2['Mgas'] = np.log10(hr_simulation['m500c[Msol/h]']*h0*hr_simulation['gas_frac'])
    table2['Mgas'].unit = "Log10(M / M_sun)"
    table2['L'] = np.log10(hr_simulation['Lx[1e44erg/s]'])+44
    table2['L'].unit =  "Log10(L / erg s-1)"
    table2['T'] = np.log10(hr_simulation['T[kev]'])
    table2['T'].unit = "Log10(T / keV)"
    table2['z'] = hr_simulation['z']
    table2['Msta'] = np.log10(hr_simulation['m500c[Msol/h]']*h0*hr_simulation['star_frac'])
    table2['M_TOL'] = np.log10(hr_simulation['m500c[Msol/h]']*h0)
    table2['M_RVIR'] = -1
    table2['label'] = 'HR'
    table2['Omega'] =  table1[table1['label']==['C8']]['Omega'][0]
    table2['OmegaB'] =  table1[table1['label']==['C8']]['OmegaB'][0]
    table2['Sigm8'] =  table1[table1['label']==['C8']]['Sigm8'][0]
    table2['Hubble'] =  table1[table1['label']==['C8']]['Hubble'][0]
    hr_simulated_cosmology = cosmology(table2['Hubble'][0], table2['OmegaB'][0])
    hr_simulated_rho_z = hr_simulated_cosmology.critical_density(table2['z'])
    fiducial_rho_z = fiducial_cosmolgy.critical_density(table2['z'])
    table2['rho_z'] =  hr_simulated_rho_z/fiducial_rho_z

    table = vstack([table1, table2])
    rho_crit = table['rho_z']*fiducial_cosmolgy.critical_density(table['z'].data).to(u.Msun/u.kpc**3)
    R500_mass = 3/(4*np.pi)*((10**table['Mgas'].data)*u.Msun/rho_crit)**(1/3)
    table['R_mass'] = np.log10(R500_mass.data)
    table['R_mass'].unit =  "Log10(R / kpc)"
    R_mass_e = 3/(4*np.pi)*((10**(table['Mgas'].data-table['Mgas_e'].data))*u.Msun/rho_crit)**(1/3)
    table['R_mass_e'] = table['R_mass'].data - np.log10(R_mass_e.data)
    R_mass_E = 3/(4*np.pi)*((10**(table['Mgas'].data+table['Mgas_E'].data))*u.Msun/rho_crit)**(1/3)
    table['R_mass_E'] = np.log10(R_mass_E.data) - table['R_mass'].data
    table['R_corr'] = table['R'].data + np.log10(table['rho_z'].data)/3
    table['R_corr'].unit =  "Log10(R / kpc)"
    table['R_corr_e'] = table['R_e'].data #+ np.log10(table['rho_z'].data)/3
    table['R_corr_E'] = table['R_E'].data #+ np.log10(table['rho_z'].data)/3
    table['z'].unit = "z"
    table['z_E'] = table['z_e'] = z_error
    return table

# http://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Downloads/Singh_2020_data.tar.gz
#For each cosmology, there is folder named: mr_Omega_OmegaB_Sigm8_Hubble.
#In each folder, there are six files named snap_xxx.pkl, where xxx = 014, 013, 012, 011, 010 and 009
#corresponding to z = 0, 0.14, 0.29, 0.47, 0.67 and 0.9 respectively.
#Each of the halo property has six values corresponding to radii 
#RVIR, R200M, R500M, R200C, R500C and R2500C saved in the same order
#(except 'SIGMAV' which has only one value per halo).
# reading single mr simulation data and save as .csv
def read_redshift_snapshot(snap_z, z):
    # 'SIGMAV' is not inclued
    keys_list = ['RADII', 'MGAS', 'TGAS', 'MASSES', 'MSTR', 'LGAS']
    save_data = {}
    for key in keys_list:
        temp_data = {'RVIR':snap_z[key][:,0], 
                    'R200M':snap_z[key][:,1], 
                    'R500M':snap_z[key][:,2],
                    'R200C':snap_z[key][:,3], 
                    'R500C':snap_z[key][:,4], 
                    'R2500C':snap_z[key][:,5]}
        for ky in temp_data.keys():
            new_ky = key + '_' + ky
            save_data[new_ky] = temp_data[ky]
    save_data['SIGMAV'] = snap_z['SIGMAV']
    save_data['z'] = z*np.ones_like(snap_z['SIGMAV'])
    save_data = Table(save_data)
    return save_data

def read_mr_cosmology_simulation(sim_name):
    path = './Data/Singh_2020_data/'
    snap_z_000 = pd.read_pickle(path+f'{sim_name}/snap_014.pkl')
    snap_z_014 = pd.read_pickle(path+f'{sim_name}/snap_013.pkl')
    snap_z_029 = pd.read_pickle(path+f'{sim_name}/snap_012.pkl')
    snap_z_047 = pd.read_pickle(path+f'{sim_name}/snap_011.pkl')
    snap_z_067 = pd.read_pickle(path+f'{sim_name}/snap_010.pkl')
    snap_z_090 = pd.read_pickle(path+f'{sim_name}/snap_009.pkl')
    Table_list = [  read_redshift_snapshot(snap_z_000, z=0.00),
                    read_redshift_snapshot(snap_z_014, z=0.14),
                    read_redshift_snapshot(snap_z_029, z=0.20),
                    read_redshift_snapshot(snap_z_047, z=0.47),
                    read_redshift_snapshot(snap_z_067, z=0.67),
                    read_redshift_snapshot(snap_z_090, z=0.90)]
    simulated_data = vstack(Table_list)
    simulated_data['simulation_name'] = sim_name
    return simulated_data

# Turn the simluation units to observation (eFEDS)
def units_transfer_mr_simulated_data(mr_data):
    h0 = mr_data['Hubble']
    # RADII_R500C
    # ['kpc/h0/(1+z)'] -> ['kpc/h0']
    mr_data['R500_kpc/h0'] = mr_data['RADII_R500C']*(1+mr_data['z'])
    mr_data['R500_kpc'] = mr_data['R500_kpc/h0']*h0
    mr_data['R200_kpc/h0'] = mr_data['RADII_R200C']*(1+mr_data['z'])
    mr_data['R200_kpc'] = mr_data['R200_kpc/h0']*h0
    mr_data['R2500_kpc/h0'] = mr_data['RADII_R2500C']*(1+mr_data['z'])
    mr_data['R2500_kpc'] = mr_data['R2500_kpc/h0']*h0
    # MGAS_R500C
    # 1e10/h0 -> 1e+12solMass
    mr_data['Mgas500'] = mr_data['MGAS_R500C']*h0/1e2
    mr_data['Mgas200'] = mr_data['MGAS_R200C']*h0/1e2
    mr_data['Mgas2500'] = mr_data['MGAS_R2500C']*h0/1e2
    mr_data['Msta500'] = mr_data['MSTR_R500C']*h0/1e2
    mr_data['Msta200'] = mr_data['MSTR_R200C']*h0/1e2
    mr_data['Msta2500'] = mr_data['MSTR_R2500C']*h0/1e2
    mr_data['M_R500'] = (mr_data['MSTR_R500C']+mr_data['MGAS_R500C']+mr_data['MASSES_R500C'])*h0/1e2 #total mass at R=500C
    mr_data['M_R200'] = (mr_data['MSTR_R200C']+mr_data['MGAS_R200C']+mr_data['MASSES_R200C'])*h0/1e2 #total mass at R=500C
    mr_data['M_R2500'] = (mr_data['MSTR_R2500C']+mr_data['MGAS_R2500C']+mr_data['MASSES_R2500C'])*h0/1e2 #total mass at R=500C
    mr_data['M_RVIR'] = (mr_data['MSTR_RVIR']+mr_data['MGAS_RVIR']+mr_data['MASSES_RVIR'])*h0/1e2 # total virial mass
    # LGAS_R500C
    # 1e44 erg -> 1e+35W (10^42^erg/s)
    # 1 W = 1.0E-7 erg/s
    mr_data['Lbol500'] = mr_data['LGAS_R500C']*1e2
    mr_data['Lbol200'] = mr_data['LGAS_R200C']*1e2
    mr_data['Lbol2500'] = mr_data['LGAS_R2500C']*1e2
    # TGAS_R500C
    # Mass weighted gas temperatures in keV -> keV
    mr_data['T500'] = mr_data['TGAS_R500C']
    mr_data['T200'] = mr_data['TGAS_R200C']
    mr_data['T2500'] = mr_data['TGAS_R2500C']
    return mr_data

# read 15 mid-resolution simulation
def read_15mr_simulation():
    # use all 15 simulation in Singh et al. 2020
    simulation_maps = { 'C1':'mr_0.153_0.0408_0.614_0.666', 'C2':'mr_0.189_0.0455_0.697_0.703', 'C3':'mr_0.200_0.0415_0.850_0.730', 
                        'C4':'mr_0.204_0.0437_0.739_0.689', 'C5':'mr_0.222_0.0421_0.793_0.676', 'C6':'mr_0.232_0.0413_0.687_0.670',
                        'C7':'mr_0.268_0.0449_0.721_0.699', 'C8':'mr_0.272_0.0456_0.809_0.704', 'C9':'mr_0.301_0.0460_0.824_0.707',
                        'C10':'mr_0.304_0.0504_0.886_0.740','C11':'mr_0.342_0.0462_0.834_0.708','C12':'mr_0.363_0.0490_0.884_0.729',
                        'C13':'mr_0.400_0.0485_0.650_0.675','C14':'mr_0.406_0.0466_0.867_0.712','C15':'mr_0.428_0.0492_0.830_0.732',
                        'C8a1':'mr_0.272_0.0456_0.809_0.704_a1', 'C8a2':'mr_0.272_0.0456_0.809_0.704_a2'}
    simulation_list = []
    for label, name in simulation_maps.items():
        print(f'simulation name: {name}, label: {label}')
        mr_simulation = read_mr_cosmology_simulation(name)
        temp = name.rsplit('/')[-1].split('_')
        # get simulation cosmology parameters
        mr_simulation['label'] = label
        '''
        mr_simulation['Omega'] = float(temp[-4])
        mr_simulation['OmegaB'] = float(temp[-3])
        mr_simulation['Sigm8'] = float(temp[-2])
        mr_simulation['Hubble'] = float(temp[-1])
        '''
        mr_simulation['Omega'] = float(temp[1])
        mr_simulation['OmegaB'] = float(temp[2])
        mr_simulation['Sigm8'] = float(temp[3])
        mr_simulation['Hubble'] = float(temp[4])
        simulated_cosmology = cosmology(mr_simulation['Hubble'][0], mr_simulation['OmegaB'][0])
        simulated_rho_z = simulated_cosmology.critical_density(mr_simulation['z'].data)
        fiducial_rho_z = fiducial_cosmolgy.critical_density(mr_simulation['z'].data)
        mr_simulation['rho_z'] = simulated_rho_z/fiducial_rho_z
        mr_simulation = units_transfer_mr_simulated_data(mr_simulation)
        simulation_list.append(mr_simulation)
    simulation_data = vstack(simulation_list)
    simulation_data.write('./Data/mr_simulation_data.fits', format="fits", overwrite=True)
    return simulation_data


# 设置极限光度 [erg/s]
L_lim_fixed = 1e38
sigma_logL = 0.5  # 对数光度误差
D0 = cosmo.luminosity_distance(z=0.001).to('cm').value
flux_limit = L_lim_fixed/(4*np.pi*D0**2)
# 每个 z 下的极限光度（用于与目标 L 比较）
def L_lim_z(z):
    D_L = cosmo.luminosity_distance(z).to('cm').value  # 输出单位：cm
    L_lim = 4 * np.pi * D_L**2 * flux_limit
    return L_lim
