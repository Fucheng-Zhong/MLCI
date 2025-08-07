import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from astropy.table import Table,vstack
import numpy as np

simulation_error = {'R':0.02, 'VDIS':0.3, 'Mgas':0.08, 'L':0.1, 'T':0.1, 'z':0.05}


def calculate_correlated_coefficient(data, vars_name):
    '''
    The correlation of whole dataset
    '''
    means = []
    for key in vars_name:
        temp_mean = data[key]
        temp_mean = temp_mean.reshape(-1,1)
        means.append(temp_mean)
    means_values = np.concatenate(means, axis=-1)
    correlated_coefficient = np.corrcoef(means_values, rowvar=False)
    return correlated_coefficient


def read_mu_error(data, vars_name):
    '''
    Reading the measured values as expectation and errors as sigma for each cluster
    '''
    #=====
    expectations, error_plus, error_minus = [], [], []
    for name in vars_name:
        mu = data[name].data
        expectations.append(mu[:,None])
        err_plus  = data[name+'_E'].data 
        err_minus = data[name+'_e'].data 
        error_plus.append(err_plus[:,None])
        error_minus.append(err_minus[:,None])
    expectations = np.concatenate(expectations, axis=-1)
    error_plus = np.concatenate(error_plus, axis=-1)
    error_minus = np.concatenate(error_minus, axis=-1)
    return expectations, error_plus, error_minus
    

def boosting(data, vars_name, num, sampler, max_num=None):
    boost_info, boost_samples = [], []
    label_names = data['label'].data
    label_list = np.unique(label_names)
    for label in label_list:
        subset = data[data['label']==label]
        print('boosting subset', label)
        info, samples  = subset_boosting(subset, vars_name, num, sampler, max_num)
        boost_info.append(info)
        boost_samples.append(samples)

    boost_info = vstack(boost_info)
    boost_samples = vstack(boost_samples)
    return boost_info, boost_samples
    


def simulation_boosting(data, number, column_rangs, balance=False, clip=3, ramdom_seed=42):
    np.random.seed(ramdom_seed)
    boosting_samples = []
    label_names = data['label'].data
    label_list = np.unique(label_names)
    noise = np.array([simulation_error[name] for name in column_rangs.keys()])
    for label in label_list:
        subset = data[data['label']==label]
        vars_name = column_rangs.keys()
        corr_matrix = calculate_correlated_coefficient(subset, vars_name)
        Omega = np.outer(noise, noise)*corr_matrix
        print(label, len(subset))
        # ==== radom select
        if not balance:
            num = np.min([len(subset), number])
            row_indices = np.random.choice(len(subset), size=num, replace=False)
        else:
            num = number
            row_indices = np.random.choice(len(subset), size=num, replace=True)
        boosted_subset = subset[row_indices]
        # === error generating
        error = multi_normal(np.zeros(len(column_rangs)), Omega, size=num, seed=None)
        error = np.clip(error, -clip*noise, clip*noise)
        # ==== add error for each column
        for col_name, i in zip(column_rangs.keys(), range(len(column_rangs))):
            boosted_subset[col_name] = boosted_subset[col_name] + error[:,i]
            #boosted_subset[col_name] = boosted_subset[col_name]*(1+error[:,i])
        boosting_samples.append(boosted_subset)
    boosting_samples = vstack(boosting_samples)
    row_indices = np.arange(len(boosting_samples))
    np.random.shuffle(row_indices)
    boosting_samples = boosting_samples[row_indices]
    return boosting_samples



def sample_mvn_skewnorm(mu, Omega, err_plus, err_minus, size, seed=None):
    # Skewness 参数估计
    skewness_ratio = (err_plus-err_minus)/(err_plus + err_minus + 1e-8)
    alpha = 10.0 * skewness_ratio  #
    rng = np.random.default_rng(seed)
    d = len(mu)
    z = rng.multivariate_normal(mean=np.zeros(d), cov=Omega, size=size)
    u = rng.normal(size=size)
    delta = alpha / np.sqrt(1 + np.dot(alpha, alpha))
    mask = (u < np.dot(z, delta))
    z[~mask] *= -1
    return mu + z

def sample_correlated_split_normal(mu, corr_matrix, err_plus, err_minus, size, seed=None):
    d = mu.shape[0]
    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)  # [d, d]
    # Sample from standard normal
    z = np.random.randn(size, d)  # [N, d]
    # Induce correlation: z_corr ~ N(0, cov)
    z_corr = z @ L.T  # [N, d]
    # Apply asymmetric scaling (split-normal transform)
    mu = mu[None, :]                # [1, d]
    err_plus = err_plus[None, :]  # [1, d]
    err_minus = err_minus[None, :]    # [1, d]
    # Sharp threshold at z_corr = 0
    sigma = np.where(z_corr<0, err_minus, err_plus)
    x = mu + z_corr * sigma
    return x

def multi_normal(mu, Omega, size, seed=None):
    muti_norm = multivariate_normal(mu, Omega, allow_singular=True, seed=seed)
    x = muti_norm.rvs(size)
    return x


def subset_boosting(data, vars_name, num, sampler, max_num=None):
    np.random.seed(42)
    boosted_samples = []
    samples_q16, samples_q50, samples_q84, samples_std, samples_ave = [], [], [], [], []
    if max_num is not None and len(data) > int(max_num):
        row_indices = np.random.choice(len(data), size=int(max_num), replace=False)
        data = data[row_indices]
    print('subset num:', len(data))
    expectations, error_plus, error_minus = read_mu_error(data, vars_name)
    corr_matrix = calculate_correlated_coefficient(data, vars_name)
    for mu, err_plus, err_minus in zip(expectations, error_plus, error_minus):
        sigma = (err_plus+err_minus)/2
        Omega = np.outer(sigma, sigma)*corr_matrix
        #print(sigma, Omega)
        if sampler == 'skewnorm':
            samples = sample_mvn_skewnorm(mu, Omega, err_plus, err_minus, num)
        elif sampler == 'splitnorm':
            samples = sample_correlated_split_normal(mu, corr_matrix, err_plus, err_minus, num)
        elif sampler == 'multinorm':
            samples = multi_normal(mu, Omega, num)
        boosted_samples.append(samples)
        q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
        std = np.std(samples, axis=0, keepdims=True)
        ave = np.mean(samples, axis=0, keepdims=True)

        samples_q16.append(q16[None,:])
        samples_q50.append(q50[None,:])
        samples_q84.append(q84[None,:])
        samples_std.append(std)
        samples_ave.append(ave)

    samples_q16 = np.concatenate(samples_q16, axis=0)
    samples_q50 = np.concatenate(samples_q50, axis=0)
    samples_q84 = np.concatenate(samples_q84, axis=0)
    samples_std = np.concatenate(samples_std, axis=0)
    samples_ave = np.concatenate(samples_ave, axis=0)

    boosted_info = {'samples': boosted_samples, 'q16':samples_q16, 'q50':samples_q50,'q84':samples_q84, 'std':samples_std, 'ave':samples_ave,
                    'mu':expectations, 'err_plus':error_plus, 'err_minus':error_minus}
    boosted_info = Table(boosted_info)
    new_samples = np.concatenate(boosted_samples, axis=0)
    new_samples = Table(data=new_samples, names=vars_name)
    new_samples['label'] = data['label'][0]
    new_samples['sampler'] = sampler
    return boosted_info, new_samples


def show_boosting_results(boosted_info, vars_name, fname=''):
    mu =  boosted_info['mu']
    err_plus = boosted_info['err_plus']
    err_min = boosted_info['err_minus']
    err_ave = (err_plus+err_min)/2
    q16, q50, q84 = boosted_info['q16'],  boosted_info['q50'],  boosted_info['q84']
    ave, std = boosted_info['ave'], boosted_info['std']
    dim = mu.shape[-1]
    fig, axs = plt.subplots(1, dim, figsize=(4*dim, 4), dpi=160)
    f_size = 12
    for i in range(dim):
        ax = axs[i]
        ax.scatter(mu[:, i], (mu[:, i]-ave[:, i])/mu[:, i], s=2, alpha=0.5, label= f'{vars_name[i]}: mu')
        ax.scatter(err_ave[:, i], (err_ave[:, i]-std[:, i])/err_ave[:, i], s=2, alpha=0.5, label= f'{vars_name[i]}: err_ave')
        ax.scatter(err_min[:, i], (err_min[:, i]-(q50[:, i]-q16[:, i]))/err_min[:, i], s=2, alpha=0.5, label= f'{vars_name[i]}: err_minus')
        ax.scatter(err_plus[:, i], (err_plus[:, i]-(q84[:, i]-q50[:, i]))/err_plus[:, i], s=2, alpha=0.5, label= f'{vars_name[i]}: err_plus')
        ax.set_xlabel('original', fontsize=f_size)
        ax.set_ylabel('boosted', fontsize=f_size)
        legend = ax.legend()
    plt.title("boosting")
    plt.grid(True)
    if fname == '':
        fname = './figures/boosting_results.pdf'
    plt.savefig(fname)
    plt.close()


def show_error_dist(boosted_info, vars_name, ave_err=False, fname=''):
    mu =  boosted_info['mu']
    err_plus = boosted_info['err_plus']
    err_min = boosted_info['err_minus']
    err_ave = (err_plus+err_min)/2
    q16, q50, q84 = boosted_info['q16'],  boosted_info['q50'],  boosted_info['q84']
    ave, std = boosted_info['ave'], boosted_info['std']
    dim = mu.shape[-1]
    fig, axs = plt.subplots(1, dim, figsize=(4*dim, 4), dpi=160)
    for i in range(dim):
        ax = axs[i]
        if ave_err:
            ax.hist((err_plus[:, i]+err_min[:, i])/2, bins=50, density=False, alpha=0.8, label= f'{vars_name[i]}: err_ave')
        else:
            ax.hist(err_plus[:, i], bins=50, density=False, alpha=0.8, label= f'{vars_name[i]}: err_plus')
            ax.hist(err_min[:, i], bins=50, density=False, alpha=0.8, label= f'{vars_name[i]}: err_minus')
        ax.set_xlabel("error")
        ax.set_ylabel("count")
        legend = ax.legend()
    plt.grid(True)
    if fname == '':
        fname = './figures/boosting_error_dis.pdf'
    plt.savefig(fname)
    plt.close()