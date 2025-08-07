import model
import pandas as pd
import numpy as np
import time
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", palette="deep")


def get_impurity_importance(model_name, column_rangs):
    feature_names =  list(column_rangs.keys())
    randomforest = model.RandomForest()
    randomforest.config['model_name'] = model_name
    rf = randomforest.get_model()
    importances = rf.feature_importances_
    importances_std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    df_importance = pd.DataFrame({'Feature':feature_names, 'impurity_importance': importances, 'impurity_importance_std':importances_std})
    return df_importance



def get_permutation_importance(model_name, valid_data, column_rangs):
    feature_names =  list(column_rangs.keys())
    randomforest = model.RandomForest()
    randomforest.config['model_name'] = model_name
    test_data = model.transform(valid_data, column_rangs)
    col_names = list(column_rangs.keys())
    col_names = [x + '_norm' for x in col_names]
    x_test = test_data[col_names].values
    y_test = test_data['label'].values
    forest = randomforest.get_model()
    start_time = time.time()
    result = permutation_importance(
        forest, x_test, y_test, n_repeats=10, random_state=42, n_jobs=8)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    importance = pd.DataFrame({'Feature': feature_names, 'permutation_importance': result.importances_mean, 'permutation_importance_std':result.importances_std})

    return importance


def plot(fname, importance, show_one=True):
    plt.figure(figsize=(8, 5), dpi=160)
    fsize = 12
    bar_width = 0.4  
    x = np.arange(len(importance['Feature']))
    if show_one:
        plt.bar(x, importance['impurity_importance'], yerr=importance['impurity_importance_std'], width=bar_width, label='impurity importance', color='skyblue')
    else:
        plt.bar(x-bar_width/2, importance['impurity_importance'], yerr=importance['impurity_importance_std'], width=bar_width, label='impurity importance', color='skyblue')
        plt.bar(x+bar_width/2, importance['permutation_importance'], yerr=importance['permutation_importance_std'],width=bar_width, label='permutation importance', color='orange')
    plt.xticks(x, importance['Feature'], fontsize=fsize)
    plt.yticks(fontsize=fsize)            
    plt.xlabel("Feature", fontsize=fsize)
    plt.ylabel("Importance score", fontsize=fsize)
    plt.legend(fontsize=fsize)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    model_name = 'RF_R_corr'
    impur_importance = get_impurity_importance(model_name, column_rangs=column_rangs)
    permu_importances = get_permutation_importance(model_name, valid_data, column_rangs=column_rangs)
    importances = impur_importance.merge(permu_importances, on='Feature', how='left')
    importances.to_csv(f'./results/{model_name}_feature_importance.csv')
    importances = pd.read_csv(f'./results/{model_name}_feature_importance.csv')
    plot(importances)