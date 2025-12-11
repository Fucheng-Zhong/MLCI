import yaml
import os
from copy import deepcopy

benchmark = {'mode':'RF', 'agn_feedback':0.0, 'proportional':True, 'noise_level':[1/3, 1/20],
            'leaf_size':1e4, 'max_num':1e4, 'sample_num':100, 'max_depth':10, 'exclude_outlier':False, 'description':'Fiducial',
            'delta_prop':1,
            'simulations': [f'C{i+3}' for i in range(13)],
            'boost_real':True,
            'cols':{'R_corr':[2.5, 3.5], 'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]},
            'p_threshold':0.0
            }

test_set = {}
# observed level error, ordinary agn feedback, different simulations
test_set['RFtest1'] = deepcopy(benchmark)
test_set['RFtest1']['description'] = 'Fiducial'

test_set['RFtest2'] = deepcopy(benchmark)
test_set['RFtest2']['simulations'] = [f'C{i+2}' for i in range(14)]
test_set['RFtest2']['description'] = '+C2'

test_set['RFtest3'] = deepcopy(benchmark)
test_set['RFtest3']['simulations'] = ['C2','C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
test_set['RFtest3']['description'] = '+C2-C4'


test_set['RFtest4'] = deepcopy(benchmark)
test_set['RFtest4']['simulations'] = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C11', 'C13', 'C14']
test_set['RFtest4']['description'] = '-C3-C10-C12-C15'

test_set['RFtest5'] = deepcopy(benchmark)
test_set['RFtest5']['simulations'] = ['C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
test_set['RFtest5']['description'] = '-C3-C4-C5-C6'


test_set['RFtest6'] = deepcopy(benchmark)
test_set['RFtest6']['agn_feedback'] = '+delta'
test_set['RFtest6']['description'] = 'feedback +delta'


test_set['RFtest7'] = deepcopy(benchmark)
test_set['RFtest7']['agn_feedback'] = '-delta'
test_set['RFtest7']['description'] = 'feedback -delta'


test_set['RFtest8'] = deepcopy(benchmark)
test_set['RFtest8']['agn_feedback'] = 0.1
test_set['RFtest8']['description'] = 'feedback +0.1'


test_set['RFtest9'] = deepcopy(benchmark)
test_set['RFtest9']['agn_feedback'] = -0.1
test_set['RFtest9']['description'] = 'feedback -0.1'


test_set['RFtest10'] = deepcopy(benchmark)
test_set['RFtest10']['agn_feedback'] = 0.2
test_set['RFtest10']['description'] = 'feedback +0.2'


test_set['RFtest11'] = deepcopy(benchmark)
test_set['RFtest11']['agn_feedback'] = -0.2
test_set['RFtest11']['description'] = 'feedback -0.2'

test_set['RFtest12'] = deepcopy(benchmark)
test_set['RFtest12']['agn_feedback'] = 0.3
test_set['RFtest12']['description'] = 'feedback +0.3'

test_set['RFtest13'] = deepcopy(benchmark)
test_set['RFtest13']['agn_feedback'] = -0.3
test_set['RFtest13']['description'] = 'feedback -0.3'

test_set['RFtest14'] = deepcopy(benchmark)
test_set['RFtest14']['noise_level'] = [0.1, 0.1]
test_set['RFtest14']['description'] = 'noise level 0.1'

test_set['RFtest15'] = deepcopy(benchmark)
test_set['RFtest15']['noise_level'] = [0.01, 0.01]
test_set['RFtest15']['description'] = 'noise level 0.01'

#=======
test_set['RFtest16'] = deepcopy(benchmark)
test_set['RFtest16']['simulations'] = [f'C{i+1}' for i in range(15)]
test_set['RFtest16']['description'] = 'all simulations'

test_set['RFtest17'] = deepcopy(benchmark)
test_set['RFtest17']['leaf_size'] = 1e3
test_set['RFtest17']['description'] = 'leaf size 1000'

test_set['RFtest18'] = deepcopy(benchmark)
test_set['RFtest18']['leaf_size'] = 1e3
test_set['RFtest18']['noise_level'] = [0.01, 0.01]
test_set['RFtest18']['description'] = 'noise 0.01+leaf 1000'

test_set['RFtest19'] = deepcopy(benchmark)
test_set['RFtest19']['agn_feedback'] = '+delta'
test_set['RFtest19']['delta_prop'] = 2
test_set['RFtest19']['description'] = 'feedback +2delta'


test_set['RFtest20'] = deepcopy(benchmark)
test_set['RFtest20']['agn_feedback'] = '-delta'
test_set['RFtest20']['delta_prop'] = 2
test_set['RFtest20']['description'] = 'feedback -2delta'

test_set['RFtest21'] = deepcopy(benchmark)
test_set['RFtest21']['mode'] = 'NB'

test_set['RFtest22'] = deepcopy(benchmark)
test_set['RFtest22']['simulations'] = ['C2','C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
test_set['RFtest22']['description'] = '+C2-C12-C13-C14-C15'

test_set['RFtest23'] = deepcopy(benchmark)
test_set['RFtest23']['simulations'] = [f'C{i+2}' for i in range(14)]
test_set['RFtest23']['description'] = '+C2_4k'
test_set['RFtest23']['max_num'] = 4e3

test_set['RFtest24'] = deepcopy(benchmark)
test_set['RFtest24']['description'] = 'z<0.5'
test_set['RFtest24']['cols']['z'] = [0.0, 0.5]

test_set['RFtest25'] = deepcopy(benchmark)
test_set['RFtest25']['simulations'] = [f'C{i+2}' for i in range(14)]
test_set['RFtest25']['description'] = '+C2_4k_z<0.5'
test_set['RFtest25']['max_num'] = 4e3
test_set['RFtest25']['cols']['z'] = [0.0, 0.5]

test_set['RFtest26'] = deepcopy(benchmark)
test_set['RFtest26']['simulations'] = ['C2','C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
test_set['RFtest26']['description'] = '+C2-C13-C14-C15'


test_set['RFtest27'] = deepcopy(benchmark)
test_set['RFtest27']['description'] = '0.2<z<0.9'
test_set['RFtest27']['cols']['z'] = [0.2, 0.9]

test_set['RFtest28'] = deepcopy(benchmark)
test_set['RFtest28']['description'] = '0.2<z<0.8'
test_set['RFtest28']['cols']['z'] = [0.2, 0.8]

test_set['RFtest29'] = deepcopy(benchmark)
test_set['RFtest29']['description'] = '0.2<z<0.7'
test_set['RFtest29']['cols']['z'] = [0.2, 0.7]

test_set['RFtest30'] = deepcopy(benchmark)
test_set['RFtest30']['description'] = '0.3<z<0.5'
test_set['RFtest30']['cols']['z'] = [0.3, 0.5]

test_set['RFtest31'] = deepcopy(benchmark)
test_set['RFtest31']['simulations'] = ['C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
test_set['RFtest31']['description'] = '-C13-C14-C15'

test_set['RFtest32'] = deepcopy(benchmark)
test_set['RFtest32']['simulations'] = [f'C{i+3}' for i in range(13)]
test_set['RFtest32']['description'] = '4k'
test_set['RFtest32']['max_num'] = 4e3

test_set['RFtest33'] = deepcopy(benchmark)
test_set['RFtest33']['simulations'] = ['C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
test_set['RFtest33']['description'] = '-C12-C13-C14-C15'


test_set['RFtest34'] = deepcopy(benchmark)
test_set['RFtest34']['simulations'] = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
test_set['RFtest34']['description'] = '-C3'


test_set['RFtest35'] = deepcopy(benchmark)
test_set['RFtest35']['description'] = '0.1<z<0.35'
test_set['RFtest35']['cols']['z'] = [0.1, 0.35]

test_set['RFtest36'] = deepcopy(benchmark)
test_set['RFtest36']['description'] = '0.35<z<0.80'
test_set['RFtest36']['cols']['z'] = [0.35, 0.80]

test_set['RFtest37'] = deepcopy(benchmark)
test_set['RFtest37']['description'] = 'Lx>43.5'
test_set['RFtest37']['cols']['L']  = [43.5, 46.0]

test_set['RFtest38'] = deepcopy(benchmark)
test_set['RFtest38']['description'] = 'Lx=44.5-46.0'
test_set['RFtest38']['cols']['L']  = [44.5, 46.0]

test_set['RFtest39'] = deepcopy(benchmark)
test_set['RFtest39']['description'] = 'Lx=43.0-44.5'
test_set['RFtest39']['cols']['L'] = [43.0, 44.5]

test_set['RFtest40'] = deepcopy(benchmark)
test_set['RFtest40']['description'] = 'Lx=44.0-46.0'
test_set['RFtest40']['cols']['L'] = [44.0, 46.0]

test_set['RFtest41'] = deepcopy(benchmark)
test_set['RFtest41']['description'] = 'Lx=43.0-44.0'
test_set['RFtest41']['cols']['L']  = [43.0, 44.0]

test_set['RFtest42'] = deepcopy(benchmark)
test_set['RFtest42']['description'] = 'No R500'
test_set['RFtest42']['cols']  = {'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]}

test_set['RFtest43'] = deepcopy(benchmark)
test_set['RFtest43']['simulations'] = ['C4', 'C5', 'C6', 'C11', 'C13', 'C14']
test_set['RFtest43']['description'] = '-C3-C7-C8-C9-C10-C12-C15'

test_set['RFtest44'] = deepcopy(benchmark)
test_set['RFtest44']['p_threshold'] = 0.04

test_set['RFtest45'] = deepcopy(benchmark)
test_set['RFtest45']['description'] = 'No R500'
test_set['RFtest45']['cols']  = {'Mgas':[12.0, 14.0], 'L':[43, 46], 'T':[-0.2, 1], 'z':[0, 1.0]}
test_set['RFtest45']['p_threshold'] = 0.05

class YamlCreator:
    def __init__(self, fname='default_test_para.yaml'):
        self.file_path = './config/'
        self.para_fname = os.path.join(self.file_path, fname) 
        self.test_set = test_set

    def create_test_para_yaml(self):
        with open(self.para_fname, 'w') as f:
            yaml.dump(self.test_set, f, sort_keys=False)

    def load_yaml(self):
        with open(self.para_fname, 'r') as file:
            return yaml.safe_load(file)


if __name__ == "__main__":
    yaml_creter = YamlCreator(fname='default_test_para.yaml')
    yaml_creter.create_test_para_yaml()