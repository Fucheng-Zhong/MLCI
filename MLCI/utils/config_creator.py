import yaml
import os


benchmark = {'mode':'RF', 'R_type':'R_corr', 'agn_feedback':0.0, 'proportional':True, 'noise_level':[1/3, 1/20],
            'leaf_size':1e4, 'max_num':1e4, 'sample_num':100, 'max_depth':10, 'exclude_outlier':False, 'description':'Fiducial',
            'delta_prop':1,
            'simulations': [f'C{i+3}' for i in range(13)]}
test_set = {}
# observed level error, ordinary agn feedback, different simulations
test_set['RFtest1'] = benchmark.copy()
test_set['RFtest1']['description'] = 'Fiducial'

test_set['RFtest2'] = benchmark.copy()
test_set['RFtest2']['simulations'] = [f'C{i+2}' for i in range(14)]
test_set['RFtest2']['description'] = '+C2'

test_set['RFtest3'] = benchmark.copy()
test_set['RFtest3']['simulations'] = ['C2','C3', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
test_set['RFtest3']['description'] = '+C2-C4'


test_set['RFtest4'] = benchmark.copy()
test_set['RFtest4']['simulations'] = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C11', 'C13', 'C14']
test_set['RFtest4']['description'] = '-C3-C10-C12-C15'

test_set['RFtest5'] = benchmark.copy()
test_set['RFtest5']['simulations'] = ['C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
test_set['RFtest5']['description'] = '-C3-C4-C5-C6'


test_set['RFtest6'] = benchmark.copy()
test_set['RFtest6']['agn_feedback'] = '+delta'
test_set['RFtest6']['description'] = 'feedback +delta'


test_set['RFtest7'] = benchmark.copy()
test_set['RFtest7']['agn_feedback'] = '-delta'
test_set['RFtest7']['description'] = 'feedback -delta'


test_set['RFtest8'] = benchmark.copy()
test_set['RFtest8']['agn_feedback'] = 0.1
test_set['RFtest8']['description'] = 'feedback +0.1'


test_set['RFtest9'] = benchmark.copy()
test_set['RFtest9']['agn_feedback'] = -0.1
test_set['RFtest9']['description'] = 'feedback -0.1'


test_set['RFtest10'] = benchmark.copy()
test_set['RFtest10']['agn_feedback'] = 0.2
test_set['RFtest10']['description'] = 'feedback +0.2'


test_set['RFtest11'] = benchmark.copy()
test_set['RFtest11']['agn_feedback'] = -0.2
test_set['RFtest11']['description'] = 'feedback -0.2'

test_set['RFtest12'] = benchmark.copy()
test_set['RFtest12']['agn_feedback'] = 0.3
test_set['RFtest12']['description'] = 'feedback +0.3'

test_set['RFtest13'] = benchmark.copy()
test_set['RFtest13']['agn_feedback'] = -0.3
test_set['RFtest13']['description'] = 'feedback -0.3'

test_set['RFtest14'] = benchmark.copy()
test_set['RFtest14']['noise_level'] = [0.1, 0.1]
test_set['RFtest14']['description'] = 'noise level 0.1'

test_set['RFtest15'] = benchmark.copy()
test_set['RFtest15']['noise_level'] = [0.01, 0.01]
test_set['RFtest15']['description'] = 'noise level 0.01'

#=======
test_set['RFtest16'] = benchmark.copy()
test_set['RFtest16']['simulations'] = [f'C{i+1}' for i in range(15)]
test_set['RFtest16']['description'] = 'all simulations'

test_set['RFtest17'] = benchmark.copy()
test_set['RFtest17']['leaf_size'] = 1e3
test_set['RFtest17']['description'] = 'leaf size 1000'

test_set['RFtest18'] = benchmark.copy()
test_set['RFtest18']['leaf_size'] = 1e3
test_set['RFtest18']['noise_level'] = [0.01, 0.01]
test_set['RFtest18']['description'] = 'noise 0.01+leaf 1000'

test_set['RFtest19'] = benchmark.copy()
test_set['RFtest19']['agn_feedback'] = '+delta'
test_set['RFtest19']['delta_prop'] = 2
test_set['RFtest19']['description'] = 'feedback +2delta'


test_set['RFtest20'] = benchmark.copy()
test_set['RFtest20']['agn_feedback'] = '-delta'
test_set['RFtest20']['delta_prop'] = 2
test_set['RFtest20']['description'] = 'feedback -2delta'


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