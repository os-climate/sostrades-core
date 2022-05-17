'''
Copyright 2022 Airbus SA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd

class Study(StudyManager):
    '''This is an example of usecase study for
     the test_multiscenario_of_DoE_Eval specify process.
    This process is an example of a multiscenario of doe eval.
    It uses the 2 wrapped disciplines : disc1_scenario.Disc1
     (orchestrated by the test_disc1_scenario process) and disc3_scenario.Disc3.
    '''
    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)


    def setup_usecase(self):
        ######### Numerical values   ####
        x_1 = 2
        x_2_a = 4
        x_2_b = 5

        a_1 = 3
        b_1 = 4
        a_2 = 6
        b_2 = 2

        constant = 3
        power = 2
        z_1 =  1.2
        z_2 =  1.5

        my_doe_algo = "lhs"
        n_samples = 4


        ######### Selection of variables and DS  ####
        input_selection_z_scenario_1 = {
            'selected_input': [False, False, False, True, False, False],
            'full_name': ['x', 'a','b','multi_scenarios.scenario_1.Disc3.z','constant','power']}
        input_selection_z_scenario_1 = pd.DataFrame(input_selection_z_scenario_1)

        input_selection_z_scenario_2 = {
            'selected_input': [False, False, False, True, False, False],
            'full_name': ['x', 'a','b','multi_scenarios.scenario_2.Disc3.z','constant','power']}
        input_selection_z_scenario_2 = pd.DataFrame(input_selection_z_scenario_2)

        output_selection_o_scenario_1 = {
            'selected_output': [False, False, True],
            'full_name': ['indicator', 'y', 'multi_scenarios.scenario_1.o']}
        output_selection_o_scenario_1 = pd.DataFrame(output_selection_o_scenario_1)

        output_selection_o_scenario_2 = {
            'selected_output': [False, False, True],
            'full_name': ['indicator', 'y', 'multi_scenarios.scenario_2.o']}
        output_selection_o_scenario_2 = pd.DataFrame(output_selection_o_scenario_2)

        dspace_dict_z = {'variable': ['z'],
                          'lower_bnd': [0.],
                          'upper_bnd': [10.],
                          'enable_variable': [True],
                          'activated_elem': [[True]]}
        dspace_z = pd.DataFrame(dspace_dict_z)

        my_name_list = ['name_1', 'name_2']

        my_x_trade = [x_1, x_2_a]

        my_trade_variables = {'name_1.x': 'float'}

        ######### Fill the dictionary for dm   ####
        dict_values = {}

        prefix = f'{self.study_name}.multi_scenarios'

        dict_values[f'{self.study_name}.name_2.x'] = x_2_b
        dict_values[f'{self.study_name}.name_1.a'] = a_1
        dict_values[f'{self.study_name}.name_2.a'] = a_2

        dict_values[f'{prefix}.name_1.x_trade'] = my_x_trade
        dict_values[f'{prefix}.trade_variables'] = my_trade_variables

        dict_values[f'{prefix}.name_list'] =  my_name_list

        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc1.name_1.b'] = b_1
        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc1.name_2.b'] = b_2


        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc1.name_1.b'] = b_1
        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc1.name_2.b'] = b_2

        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc3.constant'] = constant
        dict_values[f'{prefix}.scenario_1.DoE_Eval.Disc3.power'] = power
        dict_values[f'{prefix}.scenario_1.Disc3.z'] = z_1 # reference value (computed in any case)

        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.constant'] = constant
        dict_values[f'{prefix}.scenario_2.DoE_Eval.Disc3.power'] = power
        dict_values[f'{prefix}.scenario_2.Disc3.z'] = z_2 # reference value (computed in any case)

        dict_values[f'{prefix}.scenario_1.DoE_Eval.sampling_algo'] = my_doe_algo
        dict_values[f'{prefix}.scenario_1.DoE_Eval.eval_inputs'] = input_selection_z_scenario_1
        dict_values[f'{prefix}.scenario_1.DoE_Eval.eval_outputs'] = output_selection_o_scenario_1


        dict_values[f'{prefix}.scenario_2.DoE_Eval.sampling_algo'] = my_doe_algo
        dict_values[f'{prefix}.scenario_2.DoE_Eval.eval_inputs'] = input_selection_z_scenario_2
        dict_values[f'{prefix}.scenario_2.DoE_Eval.eval_outputs'] = output_selection_o_scenario_2

        dict_values[f'{prefix}.scenario_1.DoE_Eval.design_space'] = dspace_z
        dict_values[f'{prefix}.scenario_1.DoE_Eval.algo_options'] = {'n_samples': n_samples}
        dict_values[f'{prefix}.scenario_2.DoE_Eval.design_space'] = dspace_z
        dict_values[f'{prefix}.scenario_2.DoE_Eval.algo_options'] = {'n_samples': n_samples}

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
