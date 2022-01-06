'''
Copyright 2022 Airbus SAS

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

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
            'scenario_1', 'scenario_2']

        ns_scenario_1_disc_1 = f'multi_scenarios.scenario_1.MORPH_MATRIX.Disc1'
        ns_scenario_2_disc_1 = f'multi_scenarios.scenario_2.MORPH_MATRIX.Disc1'

        eval_inputs_scenario_1 = pd.DataFrame({'selected_input': [True, True, False, False],
                                               'name': ['a', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1],
                                               'input_variable_name': ['a_list', 'b_list', '', '']})

        eval_inputs_scenario_2 = pd.DataFrame({'selected_input': [False, False, False, True],
                                               'name': ['a', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1],
                                               'input_variable_name': ['', '', '', 'x_list']})

        eval_outputs_scenario_1 = pd.DataFrame({'selected_output': [False, False, True, False],
                                                'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                                'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1],
                                                'output_variable_name': ['', '', 'y_out', '']})

        eval_outputs_scenario_2 = pd.DataFrame({'selected_output': [False, False, True, False],
                                                'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                                'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1],
                                                'output_variable_name': ['', '', 'y_out', '']})

        # set eval_inputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_2
        # set eval_outputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_2
        # set eval_inputs values list
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.a_list'] = [
            0, 2]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.b_list'] = [
            2, 5]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.x_list'] = [
            5.0, 6.0, 7.0]
        # set other inputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.x'] = 5.0
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.name'] = 'A1'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.a'] = 1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.b'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.name'] = 'A1'

        return dict_values


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
