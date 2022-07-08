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

        self.morph_matrix = 'MORPH_MATRIX'

        activation_df = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                      'scenario_name': ['scenario_A', 'scenario_B', 'scenario_C', 'scenario_D'],
                                      'AC1_dyn_input_list': [1.0, 1.0, 3.0, 3.0, ],
                                      'b_list': [0.0, 2.0, 0.0, 2.0], })
        eval_inputs = pd.DataFrame({'selected_input': [False, True, False, True, False],
                                    'name': ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'b', 'x'],
                                    'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', ],
                                    'input_variable_name': ['', 'AC1_dyn_input_list', '', 'b_list', '']})

        eval_outputs = pd.DataFrame({'selected_output': [True, False, False, False, True],
                                     'name': ['AC1.dyn_output', 'AC2.dyn_output', 'indicator', 'residuals_history', 'y'],
                                     'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}'],
                                     'output_variable_name': ['dyn_outputs_out', '', '', '', 'y_out']})
        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.AC_list': ['AC1', 'AC2'],
            f'{self.study_name}.{self.morph_matrix}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.morph_matrix}.eval_outputs': eval_outputs,
            f'{self.study_name}.{self.morph_matrix}.b_list': [0.0, 2.0],
            f'{self.study_name}.{self.morph_matrix}.x': 5.0,
            f'{self.study_name}.{self.morph_matrix}.AC1_dyn_input_list': [1.0, 3.0],
            f'{self.study_name}.{self.morph_matrix}.activation_morphological_matrix': activation_df,
            f'{self.study_name}.{self.morph_matrix}.Disc1.a': 3}

        return dict_values


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes(display_variables=True)
    uc_cls.run()
