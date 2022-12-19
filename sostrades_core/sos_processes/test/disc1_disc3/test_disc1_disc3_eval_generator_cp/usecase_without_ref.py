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
import pandas as pd
from numpy import array

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.Eval.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.Sample_Generator.sampling_method'] = 'cartesian_product'

        b1 = 4
        b2 = 2
        z1 = 1.2
        z2 = 1.5
        dict_of_list_values = {
            'Disc1.b': [b1, b2],
            'Disc3.z': [z1, z2]
        }
        list_of_values_b_z = [[], dict_of_list_values['Disc1.b'],
                              [], [], dict_of_list_values['Disc3.z']]
        input_selection_cp_b_z = pd.DataFrame({'selected_input': [False, True, False, False, True],
                                               'full_name': ['', 'Disc1.b', '', '', 'z'],
                                               'list_of_values': list_of_values_b_z
                                               })
        dict_values[f'{self.study_name}.Eval.eval_inputs_cp'] = input_selection_cp_b_z

        self.x1 = 2.
        self.a1 = 3
        self.constant = 3
        self.power = 2
        self.z1 = 1.2

        scenario_list = ['scenario_1', 'scenario_2',
                         'scenario_3', 'scenario_4']
        for scenario in scenario_list:
            dict_values[f'{self.study_name}.Eval.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.Eval.{scenario}.x'] = self.x1
            dict_values[f'{self.study_name}.Eval.{scenario}.Disc3.constant'] = self.constant
            dict_values[f'{self.study_name}.Eval.{scenario}.Disc3.power'] = self.power
            dict_values[f'{self.study_name}.Eval.{scenario}.z'] = self.z1

        return [dict_values]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
