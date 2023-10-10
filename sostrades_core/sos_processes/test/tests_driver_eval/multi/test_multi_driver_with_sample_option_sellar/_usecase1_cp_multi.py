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
        """
        Usecase for lhs DoE and Eval on x variable of Sellar Problem
        """

        ns = f'{self.study_name}'
        ns = f'{self.study_name}'

        dict_of_list_values = {
            'x': [0., 3., 4., 5., 7.],
            'z': [[-10., 0.], [-5., 4.], [10, 10]]
        }
        list_of_values = [[], dict_of_list_values['x'],
                          [], [], dict_of_list_values['z']]

        input_selection_cp_x_z = {'selected_input': [False, True, False, False, True],
                                  'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                'y_2',
                                                'z'],
                                  'list_of_values': list_of_values
                                  }
        input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

        
        disc_dict = {}
        # DoE + Eval inputs
        disc_dict[f'{ns}.Eval.builder_mode'] = 'multi_instance'
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'
        disc_dict[f'{ns}.eval_inputs_cp'] = input_selection_cp_x_z

        # Sellar inputs
        local_dv = 10.
        disc_dict[f'{ns}.x'] = array([1.])
        disc_dict[f'{ns}.y_1'] = array([1.])
        disc_dict[f'{ns}.y_2'] = array([1.])
        disc_dict[f'{ns}.z'] = array([1., 1.])
        disc_dict[f'{ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
