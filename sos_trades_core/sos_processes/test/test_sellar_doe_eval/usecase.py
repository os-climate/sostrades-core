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

from sos_trades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self):
        super().__init__(__file__)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        dspace_dict = {'variable': ['x', 'z'],

                       'lower_bnd': [0., [-10., 0.]],
                       'upper_bnd': [10., [10., 10.]],

                       }
        dspace = pd.DataFrame(dspace_dict)

        input_selection_x_z = {'selected_input': [False, True, False, False, True],
                               'full_name': ['DoEEval.Sellar_Problem.local_dv', 'x', 'y_1',
                                             'y_2',
                                             'z']}
        input_selection_x_z = pd.DataFrame(input_selection_x_z)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj',
                                                    'y_1', 'y_2']}
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        disc_dict = {}
        # DoE inputs
        n_samples = 100
        disc_dict[f'{ns}.DoEEval.sampling_algo'] = "fullfact"
        disc_dict[f'{ns}.DoEEval.design_space'] = dspace
        disc_dict[f'{ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{ns}.DoEEval.eval_inputs'] = input_selection_x_z
        disc_dict[f'{ns}.DoEEval.eval_outputs'] = output_selection_obj_y1_y2

        # Sellar inputs
        local_dv = 10.
        disc_dict[f'{ns}.x'] = 1.
        disc_dict[f'{ns}.y_1'] = 1.
        disc_dict[f'{ns}.y_2'] = 1.
        disc_dict[f'{ns}.z'] = array([1., 1.])
        disc_dict[f'{ns}.DoEEval.Sellar_Problem.local_dv'] = local_dv

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
