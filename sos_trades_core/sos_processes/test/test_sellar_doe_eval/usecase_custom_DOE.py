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
        disc_dict[f'{ns}.DoEEval.sampling_algo'] = "CustomDOE"
        disc_dict[f'{ns}.DoEEval.eval_inputs'] = input_selection_x_z
        disc_dict[f'{ns}.DoEEval.eval_outputs'] = output_selection_obj_y1_y2
        x_values = [9.379763880395856, 8.88644794300546,
                    3.7137135749628882, 0.0417022004702574, 6.954954792150857]
        z_values = [array([1.515949043849158, 5.6317362409322165]),
                    array([-1.1962705421254114, 6.523436208612142]),
                    array([-1.9947578026244557, 4.822570933860785]
                          ), array([1.7490668861813, 3.617234050834533]),
                    array([-9.316161097119341, 9.918161285133076])]

        samples_dict = {'x': x_values, 'z': z_values}
        samples_df = pd.DataFrame(samples_dict)
        disc_dict[f'{ns}.DoEEval.custom_samples_df'] = samples_df

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
