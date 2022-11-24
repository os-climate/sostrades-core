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

    # def setup_usecase(self):
    #     """
    #     Usecase for lhs DoE and Eval on x variable of Sellar Problem
    #     """
    #
    #     ns = f'{self.study_name}'
    #     dspace_dict = {'variable': ['x'],
    #
    #                      'lower_bnd': [0.],
    #                      'upper_bnd': [10.],
    #
    #                      }
    #     dspace = pd.DataFrame(dspace_dict)
    #
    #     input_selection_x = {'selected_input': [False, True, False, False, False],
    #                          'full_name': ['Eval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
    #                                        'y_2',
    #                                        'z']}
    #     input_selection_x = pd.DataFrame(input_selection_x)
    #
    #     output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
    #                                   'full_name': ['c_1', 'c_2', 'obj',
    #                                                 'y_1', 'y_2']}
    #     output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)
    #
    #     disc_dict = {}
    #     # DoE inputs
    #     n_samples = 100
    #     disc_dict[f'{ns}.DoE.sampling_algo'] = "lhs"
    #     disc_dict[f'{ns}.DoE.design_space'] = dspace
    #     disc_dict[f'{ns}.DoE.algo_options'] = {'n_samples': n_samples}
    #     disc_dict[f'{ns}.eval_inputs'] = input_selection_x
    #     disc_dict[f'{ns}.eval_outputs'] = output_selection_obj_y1_y2
    #
    #     # Sellar inputs
    #     local_dv = 10.
    #     disc_dict[f'{ns}.x'] = array([1.])
    #     disc_dict[f'{ns}.y_1'] = array([1.])
    #     disc_dict[f'{ns}.y_2'] = array([1.])
    #     disc_dict[f'{ns}.z'] = array([1., 1.])
    #     disc_dict[f'{ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
    #
    #     return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
