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
        dspace_dict = {'variable': ['Eval.x'],
                       'lower_bnd': [0.],
                       'upper_bnd': [10.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': ['Eval.SellarCoupling.Sellar_Problem.local_dv', 'Eval.x', 'Eval.y_1',
                                           'Eval.y_2',
                                           'Eval.z']}
        input_selection_x = pd.DataFrame(input_selection_x)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['Eval.c_1', 'Eval.c_2', 'Eval.obj',
                                                    'Eval.y_1', 'Eval.y_2']}
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        repo = 'sostrades_core.sos_processes.test'
        mod_id = 'test_sellar_coupling'
        my_usecase = 'usecase'
        anonymize_input_dict_from_usecase = self.static_load_raw_usecase_data(
            repo, mod_id, my_usecase)

        disc_dict = {}
        # DoE + Eval inputs
        disc_dict[f'{ns}.Eval.builder_mode'] = 'mono_instance'
        n_samples = 20
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'doe_algo'
        disc_dict[f'{ns}.SampleGenerator.sampling_algo'] = "lhs"
        disc_dict[f'{ns}.SampleGenerator.design_space'] = dspace
        disc_dict[f'{ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples}
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x
        disc_dict[f'{ns}.Eval.eval_outputs'] = output_selection_obj_y1_y2
        disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        disc_dict[f'{ns}.Eval.instance_reference'] = True

        # Sellar inputs
        local_dv = 10.
        disc_dict[f'{ns}.Eval.x'] = array([1.])
        disc_dict[f'{ns}.Eval.y_1'] = array([1.])
        disc_dict[f'{ns}.Eval.y_2'] = array([1.])
        disc_dict[f'{ns}.Eval.z'] = array([1., 1.])
        disc_dict[f'{ns}.Eval.SellarCoupling.Sellar_Problem.local_dv'] = local_dv

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
