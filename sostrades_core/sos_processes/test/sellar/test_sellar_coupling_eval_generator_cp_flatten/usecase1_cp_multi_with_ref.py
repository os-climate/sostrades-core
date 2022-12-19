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
        Usecase for Eval and generator cp with reference of Sellar Problem in flatten mode
        """

        ns = f'{self.study_name}'

        dict_of_list_values = {
            'SellarCoupling.x': [array([3.]), array([4.])],
            'SellarCoupling.z': [array([-10., 0.])],
            'SellarCoupling.Sellar_Problem.local_dv': [10.],
            'SellarCoupling.y_1': [array([1.])],
            'SellarCoupling.y_2': [array([1.])]
        }
        list_of_values = [dict_of_list_values['SellarCoupling.Sellar_Problem.local_dv'], dict_of_list_values['SellarCoupling.x'],
                          dict_of_list_values['SellarCoupling.y_1'], dict_of_list_values['SellarCoupling.y_2'], dict_of_list_values['SellarCoupling.z']]

        input_selection_cp_x_z = {'selected_input': [False, True, True, True, True],
                                  'full_name': ['SellarCoupling.Sellar_Problem.local_dv', 'SellarCoupling.x', 'SellarCoupling.y_1',
                                                'SellarCoupling.y_2',
                                                'SellarCoupling.z'],
                                  'list_of_values': list_of_values
                                  }
        input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

        anonymize_input_dict_from_usecase = {}

        disc_dict = {}
        # CP + Eval inputs
        disc_dict[f'{ns}.Eval.builder_mode'] = 'multi_instance'
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'
        disc_dict[f'{ns}.Eval.eval_inputs_cp'] = input_selection_cp_x_z
        disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase
        disc_dict[f'{ns}.Eval.instance_reference'] = True

        # Sellar referene inputs
        local_dv = 10.
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.x'] = array([
                                                                           2.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.y_1'] = array([
                                                                             1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.y_2'] = array([
                                                                             1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.z'] = array([
                                                                           1., 1.])
        disc_dict[f'{ns}.Eval.ReferenceScenario.SellarCoupling.Sellar_Problem.local_dv'] = local_dv

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
