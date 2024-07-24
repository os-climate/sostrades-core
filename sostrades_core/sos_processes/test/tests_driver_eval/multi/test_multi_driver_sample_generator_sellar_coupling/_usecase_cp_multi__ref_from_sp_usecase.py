'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2024/05/16 Copyright 2023 Capgemini

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
from sostrades_core.tools.proc_builder.process_builder_parameter_type import (
    ProcessBuilderParameterType,
)


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for Eval and generator cp with reference and import from usecase of Sellar Problem
        """

        ns = f'{self.study_name}'

        # ======================================================================
        # dict_of_list_values = {
        #     'SellarCoupling.x': [array([3.]), array([4.])],
        #     'SellarCoupling.z': [array([-10., 0.])],
        #     'SellarCoupling.Sellar_Problem.local_dv': [10.],
        #     'SellarCoupling.y_1': [array([1.])],
        #     'SellarCoupling.y_2': [array([1.])]
        # }
        # ======================================================================

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

        # Subprocess and usecase
        repo = 'sostrades_core.sos_processes.test.sellar'
        mod_id = 'test_sellar_coupling'
        my_usecase = 'usecase'

        # Find anonymised dict
        anonymize_input_dict_from_usecase = self.static_load_raw_usecase_data(
            repo, mod_id, my_usecase)

        disc_dict = {}
        # CP + Eval inputs
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'
        disc_dict[f'{ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_cp_x_z
        disc_dict[f'{ns}.Eval.instance_reference'] = True
        disc_dict[f'{ns}.Eval.reference_mode'] = 'linked_mode'

        with_modal = True
        if with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            disc_dict[f'{ns}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # Sellar reference inputs
        # Provided by usecase import

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
