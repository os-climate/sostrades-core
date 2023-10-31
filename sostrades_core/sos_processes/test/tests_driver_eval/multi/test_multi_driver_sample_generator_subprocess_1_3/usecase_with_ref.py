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
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for disc1 disc2 eval generator cp with reference
        """
        dict_values = {}
        dict_values[f'{self.study_name}.Sample_Generator.sampling_method'] = 'cartesian_product'
        dict_values[f'{self.study_name}.Eval.with_sample_generator'] = True
        dict_values[f'{self.study_name}.Eval.instance_reference'] = True
        dict_values[f'{self.study_name}.Eval.reference_mode'] = 'linked_mode'

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

        with_modal = True
        anonymize_input_dict_from_usecase = {}
        if with_modal:
            repo = 'sostrades_core.sos_processes.test.disc1_disc3'
            mod_id = 'test_disc1_disc3_list'
            my_usecase = 'Empty'
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            dict_values[f'{self.study_name}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            dict_values[f'{self.study_name}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # reference var values
        self.x = 2.
        self.a = 3
        self.b = 8
        self.z = 12
        self.constant = 3
        self.power = 2
        # configure the Reference scenario
        # Non-trade variables (to propagate)
        dict_values[self.study_name + '.Eval.ReferenceScenario.a'] = self.a
        dict_values[self.study_name + '.Eval.ReferenceScenario.x'] = self.x
        dict_values[self.study_name +
                    '.Eval.ReferenceScenario.Disc3.constant'] = self.constant
        dict_values[self.study_name +
                    '.Eval.ReferenceScenario.Disc3.power'] = self.power
        # Trade variables reference (not to propagate)
        dict_values[self.study_name +
                    '.Eval.ReferenceScenario.Disc1.b'] = self.b
        dict_values[self.study_name + '.Eval.ReferenceScenario.z'] = self.z

        return [dict_values]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    uc_cls.run()
