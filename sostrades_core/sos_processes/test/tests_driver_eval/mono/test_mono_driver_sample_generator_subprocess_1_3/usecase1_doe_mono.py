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

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.proc_builder.process_builder_parameter_type import (
    ProcessBuilderParameterType,
)


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for disc1 disc2 eval generator doe with reference
        """

        ns = f'{self.study_name}'

        # coupling_name = 'subprocess'
        coupling_name = 'D1_D3_Coupling'

        dspace_dict = {'variable': [f'Eval.{coupling_name}.Disc1.b', 'Eval.{coupling_name}.z', ],
                       'lower_bnd': [0., 0.],
                       'upper_bnd': [10., 10.],
                       }

        dspace = pd.DataFrame(dspace_dict)

        input_selection_b_z = pd.DataFrame({'selected_input': [True, True],
                                            'full_name': [f'{coupling_name}.Disc1.b', f'{coupling_name}.z']
                                            })

        input_selection_b_z = pd.DataFrame(input_selection_b_z)

        output_selection_obj_y_o = {'selected_output': [False, True, True],
                                    'full_name': [f'{coupling_name}.indicator', f'{coupling_name}.y',
                                                  f'{coupling_name}.o']}

        output_selection_obj_y_o = pd.DataFrame(output_selection_obj_y_o)

        disc_dict = {}
        # DoE + Eval inputs
        n_samples = 20
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'doe_algo'
        disc_dict[f'{ns}.SampleGenerator.sampling_generation_mode'] = 'at_run_time'
        disc_dict[f'{ns}.SampleGenerator.sampling_algo'] = "lhs"
        disc_dict[f'{ns}.SampleGenerator.design_space'] = dspace
        disc_dict[f'{ns}.SampleGenerator.algo_options'] = {
            'n_samples': n_samples}

        disc_dict[f'{ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_b_z
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_obj_y_o

        with_modal = True
        anonymize_input_dict_from_usecase = {}
        if with_modal:
            repo = 'sostrades_core.sos_processes.test.disc1_disc3'
            mod_id = 'test_disc1_disc3_coupling'
            my_usecase = 'Empty'
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            disc_dict[f'{ns}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # Nested process inputs
        self.a = 3
        self.x = 2.
        self.b = 4
        self.constant = 3
        self.power = 2
        self.z = 1.2

        disc_dict[f'{ns}.Eval.{coupling_name}.a'] = self.a
        disc_dict[f'{ns}.Eval.{coupling_name}.x'] = self.x
        disc_dict[f'{ns}.Eval.{coupling_name}.z'] = self.z
        disc_dict[f'{ns}.Eval.{coupling_name}.Disc1.b'] = self.b
        disc_dict[f'{ns}.Eval.{coupling_name}.Disc3.constant'] = self.constant
        disc_dict[f'{ns}.Eval.{coupling_name}.Disc3.power'] = self.power

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    # uc_cls.execution_engine.display_treeview_nodes(True)
    uc_cls.run()
