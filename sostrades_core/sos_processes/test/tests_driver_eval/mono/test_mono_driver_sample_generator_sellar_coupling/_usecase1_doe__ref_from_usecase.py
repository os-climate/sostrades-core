'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/11-2024/05/16 Copyright 2023 Capgemini

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
        Usecase for lhs DoE and Eval on x variable of Sellar Problem
        """

        coupling_name = 'SellarCoupling'
        ns = f'{self.study_name}'
        dspace_dict = {'variable': [f'{coupling_name}.x'],
                       'lower_bnd': [0.],
                       'upper_bnd': [10.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        input_selection_x = {'selected_input': [False, True, False, False, False],
                             'full_name': [f'{coupling_name}.Sellar_Problem.local_dv', f'{coupling_name}.x', f'{coupling_name}.y_1',
                                           f'{coupling_name}.y_2',
                                           f'{coupling_name}.z']}
        input_selection_x = pd.DataFrame(input_selection_x)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': [f'{coupling_name}.c_1', f'{coupling_name}.c_2', f'{coupling_name}.obj',
                                                    f'{coupling_name}.y_1', f'{coupling_name}.y_2']}
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        # Sub_Process and use case selection

        # Subprocess and usecase
        repo = 'sostrades_core.sos_processes.test.sellar'
        mod_id = 'test_sellar_coupling'
        my_usecase = 'usecase'

        # Find anonymised dict
        based_on_uc_name = True
        if based_on_uc_name:  # Full dictionary based on usecase name and process
            anonymize_input_dict_from_usecase = self.static_load_raw_usecase_data(
                repo, mod_id, my_usecase)
        else:  # Manualy provided restricted dictionary
            anonymize_input_dict_from_usecase = {}
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = [
                1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = [
                1., 1.]
            anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.

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
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_x
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_obj_y1_y2

        with_modal = True
        if with_modal:
            process_builder_parameter_type = ProcessBuilderParameterType(
                mod_id, repo, my_usecase)
            process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase
            disc_dict[f'{ns}.Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        else:
            disc_dict[f'{ns}.Eval.usecase_data'] = anonymize_input_dict_from_usecase

        # Sellar inputs
        # Provided by usecase import

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
