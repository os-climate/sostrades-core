'''
Copyright 2022 Airbus SA

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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.study_manager.study_manager import StudyManager
from sos_trades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType
import pandas as pd


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_proc_build_disc1_all_types process.
    This process instantiates a DOE on the Discipline directly from the discipline.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # provide a process (with disciplines) to the set doe
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_proc_build_disc1_all_types'
        anonymize_input_dict_from_usecase = {}
        anonymize_input_dict_from_usecase['<study_ph>.Disc1.x'] = 5.5
        anonymize_input_dict_from_usecase['<study_ph>.Disc1.a'] = 3
        anonymize_input_dict_from_usecase['<study_ph>.Disc1.b'] = 2
        anonymize_input_dict_from_usecase['<study_ph>.Disc1.name'] = 'A1'
        anonymize_input_dict_from_usecase['<study_ph>.Disc1.x_dict'] = {}

        process_builder_parameter_type = ProcessBuilderParameterType(mod_id, repo, 'usecase1')
        process_builder_parameter_type.usecase_data = anonymize_input_dict_from_usecase

        ######### Numerical values   ####

        input_selection = {'selected_input': [True],
                           'full_name': ['DoE_Eval.Disc1.x']}
        input_selection = pd.DataFrame(input_selection)

        output_selection = {'selected_output': [True, True],
                            'full_name': ['DoE_Eval.Disc1.indicator', 'DoE_Eval.Disc1.y_dict']}  # cannot use 'DoE_Eval.Disc1.y' !
        output_selection = pd.DataFrame(output_selection)

        dspace_dict = {'variable': ['DoE_Eval.Disc1.x'],
                       'lower_bnd': [-5.],
                       'upper_bnd': [+5.],
                       }
        my_doe_algo = "lhs"
        n_samples = 4
        dspace = pd.DataFrame(dspace_dict)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        values_dict[f'{self.study_name}.DoE_Eval.sub_process_inputs'] = process_builder_parameter_type.to_data_manager_dict()
        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        return [values_dict]


if __name__ == '__main__':
    uc_cls = Study()
    print(uc_cls.study_name)
    uc_cls.load_data()
    uc_cls.run(for_test=True)
