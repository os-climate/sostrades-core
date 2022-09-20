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
import pandas as pd


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_disc_hessian_doe_eval_from_disc process.
    This process instantiates a DOE on the Hessian Discipline directly from the discipline.
    It uses the 1 wrapped discipline : sos_trades_core.sos_wrapping.test_discs.disc_hessian.DiscHessian.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # provide a process (with disciplines) to the set doe
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        anonymize_input_dict_from_usecase = {}
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.x'] = 12.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.y'] = 13.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.ax2'] = 14.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.by2'] = 15.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.cx'] = 16.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.dy'] = 17.0
        anonymize_input_dict_from_usecase['<study_ph>.Hessian.exy'] = 112.0

        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'usecase2'
        sub_process_inputs_dict['usecase_data'] = anonymize_input_dict_from_usecase

        ######### Numerical values   ####

        input_selection = {'selected_input': [True, True, False, False, False, False, False],
                           'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                         'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                         'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection = pd.DataFrame(input_selection)

        output_selection = {'selected_output': [True],
                            'full_name': ['DoE_Eval.Hessian.z']}
        output_selection = pd.DataFrame(output_selection)

        dspace_dict = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                       'lower_bnd': [-5., -5.],
                       'upper_bnd': [+5., +5.],
                       }
        my_doe_algo = "lhs"
        n_samples = 4
        dspace = pd.DataFrame(dspace_dict)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        values_dict[f'{self.study_name}.DoE_Eval.sub_process_inputs'] = sub_process_inputs_dict
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
