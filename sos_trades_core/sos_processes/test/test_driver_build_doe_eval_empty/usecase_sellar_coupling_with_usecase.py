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
from numpy import array


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_sellar_coupling process.
    This process instantiates a DOE on the Discipline directly from the discipline.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # provide a process (with disciplines) to the set doe
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_sellar_coupling'
        anonymize_input_dict_from_usecase = {}
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.x'] = array([
                                                                                 1.])
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_1'] = array([
                                                                                   1.])
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.y_2'] = array([
                                                                                   1.])
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.z'] = array([
                                                                                 1., 1.])
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.Sellar_Problem.local_dv'] = 10.
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.max_mda_iter'] = 100
        anonymize_input_dict_from_usecase['<study_ph>.SellarCoupling.tolerance'] = 1e-12

        sub_process_inputs_dict = {}
        sub_process_inputs_dict['process_repository'] = repo
        sub_process_inputs_dict['process_name'] = mod_id
        sub_process_inputs_dict['usecase_name'] = 'usecase'
        sub_process_inputs_dict['usecase_data'] = anonymize_input_dict_from_usecase

        ######### Numerical values   ####
        input_selection = {'selected_input': [True, True],
                           'full_name': ['DoE_Eval.SellarCoupling.x',
                                         'DoE_Eval.SellarCoupling.z']}
        input_selection = pd.DataFrame(input_selection)

        output_selection = {'selected_output': [False, False, True, True, True],
                            'full_name': ['DoE_Eval.SellarCoupling.c_1', 'DoE_Eval.SellarCoupling.c_2', 'DoE_Eval.SellarCoupling.obj',
                                          'DoE_Eval.SellarCoupling.y_1', 'DoE_Eval.SellarCoupling.y_2']}
        output_selection = pd.DataFrame(output_selection)

        dspace_dict = {'variable': ['DoE_Eval.SellarCoupling.x', 'DoE_Eval.SellarCoupling.z'],

                       'lower_bnd': [0., [-10., 0.]],
                       'upper_bnd': [10., [10., 10.]],
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
