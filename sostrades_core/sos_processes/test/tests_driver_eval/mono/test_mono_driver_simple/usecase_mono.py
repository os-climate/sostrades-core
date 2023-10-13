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

        samples =  {'selected_scenario': [False, True, False],
                    'scenario_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_a = {'selected_input': [False, True, False],
                             'full_name': ['x', 'Disc1.a', 'Disc1.b']}
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {'selected_output': [False, True],
                                'full_name': ['y', 'Disc1.indicator']}
        output_selection_ind = pd.DataFrame(output_selection_ind)

        disc_dict = {}
        # Eval inputs
        disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_a
        disc_dict[f'{ns}.Eval.eval_outputs'] = output_selection_ind

        # a_values = [array([2.0]), array([4.0]), array(
        #     [6.0]), array([8.0]), array([10.0])]
        a_values = [2.0, 4.0, 6.0, 8.0, 10.0]
        samples_dict = {'selected_scenario': [True]*5,
                        'scenario_name':[f'scenario_{i}' for i in range(1,6)],
                        'Disc1.a': a_values}
        samples_df = pd.DataFrame(samples_dict)

        disc_dict[f'{ns}.Eval.samples_df'] = samples_df

        # Disc1 inputs
        # disc_dict[f'{ns}.x'] = array([10.])
        # disc_dict[f'{ns}.Eval.Disc1.a'] = array([5.])
        # disc_dict[f'{ns}.Eval.Disc1.b'] = array([25.])
        # disc_dict[f'{ns}.y'] = array([4.])
        # disc_dict[f'{ns}.Eval.Disc1.indicator'] = array([53.])
        disc_dict[f'{ns}.Eval.x'] = 10.
        disc_dict[f'{ns}.Eval.Disc1.a'] = 5.
        disc_dict[f'{ns}.Eval.Disc1.b'] = 25.
        disc_dict[f'{ns}.y'] = 4.
        disc_dict[f'{ns}.Eval.Disc1.indicator'] = 53.

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
