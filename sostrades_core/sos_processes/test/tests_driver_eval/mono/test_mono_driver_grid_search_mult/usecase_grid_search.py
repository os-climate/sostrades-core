'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.study_manager.study_manager import StudyManager
import pandas as pd
import numpy as np


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):

        self.evaluator = 'Eval'
        self.sample_generator = 'SampleGenerator'

        # dspace = pd.DataFrame({'variable': ['GridSearch.Disc1.x', 'GridSearch.Disc1.a'],

        dspace = pd.DataFrame({'variable': ['subprocess.Disc1.x'],
                               'lower_bnd': [20.],
                               'upper_bnd': [25.],
                               'nb_points': [3],
                               })

        eval_inputs = pd.DataFrame({'selected_input': [True],
                                    'full_name': ['subprocess.Disc1.x']})

        gather_outputs = pd.DataFrame({'selected_output': [False, True],
                                     'full_name': ['subprocess.Disc1.indicator', 'subprocess.Disc1.y']})

        dict_values = {
            # CASE CONFIG INPUTS
            f'{self.study_name}.{self.sample_generator}.sampling_method': 'grid_search',

            # GRID SEARCH INPUTS
            f'{self.study_name}.{self.evaluator}.with_sample_generator': True,
            f'{self.study_name}.{self.evaluator}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.evaluator}.gather_outputs': gather_outputs,
            f'{self.study_name}.{self.sample_generator}.design_space': dspace,

            # DISC1 INPUTS
            f'{self.study_name}.{self.evaluator}.subprocess.Disc1.name': 'A1',
            f'{self.study_name}.{self.evaluator}.subprocess.Disc1.a': 20,
            f'{self.study_name}.{self.evaluator}.subprocess.Disc1.b': 2,
            f'{self.study_name}.{self.evaluator}.subprocess.Disc1.dd_df': pd.DataFrame({'string_val': ['str', 'str2', 'str3'], 'values1': [100., 200., 300.], 'values2': [50., 100., 150.]})
        }

        return dict_values


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    print("DONE")
