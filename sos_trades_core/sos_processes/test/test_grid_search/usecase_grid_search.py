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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd
import numpy as np


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):

        self.grid_search = 'GridSearch'

        # dspace = pd.DataFrame({'variable': ['GridSearch.Disc1.x', 'GridSearch.Disc1.a'],
        #                        'lower_bnd': [20., 4],
        #                        'upper_bnd': [25., 6],
        #                        'nb_points': [6, 3],
        #                        })
        #
        # eval_inputs = pd.DataFrame({'selected_input': [True, False, True],
        #                             'full_name': [f'{self.grid_search}.Disc1.a', f'{self.grid_search}.Disc1.b', f'{self.grid_search}.Disc1.x']})

        dspace = pd.DataFrame({'variable': ['GridSearch.Disc1.x'],
                               'lower_bnd': [20.],
                               'upper_bnd': [25.],
                               'nb_points': [3],
                               })

        eval_inputs = pd.DataFrame({'selected_input': [True],
                                    'full_name': [f'{self.grid_search}.Disc1.x']})

        eval_outputs = pd.DataFrame({'selected_output': [False, True, False],
                                     'full_name': [f'{self.grid_search}.Disc1.indicator', f'{self.grid_search}.Disc1.y', f'{self.grid_search}.Disc1.y_dict']})

        dict_values = {
            # GRID SEARCH INPUTS
            # f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            # f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,
            # f'{self.study_name}.{self.grid_search}.design_space': dspace,

            # DISC1 INPUTS
            f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
            f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
            f'{self.study_name}.{self.grid_search}.Disc1.d': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.h': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.j': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.g': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.f': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
        }

        return dict_values


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    print("DONE")
