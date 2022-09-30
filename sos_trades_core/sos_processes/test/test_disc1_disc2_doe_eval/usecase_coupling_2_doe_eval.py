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

# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):

        ns = f'{self.study_name}'
        dspace_dict = {'variable': ['x', 'a'],

                       'lower_bnd': [0., 50.],
                       'upper_bnd': [100., 200.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        input_selection_x_a = {'selected_input': [True, True],
                               'full_name': ['x', 'DoEEval.Disc1.a']}
        input_selection_x_a = pd.DataFrame(input_selection_x_a)

        output_selection_z_z = {'selected_output': [True, True, False],
                                'full_name': ['z', 'DoEEval.Disc1.z', 'DoEEval.Disc1.indicator']}
        output_selection_z_z = pd.DataFrame(output_selection_z_z)

        # private values AC model
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.DoEEval.Disc1.a': 5.,
            self.study_name + '.DoEEval.Disc1.b': 25431.,
            self.study_name + '.y': 4.,
            self.study_name + '.DoEEval.Disc2.constant': 3.1416,
            self.study_name + '.DoEEval.Disc2.power': 2}

        # DoE inputs
        disc_dict = {}
        n_samples = 100
        private_values[f'{ns}.DoEEval.sampling_algo'] = "fullfact"
        private_values[f'{ns}.DoEEval.design_space'] = dspace
        private_values[f'{ns}.DoEEval.algo_options'] = {'n_samples': n_samples}
        private_values[f'{ns}.DoEEval.eval_inputs'] = input_selection_x_a
        private_values[f'{ns}.DoEEval.eval_outputs'] = output_selection_z_z

        return [private_values]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
