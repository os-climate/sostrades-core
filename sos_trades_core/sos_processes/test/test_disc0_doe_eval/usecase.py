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

from sos_trades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        dspace_dict = {'variable': ['DoE_Eval.Disc0.r'],
                       'lower_bnd': [-5.],
                       'upper_bnd': [+5.],
                       }
        dspace = pd.DataFrame(dspace_dict)

        input_selection = {'selected_input': [True, False],
                           'full_name': ['DoE_Eval.Disc0.r', 'DoE_Eval.Disc0.mod']}
        input_selection = pd.DataFrame(input_selection)

        output_selection = {'selected_output': [True, True],
                            'full_name': ['x', 'a']}
        output_selection = pd.DataFrame(output_selection)

        disc_dict = {}
        # DoE inputs
        n_samples = 4
        disc_dict[f'{ns}.DoE_Eval.sampling_algo'] = "lhs"
        disc_dict[f'{ns}.DoE_Eval.design_space'] = dspace
        disc_dict[f'{ns}.DoE_Eval.algo_options'] = {'n_samples': n_samples}
        disc_dict[f'{ns}.DoE_Eval.eval_inputs'] = input_selection
        disc_dict[f'{ns}.DoE_Eval.eval_outputs'] = output_selection

        # inputs
        disc_dict[f'{ns}.DoE_Eval.Disc0.r'] = 4
        disc_dict[f'{ns}.DoE_Eval.Disc0.mod'] = 2

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes(True)
    uc_cls.run()
