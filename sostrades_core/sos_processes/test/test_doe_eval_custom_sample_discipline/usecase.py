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
from sos_trades_core.study_manager.study_manager import StudyManager
from numpy import array
import pandas as pd
import itertools
import numpy as np


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = self.study_name
        sc_name = 'DoE_Eval'

        input_selection_ABC = {'selected_input': [True, True, True],
                               'full_name': ['stat_A', 'stat_B', 'stat_C']}
        input_selection_ABC = pd.DataFrame(input_selection_ABC)
        output_selection_sum_stat = {'selected_output': [True],
                                     'full_name': ['sum_stat']}
        # Warning : if you select an output that does not exist
        # it will be omitted without explaining that it is because it does not exist (proposing the "possible" outputs)
        # as you can check in the tree view
        output_selection_sum_stat = pd.DataFrame(output_selection_sum_stat)
        my_doe_algo = 'CustomDOE'
        my_dict_of_vec = {}
        my_dict_of_vec['stat_A'] = [2, 8]
        my_dict_of_vec['stat_B'] = [2]
        my_dict_of_vec['stat_C'] = [3, 4, 8]

        my_dict = {}

        my_dict[f'{ns}.Combvec.my_dict_of_vec'] = my_dict_of_vec
        my_dict[f'{ns}.{sc_name}.sampling_algo'] = my_doe_algo

        my_dict[f'{ns}.{sc_name}.eval_inputs'] = input_selection_ABC
        my_dict[f'{ns}.{sc_name}.eval_outputs'] = output_selection_sum_stat

        my_dict[f'{ns}.stat_A'] = 2.
        my_dict[f'{ns}.stat_B'] = 4.
        my_dict[f'{ns}.stat_C'] = 3.

        return [my_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
    # print(uc_cls.execution_engine.dm.get_value(
    #    f'{uc_cls.study_name}.Combvec.my_dict_of_vec'))
    # print(uc_cls.execution_engine.dm.get_value(
    #    f'{uc_cls.study_name}.DoE_Eval.custom_samples_df'))
    # print(uc_cls.execution_engine.dm.get_value(
    #    f'{uc_cls.study_name}.sum_stat_dict'))
