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


class Study(StudyManager):

    def __init__(self):
        super().__init__(__file__)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "DoEEval"

        dspace_dict_eval = {'variable': ['z', 'h'],
                            'lower_bnd': [0., [-10., 0.]],
                            'upper_bnd': [10., [10., 10.]],
                            'enable_variable': [True, True],
                            'activated_elem': [[True], [True, True]]}
        dspace_eval = pd.DataFrame(dspace_dict_eval)
        dspace_eval

        input_selection_z_h = {'selected_input': [True, True, False, False, False, False, False],
                               'full_name': [f'{sc_name}.z',
                                             f'{sc_name}.DiscAllTypes.h',
                                             f'{sc_name}.DiscAllTypes.dict_in',
                                             f'{sc_name}.DiscAllTypes.df_in',
                                             f'{sc_name}.weather',
                                             f'{sc_name}.DiscAllTypes.dict_of_dict_in',
                                             f'{sc_name}.DiscAllTypes.dict_of_df_in',
                                             ]}
        input_selection_z_h = pd.DataFrame(input_selection_z_h)
        input_selection_z_h

        output_selection_o_df__out_dict__out = {'selected_output': [True, True, True, False],
                                                'full_name': [f'{sc_name}.df_out',
                                                              f'{sc_name}.o',
                                                              f'{sc_name}.dict_out',
                                                              f'{sc_name}.residuals_history']}
        output_selection_o_df__out_dict__out = pd.DataFrame(
            output_selection_o_df__out_dict__out)

        disc_dict = {}
        #-- set up disciplines in Scenario

        # Doe inputs
        disc_dict = {}
        # Doe inputs
        # 'lhs', 'fullfact', ...
        my_doe_algo = "fullfact"
        n_samples = 100
        disc_dict[f'{ns}.{sc_name}.sampling_algo'] = my_doe_algo
        disc_dict[f'{ns}.{sc_name}.design_space'] = dspace_eval
        disc_dict[f'{ns}.{sc_name}.algo_options'] = {
            'n_samples': n_samples, 'fake_option': 'fake_option'}
        disc_dict[f'{ns}.{sc_name}.eval_inputs'] = input_selection_z_h
        disc_dict[f'{ns}.{sc_name}.eval_outputs'] = output_selection_o_df__out_dict__out

        # DiscAllTypes inputs
        h_data = array([0., 0., 0., 0.])
        dict_in_data = {'key0': 0., 'key1': 0.}
        dict_string_in_data = {'key0': 'key0', 'key1': 'key1'}
        df_in_data = pd.DataFrame(array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1],
                                         [0.2, 1.2, 2.2], [-9., -8.7, 1e3]]),
                                  columns=['variable', 'c2', 'c3'])
        weather_data = 'cloudy, it is Toulouse ...'
        dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}

        a_df = pd.DataFrame(array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                   [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        dict_of_df_in_data = {'key_C': a_df,
                              'key_D': a_df * 3.1416}
        disc_dict[f'{ns}.{sc_name}.z'] = 1.
        disc_dict[f'{ns}.{sc_name}.z_list'] = [1., 2., 3.]
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.h'] = h_data
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.dict_in'] = dict_in_data
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.df_in'] = df_in_data
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.dict_string_in'] = dict_string_in_data
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.list_dict_string_in'] = [
            dict_string_in_data, dict_string_in_data]
        disc_dict[f'{ns}.{sc_name}.weather'] = weather_data
        disc_dict[f'{ns}.{sc_name}.weather_list'] = [weather_data]
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.dict_of_dict_in'] = dict_of_dict_in_data
        disc_dict[f'{ns}.{sc_name}.DiscAllTypes.dict_of_df_in'] = dict_of_df_in_data

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
