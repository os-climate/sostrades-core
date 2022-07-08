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
        sc_name = "DiscAllTypesDoeScenario"
        dspace_dict = {'variable': ['z', 'h'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        values_dict = {}
        #-- set up disciplines in Scenario

        # Doe inputs
        values_dict[f'{ns}.DiscAllTypesDoeScenario.n_samples'] = 100
        # 'lhs', 'CustomDOE', 'fullfact', ...
        values_dict[f'{ns}.DiscAllTypesDoeScenario.algo'] = "lhs"
        values_dict[f'{ns}.DiscAllTypesDoeScenario.design_space'] = dspace

        values_dict[f'{ns}.DiscAllTypesDoeScenario.formulation'] = 'MDF'
        values_dict[f'{ns}.DiscAllTypesDoeScenario.objective_name'] = 'o'
        #disc_dict[f'{ns}.DiscAllTypesDoeScenario.ineq_constraints'] = [f'c_1', f'c_2']
        # disc_dict[f'{ns}.DiscAllTypesDoeScenario.algo_options'] = {'levels':
        # 'None'

        # DiscAllTypes inputs
        h_data = array([0., 0., 0., 0.])
        dict_in_data = {'key0': 0., 'key1': 0.}
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

        values_dict[f'{ns}.{sc_name}.z'] = 1.
        values_dict[f'{ns}.DiscAllTypesDoeScenario.DiscAllTypes.h'] = h_data
        values_dict[f'{ns}.DiscAllTypesDoeScenario.DiscAllTypes.dict_in'] = dict_in_data
        values_dict[f'{ns}.DiscAllTypesDoeScenario.DiscAllTypes.df_in'] = df_in_data
        values_dict[f'{ns}.{sc_name}.weather'] = weather_data
        values_dict[f'{ns}.DiscAllTypesDoeScenario.DiscAllTypes.dict_of_dict_in'] = dict_of_dict_in_data
        values_dict[f'{ns}.DiscAllTypesDoeScenario.DiscAllTypes.dict_of_df_in'] = dict_of_df_in_data

        return [values_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
