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


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_disc1_disc3_vs_sMS_from_proc process.
    This process instantiates the multiscenario of (disc1_scenario, disc3_scenario).
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ######### Numerical values   ####
        x1 = 2.
        x2 = 3.
        a1 = 3.
        a2 = 4.
        b = 4.
        b1 = 4.
        b2 = 2.

        #scenario_list = ['scenario_1', 'scenario_2']

        scenario_dict = {'scenario_name': [
            'scenario_1', 'scenario_2'], 'x': [10, 20]}
        scenario_df = pd.DataFrame(data=scenario_dict)

        ######### Fill the dictionary for dm   ####
        dict_values = {}
        #dict_values[f'{self.study_name}.sMS.scenario_list'] = scenario_list
        # dict_values[f'{self.study_name}.sMS.x_dict'] = {
        #    'scenario_1': 10, 'scenario_2': 20}

        dict_values[f'{self.study_name}.sMS.trade_variables'] = {
            'x': 'float'}
        #            'x': 'float', 'a': 'float'}
        #dict_values[f'{self.study_name}.sMS.scenario_dict'] = scenario_dict

        dict_values[f'{self.study_name}.sMS.scenario_df'] = scenario_df

        dict_values[f'{self.study_name}.scenario_1.a'] = a1

        dict_values[f'{self.study_name}.scenario_2.a'] = a2

        dict_values[f'{self.study_name}.sMS.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.sMS.scenario_1.Disc3.constant'] = 3
        dict_values[f'{self.study_name}.sMS.scenario_1.Disc3.power'] = 1
        dict_values[f'{self.study_name}.sMS.scenario_1.Disc3.z'] = 1.2

        dict_values[f'{self.study_name}.sMS.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.sMS.scenario_2.Disc3.constant'] = 2
        dict_values[f'{self.study_name}.sMS.scenario_2.Disc3.power'] = 2
        dict_values[f'{self.study_name}.sMS.scenario_2.Disc3.z'] = 1.2

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.ee.display_treeview_nodes(True)
    uc_cls.run(for_test=True)
