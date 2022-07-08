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


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        x1 = 2
        x2 = 4

        dict_values = {f'{self.study_name}.multi_scenarios.name_1.x_trade': [x1, x2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'name_1.x': 'float'}}
        dict_values[self.study_name +
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']
        dict_values[self.study_name +
                    '.multi_scenarios.z_dict'] = {'scenario_1': 1, 'scenario_2': 2}

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2
            x2b = 5.0

            dict_values[self.study_name + '.name_1.a'] = a1
            dict_values[self.study_name + '.name_2.a'] = a2
            dict_values[self.study_name + '.name_2.x'] = x2b
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_1.b'] = b1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc1.name_2.b'] = b2

        return [dict_values]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
