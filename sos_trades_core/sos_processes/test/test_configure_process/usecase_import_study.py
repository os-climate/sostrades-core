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
from sos_trades_core.sos_processes.test.test_disc1_scenario.usecase import Study as study_disc1
from sos_trades_core import execution_engine


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        x1 = 2
        x2 = 4
        b1 = 3
        b2 = 6

        dict_values = {f'{self.study_name}.multi_scenarios.name_1.x_trade': [x1, x2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'name_1.x': 'float'}}
        dict_values[self.study_name +
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']
        dict_values[self.study_name +
                    '.multi_scenarios.z_dict'] = {'scenario_1': 1, 'scenario_2': 2}

        usecase_disc1 = study_disc1(execution_engine=self.execution_engine)
        usecase_disc1.study_name = f'{self.study_name}.name_1'
        dict_values_name1 = usecase_disc1.setup_usecase()

        usecase_disc1 = study_disc1(execution_engine=self.execution_engine)
        usecase_disc1.study_name = f'{self.study_name}.name_2'
        dict_values_name2 = usecase_disc1.setup_usecase()

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.name_2.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.name_2.b'] = b2

        list_values_dict = [dict_values]
        list_values_dict.extend(dict_values_name1)
        list_values_dict.extend(dict_values_name2)

        return list_values_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
