'''
Copyright 2022 Airbus SA

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
     the test_disc1_disc3_vs_MS_from_proc process.
    This process instantiates the multiscenario of (disc1_scenario, disc3_scenario).
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ######### Numerical values   ####
        x = 2.
        a = 3
        b = 4
        scenario_list = ['scenario_1', 'scenario_2']
        ######### Fill the dictionary for dm   ####
        dict_values = {}

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list
        dict_values[self.study_name + '.a'] = a
        for scenario in scenario_list:
            dict_values[self.study_name + '.vs_MS.' +
                        scenario + '.Disc1.b'] = b
            dict_values[self.study_name + '.vs_MS.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.vs_MS.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.vs_MS.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.vs_MS.scenario_2.Disc3.z'] = 1.5
        dict_values[self.study_name + '.x'] = x
        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
