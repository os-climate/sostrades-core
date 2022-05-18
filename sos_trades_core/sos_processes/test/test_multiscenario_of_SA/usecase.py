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
    '''This is an example of usecase study for
     the test_multiscenario_of_SA specify process.
    This process is an example of a multiscenario of Sensitivity Analysis.
    It uses the 2 wrapped disciplines : disc1_scenario.Disc1
     (orchestrated by the test_disc1_scenario process) and disc3_scenario.Disc3.
    '''
    def __init__(self, execution_engine=None, run_usecase=False):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        x_1 = 2
        x_2_a = 4

        dict_values = {f'{self.study_name}.multi_scenarios.name_1.x_trade': [x_1, x_2_a],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'name_1.x': 'float'}}
        dict_values[self.study_name +
                    '.multi_scenarios.name_list'] = ['name_1', 'name_2']

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            a_1 = 3
            b_1 = 4
            a_2 = 6
            b_2 = 2
            x_2_b = 5.0

            dict_values[self.study_name + '.name_1.a'] = a_1
            dict_values[self.study_name + '.name_2.a'] = a_2
            dict_values[self.study_name + '.name_2.x'] = x_2_b
            # If compared to process test_multiscenario_of_SA bellow we will have:
            #     .Disc1 becomes .SA.Disc1 and .Disc3 becomes .SA.Disc3
            #             for local variables (b,constant, power local)
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc1.name_1.b'] = b_1
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc1.name_2.b'] = b_2
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.SA.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
        # Begin : added SA step as regard to the standard multiscenarion process test_multiscenario
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.SA.eval_inputs'] = ['z']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.SA.eval_inputs'] = ['z']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.SA.eval_outputs'] = ['o']
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.SA.eval_outputs'] = ['o']
        # End : added SA step as regard to the standard multiscenarion process test_multiscenario

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
