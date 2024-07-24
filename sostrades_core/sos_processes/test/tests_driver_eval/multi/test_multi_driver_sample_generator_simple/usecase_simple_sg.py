'''
Copyright 2023 Capgemini

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

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        # reference var values
        self.x = 2.
        self.a = 3
        self.b = 8
        self.z = 12
        self.constant = 3
        self.power = 2

    def setup_usecase(self):
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.a'] = self.a
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.x'] = self.x
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.Disc3.constant'] = self.constant
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.Disc3.power'] = self.power
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.Disc1.b'] = self.b
        dict_values[self.study_name + '.multi_scenarios.Reference Scenario.z'] = self.z
        return [dict_values]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
