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
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_disc1_disc3_coupling.
    This process instantiates the coupling of (disc1_scenario,disc3_scenario).
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ######### Numerical values   ####
        x = 2.
        a = 3.
        b = 4.
        constant = 3.
        power = 2
        z = 1.2
        ######### Fill the dictionary for dm   ####
        dict_values = {}
        dict_values[self.study_name + '.x'] = x
        dict_values[self.study_name + '.a'] = a
        dict_values[self.study_name + '.Disc1.b'] = b
        dict_values[self.study_name + '.Disc3.constant'] = constant
        dict_values[self.study_name + '.Disc3.power'] = power
        dict_values[self.study_name + '.Disc3.z'] = z

        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
