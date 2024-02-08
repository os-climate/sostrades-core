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

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Usecase for disc1 disc2 list
        """

        # reference var values
        self.x = 3.
        self.a = 3
        self.constant = 3
        self.power = 2
        self.b = 8
        self.z = 12

        disc_dict = {}
        # build the scenarios
        # configure the Reference scenario
        # Non-trade variables (to propagate)
        disc_dict[f'{self.study_name}.a'] = self.a
        disc_dict[f'{self.study_name}.x'] = self.x
        disc_dict[self.study_name + '.Disc3.constant'] = self.constant
        disc_dict[self.study_name + '.Disc3.power'] = self.power
        disc_dict[self.study_name + '.Disc1.b'] = self.b
        disc_dict[self.study_name + '.z'] = self.z

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
