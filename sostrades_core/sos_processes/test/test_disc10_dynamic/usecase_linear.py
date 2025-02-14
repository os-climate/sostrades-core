'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
    '''This is an example of usecase study for
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        # Numerical values   ####
        x = 2.0

        # Fill the dictionary for dm   ####
        values_dict = {}
        values_dict[f'{self.study_name}.Disc10.Model_Type'] = 'Linear'
        values_dict[f'{self.study_name}.Disc10.x'] = x
        # default value a is not provided

        return [values_dict]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes()
    # uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run(for_test=True)
