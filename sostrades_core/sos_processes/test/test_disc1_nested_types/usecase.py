'''
Copyright 2024 Capgemini

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
import pandas as pd

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        df1 = pd.DataFrame({'years': [2020, 2021, 2022],
                            'type': ['alpha', 'beta', 'gamma']})
        df2 = pd.DataFrame({'years': [2020, 2021, 2022],
                            'price': [20.33, 60.55, 72.67]})
        dict_df = {'a': df1.copy(), 'b': df2.copy()}
        dict_values = {
            f'{self.study_name}.Disc1.x': 3,
            f'{self.study_name}.Disc1.a': 1,
            f'{self.study_name}.Disc1.b': 5,
            f'{self.study_name}.Disc1.name': 'A1'
            }
        return dict_values


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
