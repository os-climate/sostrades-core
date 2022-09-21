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

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        setup_data_list = []
        # private values AC model
        test_df = pd.DataFrame()
        test_df['a'] = [5]
        test_df['b'] = [2]
        c_dict = {}
        c_dict['c'] = f'{self.study_name}.x + 1'
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.Disc11.test_df': test_df,
            self.study_name + '.Disc11.c_dict': c_dict,
            self.study_name + '.Disc2.constant': 3.1416,
            self.study_name + '.Disc2.power': 2.,
            self.study_name + '.Disc11.test_string': '3+1'}
        setup_data_list.append(private_values)
        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
