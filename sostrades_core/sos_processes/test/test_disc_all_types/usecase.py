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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from numpy import array
from pandas import DataFrame

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        self.h_data = array([0., 0., 0., 0.])
        self.z_list = [0., 0., 0., 0.]
        self.dict_in_data = {'key0': 0., 'key1': 0.}
        self.df_in_data = DataFrame(array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1],
                                           [0.2, 1.2, 2.2], [-9., -8.7, 1e3]]),
                                    columns=['variable', 'c2', 'c3'])
        self.dict_string_in = {'key_C': '1st string',
                               'key_D': '2nd string'}
        self.list_dict_string_in = [self.dict_string_in, self.dict_string_in]

        self.dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                     'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}
        a_df = DataFrame(array([[5., -.05, 5.e5, 5. ** 5], [2.9, 1., 0., -209.1],
                                [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                         columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': a_df * 3.1416}
        disc_name = 'DiscAllTypes'
        values_dict = {}
        values_dict[f'{self.study_name}.{disc_name}.h'] = self.h_data
        values_dict[f'{self.study_name}.{disc_name}.z_list'] = self.z_list
        values_dict[f'{self.study_name}.{disc_name}.dict_in'] = self.dict_in_data
        values_dict[f'{self.study_name}.{disc_name}.df_in'] = self.df_in_data
        values_dict[f'{self.study_name}.{disc_name}.dict_of_dict_in'] = self.dict_of_dict_in_data
        values_dict[f'{self.study_name}.{disc_name}.dict_of_df_in'] = self.dict_of_df_in_data
        values_dict[f'{self.study_name}.{disc_name}.dict_string_in'] = self.dict_string_in
        values_dict[f'{self.study_name}.{disc_name}.list_dict_string_in'] = self.list_dict_string_in

        return values_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
    ppf = PostProcessingFactory()

    all_post_processings = ppf.get_all_post_processings(uc_cls.execution_engine, False, as_json=False, for_test=False)

    for post_proc_list in all_post_processings.values():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly().show()
