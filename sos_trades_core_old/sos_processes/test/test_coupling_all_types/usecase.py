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
from sos_trades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self):
        super().__init__(__file__)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        
        AC_list = ['A' + str(i) for i in range(0, 100)]
        values_dict = {}
        values_dict[f'{ns}.Disc1.z'] = 1.
        values_dict[f'{ns}.Disc1.x'] = 1.6
        values_dict[f'{ns}.Disc1.a'] = 10
        values_dict[f'{ns}.Disc1.b'] = 2
        values_dict[f'{ns}.Disc1.name'] = 'A1'
        values_dict[f'{ns}.Disc1.x_dict'] = {ac:1.6 for ac in AC_list}
        values_dict[f'{ns}.Disc1.AC_list'] = AC_list
        values_dict[f'{ns}.cache_type'] = 'SimpleCache'
        values_dict[f'{ns}.cache_type'] = 'None'

        return [values_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
