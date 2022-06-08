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
import numpy as np


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        dict_values = {

            # DISC INPUTS
            f'{self.study_name}.Disc1.x': 5.5,
            f'{self.study_name}.Disc1.a': 3,
            f'{self.study_name}.Disc1.b': 2,
            f'{self.study_name}.Disc1.name': 'A1',
            f'{self.study_name}.Disc1.x_dict': {'x_1':1.1,'x_2':2.1,'x_3':5.5,'x_4':9.1},
        }

        return dict_values


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    print("DONE")
