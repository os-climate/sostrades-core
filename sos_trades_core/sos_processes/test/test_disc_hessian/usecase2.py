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
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd


class Study(StudyManager):
    '''This is an example of usecase study for
     the test_disc_hessian process.
    This process instantiates the Hessian Discipline.
    It uses the 1 wrapped discipline : sos_trades_core.sos_wrapping.test_discs.disc_hessian.DiscHessian.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ######### Numerical values   ####
        x = 12.0
        y = 13.0

        ax2 = 14.0
        by2 = 15.0
        cx = 16.0
        dy = 17.0
        exy = 112.0
        ######### Fill the dictionary for dm   ####
        values_dict = {}
        values_dict[f'{self.study_name}.Hessian.x'] = x
        values_dict[f'{self.study_name}.Hessian.y'] = y

        values_dict[f'{self.study_name}.Hessian.ax2'] = ax2
        values_dict[f'{self.study_name}.Hessian.by2'] = by2
        values_dict[f'{self.study_name}.Hessian.cx'] = cx
        values_dict[f'{self.study_name}.Hessian.dy'] = dy
        values_dict[f'{self.study_name}.Hessian.exy'] = exy

        return [values_dict]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
