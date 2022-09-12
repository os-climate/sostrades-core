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
     the test_disc_hessian_vs_MS_from_proc process.
    This process instantiates the multiscenario of a Hessian Discipline.
    '''

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ######### Numerical values   ####
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0
        scenario_list = ['scenario_1', 'scenario_2']
        ######### Fill the dictionary for dm   ####
        dict_values = {}

        dict_values[f'{self.study_name}.vs_MS.scenario_list'] = scenario_list
        for scenario in scenario_list:
            my_root = f'{self.study_name}' + '.vs_MS.' + scenario
            dict_values[f'{my_root}' + '.Hessian.x'] = x
            dict_values[f'{my_root}' + '.Hessian.y'] = y
            dict_values[f'{my_root}' + '.Hessian.ax2'] = ax2
            dict_values[f'{my_root}' + '.Hessian.by2'] = by2
            dict_values[f'{my_root}' + '.Hessian.cx'] = cx
            dict_values[f'{my_root}' + '.Hessian.dy'] = dy
            dict_values[f'{my_root}' + '.Hessian.exy'] = exy
        return [dict_values]


if __name__ == '__main__':
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run(for_test=True)
