'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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
from numpy import array

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        values_dict = {}
        # build the scenarios
        scenario_df_outer = pd.DataFrame({'selected_scenario': [True, False, True],
                                          'scenario_name': ['scenario_1',
                                                            'scenario_W',
                                                            'scenario_2']})
        scenario_df_inner = pd.DataFrame({'selected_scenario': [True, True],
                                          'scenario_name': ['name_1',
                                                            'name_2']})
        values_dict[f'{self.study_name}.outer_ms.samples_df'] = scenario_df_outer

        self.constant = [1, 2]
        self.power = [1, 2]
        self.z = [1, 2]
        self.a = [0, 10]
        self.x = [0, 10]
        self.b = [[1, 2], [3, 4]]

        # configure the scenarios
        scenario_list_outer = ['scenario_1', 'scenario_2']
        scenario_list_inner = ['name_1', 'name_2']


        for i, sc in enumerate(scenario_list_outer):
            values_dict[self.study_name + '.outer_ms.'+sc+'.inner_ms.samples_df'] = scenario_df_inner
            values_dict[self.study_name + '.outer_ms.'+sc+'.Disc3.constant'] = self.constant[i]
            values_dict[self.study_name + '.outer_ms.'+sc+'.Disc3.power'] = self.power[i]
            values_dict[self.study_name + '.outer_ms.'+sc+'.z'] = self.z[i]
            for j, name in enumerate(scenario_list_inner):
                values_dict[self.study_name + '.outer_ms.'+sc+'.inner_ms.'+name+'.Disc1.b'] = self.b[i][j]
                values_dict[self.study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.a'] = self.a[i]
                values_dict[self.study_name + '.outer_ms.' + sc + '.inner_ms.' + name + '.x'] = self.x[i]

        return [values_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
