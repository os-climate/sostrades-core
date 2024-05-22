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
import numpy as np
import pandas as pd

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        coupling_name = "SellarCoupling"

        df = pd.DataFrame({'years': np.arange(1, 5)})
        df['value'] = 1.0

        dict_x = {'years': np.arange(1, 5), 'value': np.ones(4)}
        disc_dict = {}
        # Sellar inputs
        disc_dict[f'{ns}.{coupling_name}.x'] = dict_x
        disc_dict[f'{ns}.{coupling_name}.y_1'] = df
        disc_dict[f'{ns}.{coupling_name}.y_2'] = df
        disc_dict[f'{ns}.{coupling_name}.z'] = np.array([1., 1.])
        disc_dict[f'{ns}.{coupling_name}.Sellar_Problem.local_dv'] = 10.
        disc_dict[f'{ns}.{coupling_name}.max_mda_iter'] = 100
        disc_dict[f'{ns}.{coupling_name}.tolerance'] = 1e-12 

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
