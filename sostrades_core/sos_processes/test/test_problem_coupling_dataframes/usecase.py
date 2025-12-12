'''
Copyright 2025 Capgemini

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

    def __init__(self, execution_engine=None) -> None:
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        coupling_name = "CostCoupling"

        # Initialize dataframes for car cost computation over years 2025-2030
        manufacturing_cost_df = pd.DataFrame({
            'years': np.arange(2025, 2031),
            'value': [8000.0, 8200.0, 8400.0, 8600.0, 8800.0, 9000.0]  # Increasing manufacturing costs
        })

        maintenance_cost_df = pd.DataFrame({
            'years': np.arange(2025, 2031),
            'value': [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]  # Increasing maintenance costs over time
        })

        # Engine power specifications (different engine sizes for DOE)
        engine_power_dict = {'years': np.arange(1, 5), 'value': np.array([150.0, 200.0, 250.0, 300.0])}  # HP values

        disc_dict = {}
        # Car cost computation inputs
        disc_dict[f'{ns}.{coupling_name}.engine_power'] = engine_power_dict
        disc_dict[f'{ns}.{coupling_name}.manufacturing_cost'] = manufacturing_cost_df
        disc_dict[f'{ns}.{coupling_name}.maintenance_cost'] = maintenance_cost_df
        disc_dict[f'{ns}.{coupling_name}.material_specs'] = np.array([3.0, 2.5])  # [quality_factor, durability_factor]
        disc_dict[f'{ns}.{coupling_name}.cost_problem.weight_factor'] = 1.2  # Vehicle weight factor for DOE
        disc_dict[f'{ns}.{coupling_name}.max_mda_iter'] = 100
        disc_dict[f'{ns}.{coupling_name}.tolerance'] = 1e-12

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
