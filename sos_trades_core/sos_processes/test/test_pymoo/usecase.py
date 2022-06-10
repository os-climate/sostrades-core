"""
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
"""
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
from pandas import DataFrame

from gemseo.algos.design_space import DesignSpace
from sos_trades_core.study_manager.study_manager import StudyManager
from numpy import array, ones, inf


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        scenario_name = "KnapsackOptimScenario"
        discipline_name = "knapsack_problem"

        # Knapsack characteristics.
        values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
        weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
        items_capacity = 5
        weight_capacity = inf
        n_items = len(values)

        # 10 item available and a binary knapsack.
        design_space_dict = {
            "variable": ["x"],
            "value": [[1.] * n_items],
            "lower_bnd": [[0.] * n_items],
            "upper_bnd": [[1.] * n_items],
            "enable_variable": [True],
            "activated_elem": [[True] * n_items],
            "variable_type": [[DesignSpace.INTEGER] * n_items]
        }

        integer_operators = {
            "sampling": "int_lhs", 
            "crossover": "int_sbx",
            "mutation": ["int_pm", dict(prob=1.0, eta=3.0)],
        }
        integer_options = {"normalize_design_space": False, "stop_crit_n_x": 99}
        algo_options = {
            **integer_options,
            **integer_operators,
            "max_gen": 20,
        }

        disc_dict = {
            f"{self.study_name}.{scenario_name}.design_space": DataFrame(design_space_dict),
            f"{self.study_name}.{scenario_name}.formulation": "MDF",
            f"{self.study_name}.{scenario_name}.objective_name": "value",
            f"{self.study_name}.{scenario_name}.maximize_objective": True,
            f"{self.study_name}.{scenario_name}.ineq_constraints": ["excess_items"],
            f"{self.study_name}.{scenario_name}.algo": "PYMOO_NSGA2",
            f"{self.study_name}.{scenario_name}.max_iter": 800,
            f"{self.study_name}.{scenario_name}.algo_options": algo_options,
            f"{self.study_name}.{scenario_name}.{discipline_name}.x": ones(n_items, dtype=int),
            f"{self.study_name}.{scenario_name}.{discipline_name}.items_value": values,
            f"{self.study_name}.{scenario_name}.{discipline_name}.items_weight": weights,
            f"{self.study_name}.{scenario_name}.{discipline_name}.items_capacity": items_capacity,
            f"{self.study_name}.{scenario_name}.{discipline_name}.weight_capacity": weight_capacity,
        }

        return [disc_dict]


if "__main__" == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
