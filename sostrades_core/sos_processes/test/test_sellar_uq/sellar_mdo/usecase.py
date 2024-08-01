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

from __future__ import annotations

from pandas import DataFrame

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    """The main study of the Sellar problem."""

    COUPLING_NAME = "SellarCoupling"

    SCENARIO_NAME = "SellarOptim"

    def __init__(self) -> dict[str, int | float | DataFrame]:  # noqa: D107
        super().__init__(__file__)

    def setup_usecase(self):
        """Setup the usecase."""
        ns = self.study_name
        dspace_dict = {
            "variable": ["x", "z"],
            "value": [[1.0], [4.0, 3.0]],
            "lower_bnd": [[0.0], [-10.0, 0.0]],
            "upper_bnd": [[10.0], [10.0, 10.0]],
            "enable_variable": [True, True],
            "activated_elem": [[True], [True, True]],
        }
        dspace = DataFrame(dspace_dict)

        # Optim settings
        params = {
            "max_iter": 100,
            "algo": "SLSQP",
            "formulation": "DisciplinaryOpt",
            "objective_name": "f",
            "ineq_constraints": ["g1", "g2"],
            "algo_options": {
                "normalize_design_space": True,
            },
        }
        params = {f"{ns}.{self.SCENARIO_NAME}.{key}": val for key, val in params.items()}

        params[f"{ns}.design_space"] = dspace
        params[f"{ns}.{self.SCENARIO_NAME}.design_space"] = dspace
        return [params]


if __name__ == "__main__":
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)

    uc_cls.run()
