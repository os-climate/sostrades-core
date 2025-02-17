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

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from pandas import DataFrame

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import MonteCarloDriverWrapper
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.study_manager.study_manager import StudyManager

if TYPE_CHECKING:
    from typing import Any


class Study(StudyManager):
    """The main study for sampling the Sellar MDA."""

    COUPLING_NAME = "SellarCoupling"

    STUDY_NAME = "usecase"

    def __init__(self, **kwargs) -> None:  # noqa: D107
        super().__init__(__file__, **kwargs)

    def setup_usecase(self) -> dict[str, Any]:
        """Setup the usecase."""
        distributions = {
            f"{self.STUDY_NAME}.Eval_MC.x": {
                MonteCarloDriverWrapper.DISTRIBUTION_TYPE_KEY: "OTTriangularDistribution",
                "minimum": 0,
                "maximum": 10,
                "mode": 5,
            }
        }
        selected_outputs = [False, False, True, False, False]
        input_dict = {
            f"{self.STUDY_NAME}.Eval_MC.{ProxyDriverEvaluator.GATHER_OUTPUTS}": DataFrame({
                "selected_output": selected_outputs,
                "full_name": ["c_1", "c_2", "obj", "y_1", "y_2"],
            }),
            f"{self.STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.INPUT_DISTRIBUTIONS}": distributions,
            f"{self.STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.N_SAMPLES}": 10000,
        }
        input_dict.update({f"{self.STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.TARGET_CV}": 0.05})
        input_dict[f"{self.STUDY_NAME}.Eval_MC.x"] = array([1.0])
        input_dict[f"{self.STUDY_NAME}.Eval_MC.y_1"] = array([1.0])
        input_dict[f"{self.STUDY_NAME}.Eval_MC.y_2"] = array([1.0])
        input_dict[f"{self.STUDY_NAME}.Eval_MC.z"] = array([1.0, 1.0])
        input_dict[f"{self.STUDY_NAME}.Eval_MC.subprocess.Sellar_Problem.local_dv"] = 10.0
        return [input_dict]


if __name__ == "__main__":
    usecase = Study()
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)

    usecase.run(logger_level="INFO")
