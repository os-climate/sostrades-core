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

from pathlib import Path
from typing import TYPE_CHECKING

from numpy import array
from pandas import read_csv

from sostrades_core.sos_wrapping.analysis_discs.uncertainty_analysis import UncertaintyAnalysis
from sostrades_core.study_manager.study_manager import StudyManager

if TYPE_CHECKING:
    from typing import Any

POST_NAME = "MC post"


class Study(StudyManager):
    """The main study for sampling the Sellar MDA."""

    def __init__(self, **kwargs) -> None:  # noqa: D107
        super().__init__(__file__, **kwargs)

    def setup_usecase(self) -> dict[str, Any]:
        """Setup the usecase."""
        input_samples = read_csv(Path(__file__).parent / "data/input_samples.csv")
        output_samples = read_csv(Path(__file__).parent / "data/output_samples.csv")
        input_dict = {
            f"usecase.Eval_MC.{UncertaintyAnalysis.SoSInputNames.INPUT_SAMPLES}": input_samples,
            f"usecase.Eval_MC.{UncertaintyAnalysis.SoSInputNames.OUTPUT_SAMPLES}": output_samples,
            f"{self.study_name}.{POST_NAME}.{UncertaintyAnalysis.SoSInputNames.PROBABILITY_THRESHOLD}": array([
                5,
                0,
                -19.5,
                45,
                5,
                4,
            ]),
        }
        return [input_dict]


if __name__ == "__main__":
    usecase = Study()
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)

    usecase.run(logger_level="INFO")
