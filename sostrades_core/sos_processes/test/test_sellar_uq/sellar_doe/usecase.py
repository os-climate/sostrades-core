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

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gemseo.api import configure_logger
from numpy import array
from pandas import DataFrame

from sostrades_core.study_manager.study_manager import StudyManager

if TYPE_CHECKING:
    from logging import Logger


class Study(StudyManager):
    """The main study for sampling the Sellar MDA."""

    COUPLING_NAME = "SellarCoupling"

    SAMPLE_GENERATOR_NAME = "SampleGenerator"

    def __init__(self, log_level="INFO", write_to_file: bool = False, **kwargs) -> None:  # noqa: D107
        logger = self.configure_logger(log_level, write_to_file)
        super().__init__(__file__, logger=logger, **kwargs)

    @staticmethod
    def configure_logger(log_level: str, write_to_file: bool) -> Logger:
        """Configure the logger."""
        if write_to_file:
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"sellar_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            return configure_logger(level=log_level, filename=log_file)
        return configure_logger(level=log_level)

    def setup_usecase(self) -> dict[str, int | float | DataFrame]:
        """Setup the usecase."""
        dspace_dict = {
            "variable": [f"{self.COUPLING_NAME}.{v}" for v in ["x", "z"]],
            "value": [array([1.0]), array([4.0, 3.0])],
            "lower_bnd": [array([0.0]), array([-10.0, 0.0])],
            "upper_bnd": [array([10.0]), array([10.0, 10.0])],
            "enable_variable": [True, True],
            "activated_elem": [[True], [True, True]],
        }
        dspace = DataFrame(dspace_dict)

        # DOE settings
        sampling_inputs = {
            "selected_input": [True, True],
            "full_name": [f"{self.COUPLING_NAME}.{v}" for v in ["x", "z"]],
        }
        sampling_inputs = DataFrame(sampling_inputs)
        sampling_outputs = {
            "selected_output": [False, False, True, False, False],
            "full_name": [f"{self.COUPLING_NAME}.{v}" for v in ["g1", "g2", "f", "y1", "y2"]],
        }
        sampling_outputs = DataFrame(sampling_outputs)
        sampling_params = {
            "sampling_method": "doe_algo",
            "sampling_algo": "fullfact",
            "design_space": dspace,
            "algo_options": {"n_samples": 27},
            "eval_inputs": sampling_inputs,
            "sampling_generation_mode": "at_run_time",
        }
        params = {
            f"{self.study_name}.{self.SAMPLE_GENERATOR_NAME}.{key}": value for key, value in sampling_params.items()
        }

        params[f"{self.study_name}.Eval.with_sample_generator"] = True
        params[f"{self.study_name}.Eval.gather_outputs"] = sampling_outputs

        params[f"{self.study_name}.{self.SAMPLE_GENERATOR_NAME}.design_space"] = dspace

        return [params]


if __name__ == "__main__":
    usecase = Study()
    # usecase.run_usecase = False
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)

    usecase.run(logger_level="INFO")
