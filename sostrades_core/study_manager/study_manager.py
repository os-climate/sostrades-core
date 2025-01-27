'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/20-2024/05/16 Copyright 2023 Capgemini

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
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
from numpy.typing import NDArray

from sostrades_core.sos_processes.script_test_all_usecases import processed_test_one_usecase
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.tools.check_process_gradients.tools import (
    check_each_discpline_jacobians_in_process,
    generate_gradients_tests_for_disciplines_of_usecases,
)

if TYPE_CHECKING:
    from logging import Logger

    from sostrades_core.execution_engine.execution_engine import ExecutionEngine

ValueType = Union[list[Union[float, int]], NDArray, float, int]


class StudyManager(BaseStudyManager):
    """Class that contains additional methods to manage a study."""

    def __init__(
        self,
        file_path: Path | str,
        run_usecase: bool = True,
        execution_engine: ExecutionEngine | None = None,
        dump_directory: str | None = None,
        yield_method: Callable | None = None,
        logger: Logger | None = None,
        test_post_procs: bool = True,
    ) -> None:
        """
        Args:
            file_path: The path to the usecase file.
            run_usecase: Whether the usecase shall be run.
            execution_engine: The usecase ExecutionEngine.
            dump_directory: The directory where to dump the data.
            yield_method: The method to be executed between iterations of the configuration loop.
            logger: The sutdy's logger.
            test_post_procs: Whether to also test the post-processing when testing the study.
        """
        # Get the process folder name
        study_file_path = Path(file_path).resolve()
        study_file_name = study_file_path.stem
        process_name = study_file_path.parent.name

        # Find the module path
        module_path = study_file_path.parents[1]
        module_path_list = []

        # Check if __init__.py exists in the parent directory
        # If yes, it is a module
        # If not, we stop
        while (module_path / "__init__.py").exists():
            module_path_list.append(module_path.name)
            module_path = module_path.parent

        repository_name = ".".join(module_path_list[::-1])

        # init dspace dict
        self.dspace = {}
        self.dspace["dspace_size"] = 0

        super().__init__(
            repository_name,
            process_name,
            study_file_name,
            dump_directory=dump_directory,
            run_usecase=run_usecase,
            yield_method=yield_method,
            logger=logger,
            execution_engine=execution_engine,
            test_post_procs=test_post_procs,
        )

    def update_dspace_dict_with(
        self,
        name: str,
        value: ValueType,
        lower_bnd: ValueType,
        upper_bnd: ValueType,
        activated_elem: list[bool] | None = None,
        enable_variable: bool = True,
    ) -> None:
        """Add a design variable to the design space.

        Args:
            name: The variable's name.
            value: The variable's initial value(s).
            lower_bnd: The variable's lower bound(s).
            upper_bnd: The variable's upper bound(s).
            activated_elem: Whether each component of the variable is activated.
            enable_variable: Whether to enable the whole variable.
        """
        if not isinstance(lower_bnd, (list, np.ndarray)):
            lower_bnd = [lower_bnd] * len(value)
        if not isinstance(upper_bnd, (list, np.ndarray)):
            upper_bnd = [upper_bnd] * len(value)

        if activated_elem is None:
            activated_elem = [True] * len(value)
        self.dspace[name] = {
            "value": value,
            "lower_bnd": lower_bnd,
            "upper_bnd": upper_bnd,
            "enable_variable": enable_variable,
            "activated_elem": activated_elem,
        }

        self.dspace["dspace_size"] += len(value)

    def merge_design_spaces(self, dspace_list: list[dict[str, Any]]) -> None:
        """Update the design space from a list of other design spaces.

        It is necessary to use a set difference here, instead of dictionary update,
        to correctly update the design space size.

        Args:
            dspace_list: The list of design spaces to add.

        Raises:
            ValueError: If some variables are duplicated in several design spaces.
        """
        for dspace in dspace_list:
            dspace_size = dspace.pop("dspace_size")
            duplicated_variables = set(dspace.keys()).intersection(self.dspace.keys())
            if duplicated_variables:
                msg = (
                    "Failed to merge the design spaces; "
                    f"the following variables are present multiple times: {' '.join(duplicated_variables)}"
                )
                raise ValueError(msg)
            self.dspace["dspace_size"] += dspace_size
            self.dspace.update(dspace)


    def setup_usecase_sub_study_list(self) -> None:
        """Instantiate sub-studies and values dictionaries from setup_usecase.

        To be implemented in the sub-classes.
        """
        raise NotImplementedError

    def set_debug_mode(self) -> None:
        """Activate debug mode for the study."""
        self.execution_engine.set_debug_mode()

    def test(self, force_run: bool = False) -> None:
        """Test the usecase.

        Args:
            force_run: Whether to run the usecas with strong MDA couplings and MDO.

        Raises:
            Exception: If the test fails.
        """
        test_passed, error_msg = processed_test_one_usecase(usecase=self.study_full_path, force_run=force_run)
        if not test_passed:
            msg = f"Testing the study resulted in the following exception: {error_msg}"
            raise RuntimeError(msg)

    def test_jacobians_of_each_disc(self):
        check_each_discpline_jacobians_in_process(self.study_full_path)

    def generate_gradients_test_for_disciplines(self, disciplines_names: dict[str: str], path_test_files_to_write: str):
        generate_gradients_tests_for_disciplines_of_usecases(self.study_full_path, disciplines_names, path_test_files_to_write)

