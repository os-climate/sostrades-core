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

# debug mode
from collections.abc import Iterable
from copy import deepcopy
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import pandas as pd
from gemseo.core.discipline.discipline import Discipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from numpy import floating, ndarray
from pandas import DataFrame
from scipy.sparse import lil_matrix

from sostrades_core.tools.compare_data_manager_tooling import compare_dict
from sostrades_core.tools.filter.filter import filter_variables_to_convert

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from gemseo.typing import StrKeyMapping

    from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

"""
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
"""


class SoSDisciplineException(Exception):
    pass


class SoSDiscipline(Discipline):
    """
    **SoSDiscipline** is the class that overloads Discipline when using SoSTrades wrapping mode. It handles the
    execution of the user-provided wrapper of the discipline (on the GEMSEO side)

    It is instantiated by the DisciplineWrapp during the prepare_execution step, and it is in one-to-one aggregation
    with the user-defined wrapper (Specialization of SoSWrapp). The _run() method is overloaded by the user-defined wrapper.

    NB: overloading of Discipline has been limited in EEV4 namely wrt EEV3's SoSDiscipline implementation

    Attributes:
        sos_wrapp (SoSWrapp): the user-defined wrapper of the discipline
        reduced_dm (Dict[Dict]): reduced data manager for i/o handling (NB: there is only one reduced_dm per process)
        io_full_name_map (Dict[string]): map from short names to full names of model output variables
    """

    _NEW_ATTR_TO_SERIALIZE = ['reduced_dm', 'sos_wrapp']
    DEBUG_MODE = 'debug_mode'
    LINEARIZATION_MODE = 'linearization_mode'
    RESIDUAL_VARIABLES = 'residual_variables'
    RUN_SOLVE_RESIDUALS = 'run_solves_residuals'

    NUM_DESC_IN = {LINEARIZATION_MODE, 'cache_type', 'cache_file_path', DEBUG_MODE}

    def __init__(
        self,
        full_name: str,
        grammar_type: str,
        cache_type: str,
        sos_wrapp: SoSWrapp,
        reduced_dm: dict,
        logger: logging.Logger,
        debug_mode='',
    ):
        """
        Constructor

        Args:
            full_name (string): full name of the discipline
            grammar_type (string): type of GEMSEO grammar
            cache_type (string): type of cache to be passed to the Discipline
            cache_file_path (string): file path for the cache pickle
            sos_wrapp (SoSWrapp): user-defined wrapper of the discipline
            reduced_dm (Dict[Dict]): reduced version of datamanager for i/o handling
        """
        self.sos_wrapp = sos_wrapp
        self.reduced_dm = reduced_dm
        self.input_full_name_map = None
        self.output_full_name_map = None
        self.logger = logger
        self.debug_mode = debug_mode
        self.default_grammar_type = grammar_type

        super().__init__(name=full_name)
        self.set_cache(self.CacheType(cache_type))

        self.is_sos_coupling = False

        # pass the reduced_dm to the data_converter
        self.input_grammar.data_converter.reduced_dm = self.reduced_dm
        self.output_grammar.data_converter.reduced_dm = self.reduced_dm

    def _run(self, input_data: StrKeyMapping):
        """
        Call user-defined wrapper run.
        """
        # send local data to the wrapper for i/o
        self.sos_wrapp.local_data = input_data
        self.sos_wrapp.input_data_names = self.get_input_data_names()
        self.sos_wrapp.output_data_names = self.get_output_data_names()

        # debug mode: input change
        if self.debug_mode in ['input_change', 'all']:
            disc_inputs_before_execution = {
                key: {'value': value}
                for key, value in deepcopy(input_data).items()
                if key in self.input_grammar
            }

        # SoSWrapp run
        local_data = self.sos_wrapp._run()

        # debug modes
        if self.debug_mode in ['nan', 'all']:
            self._check_nan_in_data(self.io.data)

        if self.debug_mode in ['input_change', 'all']:
            disc_inputs_after_execution = {
                key: {'value': value}
                for key, value in deepcopy(self.io.get_input_data()).items()
                if key in self.input_grammar
            }
            output_error = self.check_discipline_data_integrity(
                disc_inputs_before_execution,
                disc_inputs_after_execution,
                'Discipline inputs integrity through run',
                is_output_error=True,
            )
            if output_error:
                raise ValueError(output_error)

        if self.debug_mode in ['min_max_couplings', 'all']:
            self.display_min_max_couplings()
        return local_data

    def execute(
        self,
        input_data,  # type:Optional[Dict[str, Any]]
    ):  # type: (...) -> Dict[str, Any]
        """
        Overload method in order to catch exception through a try/except
        """
        try:
            self._local_data = super().execute(input_data)
        except Exception as error:
            # Update data manager status (status 'FAILED' is not propagate correctly due to exception
            # so we have to force data manager status update in this case
            self.execution_status.value = self.execution_status.Status.FAILED
            raise
        return self._local_data

    def add_differentiated_inputs(self, input_names: Iterable[str] = ()) -> None:
        """
        Add the inputs against which to differentiate the outputs.

        Filters out the non-numerical (strings, booleans...) inputs before passing the list to GEMSEO.

        Args:
            input_names: The input variables against which to differentiate the outputs.
                If empty, use all the inputs.
        """
        input_names = input_names or self.io.input_grammar.keys()
        filtered_inputs = filter_variables_to_convert(self.reduced_dm, input_names, write_logs=True, logger=self.logger)
        if filtered_inputs:
            Discipline.add_differentiated_inputs(self, filtered_inputs)

    def add_differentiated_outputs(self, output_names: Iterable[str] = ()) -> None:
        """
        Add the outputs to be differentiated.

        Filters out the non-numerical (strings, booleans...) inputs before passing the list to GEMSEO.

        Args:
            output_names: The outputs to be differentiated.
                If empty, use all the outputs.
        """
        output_names = output_names or self.io.output_grammar.keys()
        filtered_outputs = filter_variables_to_convert(
            self.reduced_dm, output_names, write_logs=True, logger=self.logger
        )
        if filtered_outputs:
            Discipline.add_differentiated_outputs(self, filtered_outputs)

    def _prepare_io_for_check_jacobian(
        self, input_names: Iterable[str], output_names: Iterable[str]
    ) -> tuple[Iterable[str], Iterable[str]]:
        """
        Filter the inputs and outputs to keep only the one that can be used for linearization.

        Overrides the method from GEMSEO.

        Args:
            input_names: The names of the inputs to filter.
            output_names: The names of the outputs to filter.

        Returns:
            A tuple containing:
              - the filtered input names.
              - the filtered outputs names.
        """
        input_names = input_names or self.io.input_grammar.keys()
        output_names = output_names or self.io.output_grammar.keys()
        filtered_inputs = filter_variables_to_convert(self.reduced_dm, input_names, write_logs=True, logger=self.logger)
        filtered_outputs = filter_variables_to_convert(
            self.reduced_dm, output_names, write_logs=True, logger=self.logger
        )
        return filtered_inputs, filtered_outputs

    def _retrieve_diff_inouts(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Get the inputs and outputs used in the differentiation of the discipline.

        Args:
            compute_all_jacobians: Whether to compute the Jacobians of all the output
                with respect to all the inputs.
                Otherwise,
                set the input variables against which to differentiate the output ones
                with :meth:`.add_differentiated_inputs`
                and set these output variables to differentiate
                with :meth:`.add_differentiated_outputs`.
        """
        # filtered_inputs and outputs option not in GEMSEO
        if compute_all_jacobians:
            return self.get_input_data_names(filtered_inputs=True), self.get_output_data_names(filtered_outputs=True)

        return self._differentiated_inputs, self._differentiated_outputs

    def check_jacobian(
        self,
        input_data: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
        derr_approx: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-7,
        threshold: float = 1e-8,
        linearization_mode: Discipline.LinearizationMode = Discipline.LinearizationMode.AUTO,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        parallel: bool = False,
        n_processes: int = cpu_count(),
        use_threading: bool = False,
        wait_time_between_fork: float = 0,
        auto_set_step: bool = False,
        plot_result: bool = False,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        figsize_x: float = 10,
        figsize_y: float = 10,
        input_column=None,
        output_column=None,
        dump_jac_path=None,
        load_jac_path=None,
    ):
        """
        Overload check jacobian to execute the init_execution
        """
        # The init execution allows to check jacobian without an execute before the check
        # however if an execute was done, we do not want to restart the model
        # and potentially loose informations to compute gradients (some
        # gradients are computed with the model)
        if self.execution_status.value != self.execution_status.Status.DONE:
            self.sos_wrapp.init_execution()

        # if dump_jac_path is provided, we trigger GEMSEO dump
        if dump_jac_path is not None:
            reference_jacobian_path = dump_jac_path
            save_reference_jacobian = True
        # if dump_jac_path is provided, we trigger GEMSEO dump
        elif load_jac_path is not None:
            reference_jacobian_path = load_jac_path
            save_reference_jacobian = False
        else:
            reference_jacobian_path = None
            save_reference_jacobian = False

        # if inputs is None:
        #     inputs = self.get_input_data_names(filtered_inputs=True)
        # if outputs is None:
        #     outputs = self.get_output_data_names(filtered_outputs=True)
        input_names, output_names = self._prepare_io_for_check_jacobian(inputs, outputs)

        # Differentiate analytically
        self.add_differentiated_inputs(input_names)
        self.add_differentiated_outputs(output_names)
        self.linearization_mode = linearization_mode

        approx = DisciplineJacApprox(
            self,
            derr_approx,
            step,
            parallel,
            n_processes,
            use_threading,
            wait_time_between_fork,
        )

        if auto_set_step:
            approx.auto_set_step(output_names, input_names)

        # Linearize performs execute() if needed
        self.linearize(input_data)

        if input_column is None and output_column is None:
            indices = None
        else:
            indices = self._get_columns_indices(input_names, output_names, input_column, output_column)

        jac_arrays = {
            key_out: {
                key_in: value.toarray() if not isinstance(value, ndarray) else value
                for key_in, value in subdict.items()
            }
            for key_out, subdict in self.jac.items()
        }
        return approx.check_jacobian(
            output_names,
            input_names,
            analytic_jacobian=jac_arrays,
            threshold=threshold,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            fig_size_x=figsize_x,
            fig_size_y=figsize_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Over load of the GEMS function
        Compute the analytic jacobian of a discipline/model
        Check if the jacobian in compute_sos_jacobian is OK

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        if self.jac is None:
            self._init_jacobian(input_names, output_names, init_type=self.InitJacobianType.SPARSE)
        else:
            self._init_jacobian(
                input_names, output_names, init_type=self.InitJacobianType.SPARSE, fill_missing_keys=True
            )

        self.compute_sos_jacobian()
        if not self.sos_wrapp.analytic_jacobian:
            # means that there is no analytic jacobian implemented
            # we set to finite differences and rerun the same method
            self.linearization_mode = ApproximationMode.FINITE_DIFFERENCES
            self.logger.warning(
                f"No compute_sos_jacobian found for the discipline {self.name}, switch to finite difference to compute the jacobian")
            super()._compute_jacobian(input_names, output_names)

    def compute_sos_jacobian(self):
        """
        Overload compute_sos_jacobian of Discipline to call the function in the discipline wrapp
        Then retrieves the 'jac_dict' attribute of the wrapp to update the self.jac
        """
        self.sos_wrapp.compute_sos_jacobian()
        for y_key, x_key_dict in self.sos_wrapp.jac_dict.items():
            for x_key, value in x_key_dict.items():
                self.set_partial_derivative(y_key, x_key, value)
        self.sos_wrapp.jac_dict = {}

    def clear_jacobian(self):
        self.jac = None
        if hasattr(self, 'disciplines') and self.disciplines is not None:
            for discipline in self.disciplines:
                discipline.clear_jacobian()

    def set_partial_derivative(self, y_key, x_key, value):
        """
        Set the derivative of y_key by x_key inside the jacobian of GEMS self.jac
        """
        if y_key in self.jac and x_key in self.jac[y_key]:
            if isinstance(value, ndarray):
                value = lil_matrix(value)
            self.jac[y_key][x_key] = value

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the input variables from the input_grammar.

        Arguments:
            filtered_inputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the input variables.
        """
        if not filtered_inputs:
            return self.input_grammar.names
        return filter_variables_to_convert(self.reduced_dm, self.input_grammar.names, logger=self.logger)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the output variables from the output_grammar

        Arguments:
            filtered_outputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the output variables.
        """
        if not filtered_outputs:
            return self.output_grammar.names
        return filter_variables_to_convert(self.reduced_dm, self.output_grammar.names)

    def _get_columns_indices(self, inputs, outputs, input_column, output_column):
        """
        Returns indices of input_columns and output_columns
        """
        # Get boundaries of the jacobian to compare
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        indices = None
        if input_column is not None or output_column is not None:
            if len(inputs) == 1 and len(outputs) == 1:
                if hasattr(self, '_disciplines') and self._disciplines is not None:
                    for discipline in self._disciplines:
                        self.sos_wrapp.jac_boundaries.update(discipline.jac_boundaries)

                indices = {}
                if output_column is not None:
                    jac_bnd = self.sos_wrapp.jac_boundaries[f'{outputs[0]},{output_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[outputs[0]] = list(range(*tup))

                if input_column is not None:
                    jac_bnd = self.sos_wrapp.jac_boundaries[f'{inputs[0]},{input_column}']
                    tup = [jac_bnd['start'], jac_bnd['end']]
                    indices[inputs[0]] = list(range(*tup))

            else:
                msg = 'Not possible to use input_column and output_column options when \
                    there is more than one input and output'
                raise Exception(msg)

        return indices

    # ----------------------------------------------------
    # ----------------------------------------------------
    #  METHODS TO DEBUG DISCIPLINE
    # ----------------------------------------------------
    # ----------------------------------------------------

    def _check_nan_in_data(self, data):
        """
        Using entry data, check if nan value exist in data's

        :params: data
        :type: composite data

        """
        has_nan = self._check_nan_in_data_rec(data, "")
        if has_nan:
            msg = f'NaN values found in {self.name}'
            raise ValueError(msg)

    def _check_nan_in_data_rec(self, data, parent_key):
        """
        Using entry data, check if nan value exist in data's as recursive method

        :params: data
        :type: composite data

        :params: parent_key, on composite type (dict), reference parent key
        :type: str

        """
        has_nan = False

        for data_key, data_value in data.items():
            nan_found = False
            if isinstance(data_value, DataFrame):
                if data_value.isnull().any():
                    nan_found = True
            elif isinstance(data_value, ndarray):
                # None value in list throw an exception when used with isnan
                if any(x is None for x in data_value) or pd.isnull(list(data_value)).any():
                    nan_found = True
            elif isinstance(data_value, list):
                # None value in list throw an exception when used with isnan
                if any(x is None for x in data_value) or pd.isnull(data_value).any():
                    nan_found = True
            elif isinstance(data_value, dict):
                self._check_nan_in_data_rec(data_value, f'{parent_key}/{data_key}')
            elif isinstance(data_value, floating) and pd.isnull(data_value).any():
                nan_found = True

            if nan_found:
                full_key = data_key
                if len(parent_key) > 0:
                    full_key = f'{parent_key}/{data_key}'
                self.logger.debug(f'NaN values found in {full_key}')
                self.logger.debug(data_value)
                has_nan = True
        return has_nan

    def check_discipline_data_integrity(self, left_dict, right_dict, test_subject, is_output_error=False):
        """
        Compare data is equal in left_dict and right_dict and print a warning otherwise.

        Arguments:
            left_dict (dict): data dict to compare
            right_dict (dict): data dict to compare
            test_subject (string): to identify the executor of the check
            is_output_error (bool): whether to return a dict of errors

        Return:
            output_error (dict): dict with mismatches spotted in comparison
        """
        dict_error = {}
        compare_dict(left_dict, right_dict, '', dict_error)
        output_error = ''
        if dict_error != {}:
            for error in dict_error:
                output_error = '\n'
                output_error += f'Error while test {test_subject} on sos discipline {self.name} :\n'
                output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                output_error += '\n---------------------------------------------------------'

        if is_output_error:
            return output_error
        return None

    def display_min_max_couplings(self):
        """
        Method to display the minimum and maximum values among a discipline's couplings
        """
        min_coupling_dict, max_coupling_dict = {}, {}
        for key, value in self.io.data.items():
            is_coupling = self.reduced_dm[key]['coupling']
            if is_coupling:
                min_coupling_dict[key] = min(abs(value))
                max_coupling_dict[key] = max(abs(value))
        min_coupling = min(min_coupling_dict, key=min_coupling_dict.get)
        max_coupling = max(max_coupling_dict, key=max_coupling_dict.get)
        self.logger.info(
            f"in discipline <{self.name}> : <{min_coupling}> has the minimum coupling value <{min_coupling_dict[min_coupling]}>"
        )
        self.logger.info(
            f"in discipline <{self.name}> : <{max_coupling}> has the maximum coupling value <{max_coupling_dict[max_coupling]}>"
        )
