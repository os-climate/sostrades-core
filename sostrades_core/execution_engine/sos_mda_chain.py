'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/02-2024/07/04 Copyright 2023 Capgemini

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

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, Sequence

import pandas as pd
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.initialization_chain import MDOInitializationChain
from gemseo.core.execution_status import ExecutionStatus
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.utils.constants import N_CPUS, READ_ONLY_EMPTY_DICT
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from numpy import floating, ndarray
from pandas import DataFrame

from sostrades_core.execution_engine.sos_discipline import SoSDiscipline
from sostrades_core.execution_engine.sos_discipline_driver import SoSDisciplineDriver
from sostrades_core.tools.filter.filter import filter_variables_to_convert

if TYPE_CHECKING:
    from logging import Logger

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import StrKeyMapping


def get_available_linear_solvers():
    """Get available linear solvers list."""
    lsf = LinearSolverLibraryFactory()
    algos = lsf.algorithms
    del lsf

    return algos


class SoSMDAChain(MDAChain):
    """GEMSEO Overload.

    A chain of sub-MDAs. The execution sequence is provided by the :class:`.DependencyGraph`.
    """

    # TODO: remove this NUM_DESC_IN (that should be at least in
    # SoSDiscipline)
    TYPE: str = 'type'
    DEFAULT: str = 'default'
    STRUCTURING: str = 'structuring'
    POSSIBLE_VALUES: str = 'possible_values'
    NUMERICAL: str = 'numerical'
    RUN_NEEDED: str = 'run_needed'
    CACHE_TYPE: str = 'cache_type'
    CACHE_FILE_PATH: str = 'cache_file_path'
    DEBUG_MODE: str = "debug_mode"
    OPTIONAL: str = "optional"
    AVAILABLE_DEBUG_MODE: tuple[str] = ("", "nan", "input_change", "min_max_couplings", "all")
    RESIDUALS_HISTORY = "residuals_history"
    NUM_DESC_IN: ClassVar[dict] = {
        SoSDiscipline.LINEARIZATION_MODE: {TYPE: 'string', DEFAULT: 'auto', NUMERICAL: True},
        CACHE_TYPE: {TYPE: 'string', DEFAULT: MDOChain.CacheType.NONE, NUMERICAL: True, STRUCTURING: True},
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        DEBUG_MODE: {
            TYPE: 'string',
            DEFAULT: '',
            POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
            NUMERICAL: True,
            STRUCTURING: True,
            RUN_NEEDED: True,
        },
    }
    NEWTON_ALGO_LIST = ['MDANewtonRaphson', 'MDAGSNewton']

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        logger: Logger,
        reduced_dm: dict = {},
        inner_mda_name: str = "MDAJacobi",
        max_mda_iter: int = 20,
        name: str | None = None,
        chain_linearize: bool = False,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        use_lu_fact: bool = False,
        grammar_type: str = MDAChain.GrammarType.JSON,
        coupling_structure=None,
        log_convergence: bool = True,
        linear_solver: str = "DEFAULT",
        linear_solver_settings: Mapping[str, Any] | None = None,
        mdachain_parallelize_tasks: bool = False,
        mdachain_parallel_settings={},
        initialize_defaults: bool = True,
        scaling_method: MDAChain.ResidualScaling = MDAChain.ResidualScaling.N_COUPLING_VARIABLES,
        **inner_mda_options,
    ) -> None:
        """
        Args:
            inner_mda_name: The class name of the inner-MDA.
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            chain_linearize: Whether to linearize the chain of execution. Otherwise,
                linearize the overall MDA with base class method. This last option is
                preferred to minimize computations in adjoint mode, while in direct
                mode, linearizing the chain may be cheaper.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks, if
                any.
            mdachain_parallel_settings: The options of the MDOParallelChain instances, if
                any.
            initialize_defaults: Whether to create a :class:`.MDOInitializationChain`
                to compute the eventually missing :attr:`.default_input_data` at the first
                execution.
            **inner_mda_options: The options of the inner-MDAs.
        """
        self.logger = logger
        self.is_sos_coupling = True
        self.reduced_dm = reduced_dm

        # tolerance_gs is set after instanciation of the MDA by GEMSEO
        tolerance_gs = inner_mda_options.pop("tolerance_gs", None)
        max_mda_iter_gs = inner_mda_options.pop("max_mda_iter_gs", None)
        # Gauss seidel cannot be launched in parallel by construction (one discipline is launched with the results of the last one)

        mdachain_parallel_settings['n_processes'] = inner_mda_options['n_processes']
        # mdachain_parallel_settings['use_threading'] = inner_mda_options.pop('use_threading', True)
        if inner_mda_name == 'MDAGaussSeidel':
            inner_mda_options.pop('n_processes')
        self.default_grammar_type = grammar_type
        super().__init__(
            disciplines,
            inner_mda_name=inner_mda_name,
            max_mda_iter=max_mda_iter,
            name=name,
            chain_linearize=chain_linearize,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_settings=linear_solver_settings,
            mdachain_parallelize_tasks=mdachain_parallelize_tasks,
            mdachain_parallel_settings=mdachain_parallel_settings,
            initialize_defaults=initialize_defaults,
            **inner_mda_options,
        )
        # pass the reduced_dm to the data_converter
        self.input_grammar.data_converter.reduced_dm = self.reduced_dm
        self.output_grammar.data_converter.reduced_dm = self.reduced_dm
        self.residuals_history = DataFrame({})
        self.scaling = scaling_method
        if inner_mda_name == "MDAGSNewton" and tolerance_gs is not None:
            for mda in self.inner_mdas:
                mda.mda_sequence[0].settings.tolerance = tolerance_gs
                mda.mda_sequence[0].settings.max_mda_iter = max_mda_iter_gs

        if inner_mda_name in self.NEWTON_ALGO_LIST:
            for mda in self.inner_mdas:
                if isinstance(mda, MDANewtonRaphson):
                    mda.settings.newton_linear_solver_name = linear_solver
                    mda.settings.newton_linear_solver_settings = linear_solver_settings

    def clear_jacobian(self):
        return SoSDiscipline.clear_jacobian(self)  # should rather be double inheritance

    def _execute(self):
        """Call the _run method of MDAChain in case of SoSCoupling."""
        # set linear solver options for MDA
        self.linear_solver = self.linear_solver_MDA
        self.linear_solver_settings = self.linear_solver_settings_MDA
        self.linear_solver_tolerance = self.linear_solver_tolerance_MDA

        # self.pre_run_mda()

        if len(self.inner_mdas) > 0:
            self.logger.info("%s MDA history", self.name)
            self.logger.info('\tIt.\tRes. norm')

        try:
            MDAChain._execute(self)
        except Exception as error:
            # Update data manager status (status 'FAILED' is not propagate correctly due to exception
            # so we have to force data manager status update in this case
            self.execution_status.value = ExecutionStatus.Status.FAILED
            self.mdo_chain.execution_status.value = ExecutionStatus.Status.FAILED
            raise

        # save residual history
        # TODO: to write in data_out after execution
        self.residuals_history = DataFrame({f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.inner_mdas})

        # del self.local_data[self.NORMALIZED_RESIDUAL_NORM]
        # TODO: use a method to get the full name
        # out = {f'{self.name}.{self.RESIDUALS_HISTORY}': self.residuals_history}
        # self.io.update_output_data(out)

    def add_differentiated_inputs(self, input_names: Iterable[str] = ()) -> None:  # noqa: D102
        SoSDiscipline.add_differentiated_inputs(self, input_names)

    def add_differentiated_outputs(self, output_names: Iterable[str] = ()) -> None:  # noqa: D102
        SoSDiscipline.add_differentiated_outputs(self, output_names)

    def check_jacobian(
        self,
        input_data=None,
        derr_approx=ApproximationMode.FINITE_DIFFERENCES,
        step=1e-7,
        threshold=1e-8,
        linearization_mode='auto',
        inputs=None,
        outputs=None,
        parallel=False,
        n_processes=N_CPUS,
        use_threading=False,
        wait_time_between_fork=0,
        auto_set_step=False,
        plot_result=False,
        file_path="jacobian_errors.pdf",
        show=False,
        fig_size_x=10,
        fig_size_y=10,
        input_column=None,
        output_column=None,
        dump_jac_path=None,
        load_jac_path=None,
    ):
        """Overload check jacobian to execute the init_execution"""
        for disc in self.disciplines:
            disc.sos_wrapp.init_execution()

        indices = SoSDiscipline._get_columns_indices(self, inputs, outputs, input_column, output_column)

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

        if outputs is None:
            outputs = self.get_output_data_names(
                filtered_outputs=True, residual_norm_removal=True
            )  # list(set(output_list))
        if inputs is None:
            inputs = self.get_input_data_names(filtered_inputs=True)  # list(set(input_list))
        return MDAChain.check_jacobian(
            self,
            input_data=input_data,
            derr_approx=derr_approx,
            step=step,
            threshold=threshold,
            linearization_mode=linearization_mode,
            input_names=inputs,
            output_names=outputs,
            parallel=parallel,
            n_processes=n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            auto_set_step=auto_set_step,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            fig_size_x=fig_size_x,
            fig_size_y=fig_size_y,
            save_reference_jacobian=save_reference_jacobian,
            reference_jacobian_path=reference_jacobian_path,
            indices=indices,
        )

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Overload of the GEMSEO function"""
        # set linear solver options for MDO
        self.linear_solver = self.linear_solver_MDO
        self.linear_solver_settings = self.linear_solver_settings_MDO
        self.linear_solver_tolerance = self.linear_solver_tolerance_MDO
        # use the same linearization_mode for the coupling and its children
        for disc in self.coupling_structure.disciplines:
            disc.linearization_mode = self.linearization_mode

        MDAChain._compute_jacobian(self, inputs, outputs)

    #  METHODS TO DEBUG MDA CHAIN (NEEDED FOR LINEARIZE)
    # ----------------------------------------------------
    # ----------------------------------------------------

    def _check_nan_in_data(self, data):
        """Using entry data, check if nan value exist in data's

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
                self.logger.debug("NaN values found in %s", full_key)
                self.logger.debug(data_value)
                has_nan = True
        return has_nan

    def _retrieve_diff_inouts(self, compute_all_jacobians: bool = False):
        if compute_all_jacobians:
            strong_cpl = set(self.strong_couplings)
            inputs = set(self.get_input_data_names(filtered_inputs=True))
            outputs = self.get_output_data_names(filtered_outputs=True, residual_norm_removal=True)
            # Don't linearize wrt
            inputs -= strong_cpl & inputs
            # Don't do this with output couplings because
            # their derivatives wrt design variables may be needed
            # outputs = outputs - (strong_cpl & outputs)
        else:
            inputs, outputs = SoSDiscipline._retrieve_diff_inouts(self)

        return inputs, outputs

    def get_input_data_names(self, filtered_inputs=False):
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

    def get_output_data_names(self, filtered_outputs: bool = False, residual_norm_removal: bool = False) -> list[str]:
        """
        Retrieve the names of the output variables from the output_grammar.

        Args:
            filtered_outputs (bool): If True, filter variables using the filter_variables_to_convert method.
            residual_norm_removal (bool): If True, remove the residual_norm from output_data_names.

        Returns:
            List[str]: The names of the output variables.
        """
        # Initialize the output data names from the output grammar
        output_data_names = set(self.output_grammar.names)

        # Remove the residual norm if the flag is set and it's present in the output data names
        if residual_norm_removal:
            output_data_names.discard(self.NORMALIZED_RESIDUAL_NORM)

        # Return filtered or unfiltered output data names
        if filtered_outputs:
            return filter_variables_to_convert(self.reduced_dm, output_data_names)

        return list(output_data_names)

    def execute(  # noqa:D102
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> DisciplineData:

        _input_data = input_data

        # The initialization is needed for MDA loops.
        if (
            self.settings.initialize_defaults
            and len(self.coupling_structure.strong_couplings) > 0
        ):

            # Need to pre-run coupling under the coupling if there is some
            driver_sub_disciplines = [disc for disc in self.disciplines if
                                      isinstance(disc, SoSDisciplineDriver)]
            if len(driver_sub_disciplines) != 0:
                input_data = self.pre_run_driver_subcoupling(input_data, driver_sub_disciplines)
                _input_data = input_data
            disc_to_pre_run = [disc for disc in self.disciplines if not isinstance(disc, SoSDisciplineDriver)]
            if len(disc_to_pre_run) > 1:
                _input_data = self.sos_pre_run(input_data, self, disc_to_pre_run)

        return super(MDAChain, self).execute(input_data=_input_data)

    def pre_run_driver_subcoupling(self, input_data, driver_sub_disciplines):
        '''

        Args:
            input_data: the input_data to populate with outputs from the pre-run

        Returns: Input_data populated with pre-run from subcouplings
                The concept of pre-run works if you pre-run low level mdachain first to retrieve strong coupling values for high level ones
                In SoSTrades, the presence of subcoupling is now only present in the case of drivers (COupling under a driver)
                Then you need to check if there are drivers under the mdachain
                If they are then pre-run in the order of the sub disciplines to pre-run the driver sub-coupling first before pre-run the mother coupling

        '''

        for task in self.coupling_structure.sequence:
            for component in task:
                for disc in component:
                    if disc in driver_sub_disciplines:
                        input_data = self.sos_pre_run(input_data, disc._disciplines[0])
                        driver_sub_disciplines.remove(disc)
                    else:
                        new_input_data = disc.execute(input_data)
                        input_data = input_data | new_input_data
                    if len(driver_sub_disciplines) == 0:
                        return input_data
        return input_data

    def sos_pre_run(self, input_data, disc, disciplines=None):
        if disciplines is None:
            disciplines = disc.disciplines
        driver_init_chain = MDOInitializationChain(
            disciplines,
            available_data_names=input_data,
        )
        #     default_input_data = {}
        #     for key, value in init_chain.execute(input_data).items():
        #         if key in self.input_grammar.names:
        #             if key in input_data and size(input_data[key]) != size(value):
        #                 raise ValueError(f"Size mismatch between input data and result of MDA initialization pre-run "
        #                                  f"for variable {key}")
        #             default_input_data[key] = value
        #     self.default_input_data.update(default_input_data)
        #     self.settings.initialize_defaults = False
        # return super(MDAChain, self).execute(input_data=input_data)
        # FIXME: from here below this is a quick-fix for many test errors. In fine the code commented out above should
        # be reactivated and actually solve the size mismatches between input data and MDA pre-run causing the crashes
        pre_run_data = driver_init_chain.execute(input_data)
        self.default_input_data.update({
            key: value
            for key, value in pre_run_data.items()
            if key in self.input_grammar.names
        })

        disc.settings = disc.settings.model_copy(
            update={"initialize_defaults": False})
        return input_data | pre_run_data
