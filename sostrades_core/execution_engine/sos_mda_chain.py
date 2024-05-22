'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/02-2024/05/16 Copyright 2023 Capgemini

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
from itertools import repeat
from multiprocessing import cpu_count
from typing import Any
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory

from gemseo import create_mda
from gemseo.utils.derivatives.approximation_modes import ApproximationMode

from gemseo.core.chain import MDOChain
from gemseo.mda.mda_chain import MDAChain
from numpy import floating, ndarray
from pandas import DataFrame

from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline
from sostrades_core.tools.filter.filter import filter_variables_to_convert

N_CPUS = cpu_count()


def get_available_linear_solvers():
    '''Get available linear solvers list
    '''
    lsf = LinearSolversFactory()
    algos = lsf.algorithms
    del lsf

    return algos


class SoSMDAChain(MDAChain):
    """ GEMSEO Overload

    A chain of sub-MDAs.

    The execution sequence is provided by the :class:`.DependencyGraph`.
    """
    # TODO: remove this NUM_DESC_IN (that should be at least in
    # SoSMDODiscipline)
    TYPE = 'type'
    DEFAULT = 'default'
    STRUCTURING = 'structuring'
    POSSIBLE_VALUES = 'possible_values'
    NUMERICAL = 'numerical'
    CACHE_TYPE = 'cache_type'
    CACHE_FILE_PATH = 'cache_file_path'
    DEBUG_MODE = "debug_mode"
    OPTIONAL = "optional"
    AVAILABLE_DEBUG_MODE = ["", "nan", "input_change",
                            "linearize_data_change", "min_max_couplings", "all"]
    RESIDUALS_HISTORY = "residuals_history"
    NUM_DESC_IN = {
        SoSMDODiscipline.LINEARIZATION_MODE: {TYPE: 'string', DEFAULT: 'auto',
                                              # POSSIBLE_VALUES: list(MDODiscipline.AVAILABLE_MODES),
                                              NUMERICAL: True},
        CACHE_TYPE: {TYPE: 'string', DEFAULT: MDOChain.CacheType.NONE,
                     # POSSIBLE_VALUES: [MDOChain.CacheType.NONE, MDODiscipline.SIMPLE_CACHE],
                     # [MDOChain.CacheType.NONE, MDODiscipline.SIMPLE_CACHE, MDODiscipline.HDF5_CACHE, MDODiscipline.MEMORY_FULL_CACHE]
                     NUMERICAL: True,
                     STRUCTURING: True},
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        DEBUG_MODE: {TYPE: 'string', DEFAULT: '', POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
                     NUMERICAL: True, STRUCTURING: True}
    }

    def __init__(self,
                 #             ee,
                 disciplines,  # type: Sequence[MDODiscipline]
                 logger,  # type: logging.Logger
                 reduced_dm=None,  # type: dict
                 inner_mda_name="MDAJacobi",  # type: str
                 max_mda_iter=20,  # type: int
                 name=None,  # type: Optional[str]
                 n_processes=N_CPUS,  # type: int
                 chain_linearize=False,  # type: bool
                 tolerance=1e-6,  # type: float
                 linear_solver_tolerance=1e-12,  # type: float
                 use_lu_fact=False,  # type: bool
                 grammar_type=MDAChain.GrammarType.JSON,  # type: str
                 coupling_structure=None,
                 sub_coupling_structures=None,
                 log_convergence=False,  # type: bool
                 linear_solver="DEFAULT",  # type: str
                 linear_solver_options=None,  # type: Mapping[str,Any]
                 mdachain_parallelize_tasks=False,
                 mdachain_parallel_options=None,
                 initialize_defaults=False,
                 **inner_mda_options,
                 ):
        """
        Args:
            inner_mda_name: The class name of the inner-MDA.
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            chain_linearize: Whether to linearize the chain of execution. Otherwise,
                linearize the overall MDA with base class method. This last option is
                preferred to minimize computations in adjoint mode, while in direct
                mode, linearizing the chain may be cheaper.
            sub_coupling_structures: The coupling structures to be used by the
                inner-MDAs. If ``None``, they are created from the sub-disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks, if
                any.
            mdachain_parallel_options: The options of the MDOParallelChain instances, if
                any.
            initialize_defaults: Whether to create a :class:`.MDOInitializationChain`
                to compute the eventually missing :attr:`.default_inputs` at the first
                execution.
            **inner_mda_options: The options of the inner-MDAs.
        """
        self.logger = logger
        self.is_sos_coupling = True
        # =========================================================================
        # #         self.ee = ee
        # #         self.dm = ee.dm
        # =========================================================================
        # self.authorize_self_coupled_disciplines = authorize_self_coupled_disciplines
        self.reduced_dm = reduced_dm

        super().__init__(disciplines,
                         inner_mda_name=inner_mda_name,
                         max_mda_iter=max_mda_iter,
                         name=name,
                         n_processes=n_processes,
                         chain_linearize=chain_linearize,
                         tolerance=tolerance,
                         linear_solver_tolerance=linear_solver_tolerance,
                         use_lu_fact=use_lu_fact,
                         grammar_type=grammar_type,
                         coupling_structure=coupling_structure,
                         sub_coupling_structures=sub_coupling_structures,
                         log_convergence=log_convergence,
                         linear_solver=linear_solver,
                         linear_solver_options=linear_solver_options,
                         mdachain_parallelize_tasks=mdachain_parallelize_tasks,
                         mdachain_parallel_options=mdachain_parallel_options,
                         initialize_defaults=initialize_defaults,
                         **inner_mda_options
                         )
        # pass the reduced_dm to the data_converter
        self.input_grammar.data_converter.reduced_dm = self.reduced_dm
        self.output_grammar.data_converter.reduced_dm = self.reduced_dm

    def _run(self):
        '''
        Call the _run method of MDAChain in case of SoSCoupling.
        '''
        # set linear solver options for MDA
        self.linear_solver = self.linear_solver_MDA
        self.linear_solver_options = self.linear_solver_options_MDA
        self.linear_solver_tolerance = self.linear_solver_tolerance_MDA

        self.pre_run_mda()

        if len(self.sub_mda_list) > 0:
            self.logger.info(f'{self.name} MDA history')
            self.logger.info('\tIt.\tRes. norm')

        try:
            MDAChain._run(self)
        except Exception as error:
            # Update data manager status (status 'FAILED' is not propagate correctly due to exception
            # so we have to force data manager status update in this case
            self.status = self.ExecutionStatus.FAILED
            self.mdo_chain.status = self.ExecutionStatus.FAILED
            raise error

        # save residual history
        # TODO: to write in data_out after execution
        self.residuals_history = DataFrame(
            {f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.sub_mda_list})

        del self.local_data['MDA residuals norm']
        # TODO: use a method to get the full name
        out = {f'{self.name}.{self.RESIDUALS_HISTORY}': self.residuals_history}
        self.store_local_data(**out)

    # nothing saved in the DM anymore during execution

    #         self.proxy_discipline.store_sos_outputs_values(dict_out, update_dm=True)

    #         # store local data in datamanager
    #         self.proxy_discipline.update_dm_with_local_data(self.local_data)

    def __set_local_data(self, data):
        self._local_data = data

    def pre_run_mda(self):
        '''
        Pre run needed if one of the strong coupling variables is None in a MDA 
        No need of prerun otherwise 
        '''
        strong_couplings = [
            key for key in self.strong_couplings if
            key in self.local_data]  # TODO: replace local_data[key] per key should work
        if len(strong_couplings) < len(self.strong_couplings):
            self.logger.info(
                'Execute a pre-run for the coupling ' + self.name)
            self.recreate_order_for_first_execution()
            self.logger.info(
                'End of pre-run execution for the coupling ' + self.name)

    def recreate_order_for_first_execution(self):
        '''
        For each sub mda defined in the GEMS execution sequence, 
        we run disciplines by disciplines when they are ready to fill all values not initialized in the DM 
        until all disciplines have been run. 
        While loop cannot be an infinite loop because raise an exception
        if no disciplines are ready while some disciplines are missing in the list 
        '''
        for parallel_tasks in self.coupling_structure.sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_disciplines in parallel_tasks:
                # several disciplines coupled
                # first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1:
                    # or (len(coupled_disciplines) == 1
                    # and self.coupling_structure.is_self_coupled(first_disc)
                    ##### DEACTIVATE OPTION authorize_self_coupled_disciplines that was not correctly implemented for strong couplings (flag in GEMSEO to True)
                    # Option that is never used
                    # and self.authorize_self_coupled_disciplines
                    # several disciplines coupled

                    # get the disciplines from self.disciplines
                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for disc in self._disciplines:
                        if disc in coupled_disciplines:
                            sub_mda_disciplines.append(disc)
                    # submda disciplines are not ordered in a correct exec
                    # sequence...
                    # Need to execute ready disciplines one by one until all
                    # sub disciplines have been run
                    while sub_mda_disciplines != []:
                        ready_disciplines = self.get_first_discs_to_execute(
                            sub_mda_disciplines)
                        for discipline in ready_disciplines:
                            # Execute ready disciplines and update local_data
                            if discipline.is_sos_coupling:
                                # recursive call if subdisc is a SoSCoupling
                                # TODO: check if it will work for cases like
                                # Coupling1 > Driver > Coupling2
                                discipline.pre_run_mda()
                                self.local_data.update(discipline.local_data)
                            else:
                                temp_local_data = discipline.execute(
                                    self.local_data)
                                self.local_data.update(temp_local_data)

                        sub_mda_disciplines = [
                            disc for disc in sub_mda_disciplines if disc not in ready_disciplines]
                else:
                    discipline = coupled_disciplines[0]
                    if discipline.is_sos_coupling:
                        # recursive call if subdisc is a SoSCoupling
                        discipline.local_data.update(self.local_data)
                        discipline.pre_run_mda()
                        self.local_data.update(discipline.local_data)
                    else:
                        temp_local_data = discipline.execute(self.local_data)
                        self.local_data.update(temp_local_data)

        self.default_inputs.update(self.local_data)

    def get_first_discs_to_execute(self, disciplines):

        ready_disciplines = []
        disc_vs_keys_none = {}
        for disc in disciplines:
            #             # get inputs values of disc with full_name
            #             inputs_values = disc.get_inputs_by_name(
            #                 in_dict=True, full_name=True)
            # update inputs values with SoSCoupling local_data
            inputs_values = {}
            inputs_values.update(disc._filter_inputs(self.local_data))
            keys_none = [key for key in disc.get_input_data_names()
                         if inputs_values.get(key) is None and not any(
                    [key.endswith(num_key) for num_key in self.NUM_DESC_IN])]
            if keys_none == []:
                ready_disciplines.append(disc)
            else:
                disc_vs_keys_none[disc.name] = keys_none
        if ready_disciplines == []:
            message = '\n'.join(' : '.join([disc, str(keys_none)])
                                for disc, keys_none in disc_vs_keys_none.items())
            raise Exception(
                f'The MDA cannot be pre-runned, some input values are missing to run the MDA \n{message}')
        else:
            return ready_disciplines

    def get_input_data_for_gems(self):
        '''
        Get input_data for linearize ProxyDiscipline
        '''
        input_data = {}
        input_data_names = self.input_grammar.names
        if len(input_data_names) > 0:

            for data_name in input_data_names:
                input_data[data_name] = self.ee.dm.get_value(data_name)

        return input_data

    # -- Protected methods

    # def linearize(self, input_data=None, force_all=False, execute=True):
    #     '''
    #     Overload the linearize of soscoupling to use the one of sosdiscipline and not the one of MDAChain
    #     '''
    #     # LOGGER.info(
    #     # f'Computing the gradient for the MDA : {self.get_disc_full_name()}')
    #     self.logger.info(
    #         f'Computing the gradient for the MDA : {self.name}')
    #
    #     return self._old_discipline_linearize(input_data=input_data,
    #                                           force_all=force_all,
    #                                           execute=execute)

    # def _old_discipline_linearize(self, input_data=None, force_all=False, execute=True,
    #                               exec_before_linearize=True):
    #     """ Temporary call to sostrades linearize that was previously in SoSDiscipline
    #     TODO: see with IRT how we can handle it
    #     """
    #     # set GEM's default_inputs for gradient computation purposes
    #     # to be deleted during GEMS update
    #
    #     result = SoSMDODiscipline.linearize(
    #         self, input_data, force_all, execute)
    #
    #     return result

    def check_jacobian(self, input_data=None, derr_approx=ApproximationMode.FINITE_DIFFERENCES,
                       step=1e-7, threshold=1e-8, linearization_mode='auto',
                       inputs=None, outputs=None, parallel=False,
                       n_processes=MDAChain.N_CPUS,
                       use_threading=False, wait_time_between_fork=0,
                       auto_set_step=False, plot_result=False,
                       file_path="jacobian_errors.pdf",
                       show=False, fig_size_x=10, fig_size_y=10,
                       input_column=None, output_column=None,
                       dump_jac_path=None, load_jac_path=None):
        """
        Overload check jacobian to execute the init_execution
        """
        for disc in self.disciplines:
            disc.sos_wrapp.init_execution()

        indices = SoSMDODiscipline._get_columns_indices(
            self, inputs, outputs, input_column, output_column)

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
            # output_list = []
            # for disc in self.disciplines:
            #     output_list += disc.get_output_data_names(
            #         filtered_outputs=True)
            outputs = self.get_output_data_names(
                filtered_outputs=True, residual_norm_removal=True)  # list(set(output_list))
        if inputs is None:
            # input_list = []
            # for disc in self.disciplines:
            #     input_list += disc.get_input_data_names(filtered_inputs=True)
            inputs = self.get_input_data_names(
                filtered_inputs=True)  # list(set(input_list))
        print('Check jacobian mda_chain : ', linearization_mode)
        return MDAChain.check_jacobian(self,
                                       input_data=input_data,
                                       derr_approx=derr_approx,
                                       step=step,
                                       threshold=threshold,
                                       linearization_mode=linearization_mode,
                                       inputs=inputs,
                                       outputs=outputs,
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
                                       indices=indices)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Overload of the GEMSEO function 
        """
        # set linear solver options for MDO
        self.linear_solver = self.linear_solver_MDO
        self.linear_solver_options = self.linear_solver_options_MDO
        self.linear_solver_tolerance = self.linear_solver_tolerance_MDO

        MDAChain._compute_jacobian(self, inputs, outputs)

        # if self.check_min_max_gradients:
        #     print("IN CHECK of soscoupling")
        #     self._check_min_max_gradients(self.jac)

    def _create_mdo_chain(
            self,
            disciplines,
            inner_mda_name="MDAJacobi",
            sub_coupling_structures=None,
            mdachain_parallelize_tasks=False,
            mdachain_parallel_options=None,
            initialize_defaults=False,
            **inner_mda_options
    ):
        """
        ** Adapted from MDAChain Class in GEMSEO (overload)**

        Create an MDO chain from the execution sequence of the disciplines.

        Args:
            inner_mda_name: The name of the class of the sub-MDAs.
            disciplines: The disciplines.
            sub_coupling_structures: The coupling structures to be used by the sub-MDAs.
                If None, they are created from the sub-disciplines.
            **sub_mda_options: The options to be used to initialize the sub-MDAs.

        disciplines,  # type: Sequence[MDODiscipline]
        inner_mda_name="MDAJacobi",  # type: str
        # type: Optional[Iterable[MDOCouplingStructure]]
        sub_coupling_structures=None,
        **sub_mda_options  # type: Optional[Union[float,int,bool,str]]
        """
        chained_disciplines = []
        self.sub_mda_list = []

        if sub_coupling_structures is None:
            sub_coupling_structures = repeat(None)

        self.__sub_coupling_structures_iterator = iter(sub_coupling_structures)

        for parallel_tasks in self.coupling_structure.sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_disciplines in parallel_tasks:
                first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1:
                    # or (len(coupled_disciplines) == 1
                    # and self.coupling_structure.is_self_coupled(first_disc)
                    # TODO: replace by "and not
                    # isinstance(coupled_disciplines[0], MDA)" as in GEMSEO
                    # actual version
                    # and not coupled_disciplines[0].is_sos_coupling
                    ##### DEACTIVATE OPTION authorize_self_coupled_disciplines that was not correctly implemented for strong couplings (flag in GEMSEO to True)
                    # Option that is never used
                    # #self.authorize_self_coupled_disciplines

                    # several disciplines coupled

                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for disc in disciplines:
                        if disc in coupled_disciplines:
                            sub_mda_disciplines.append(disc)

                    # if activated, all coupled disciplines involved in the MDA
                    # are grouped into a MDOChain (self coupled discipline)
                    #                     if self.get_inputs_by_name("group_mda_disciplines"):
                    #                         sub_mda_disciplines = [MDOChain(sub_mda_disciplines,
                    #                                                         grammar_type=self.grammar_type)]
                    # create a sub-MDA
                    inner_mda_options["linear_solver_tolerance"] = self.linear_solver_tolerance
                    if inner_mda_name not in ['MDAGaussSeidel', 'MDAQuasiNewton']:
                        inner_mda_options["n_processes"] = self.n_processes
                    sub_mda = create_mda(
                        inner_mda_name,
                        sub_mda_disciplines,
                        max_mda_iter=self.max_mda_iter,
                        tolerance=self.tolerance,
                        grammar_type=self.grammar_type,
                        use_lu_fact=self.use_lu_fact,
                        linear_solver=self.linear_solver,
                        linear_solver_options=self.linear_solver_options,
                        coupling_structure=next(
                            self.__sub_coupling_structures_iterator),
                        **inner_mda_options
                    )
                    #                     self.set_epsilon0_and_cache(sub_mda)

                    chained_disciplines.append(sub_mda)
                    self.sub_mda_list.append(sub_mda)
                else:
                    # single discipline
                    chained_disciplines.append(first_disc)

        # TODO: reactivate parallel

        #         if self.get_inputs_by_name("n_subcouplings_parallel") > 1:
        #             chained_disciplines = self._parallelize_chained_disciplines(
        #                 chained_disciplines, self.grammar_type)

        # create the MDO chain that sequentially evaluates the sub-MDAs and the
        # single disciplines
        self.mdo_chain = MDOChain(
            chained_disciplines, name="MDA chain", grammar_type=self.grammar_type
        )

    # =========================================================================
    #     def _parallelize_chained_disciplines(self, disciplines, grammar_type):
    #         ''' replace the "parallelizable" flagged (eg, scenarios) couplings by one parallel chain
    #         with all the scenarios inside
    #         '''
    #         scenarios = []
    #         ind = []
    #         # - get scenario list, if any
    #         for i, disc in enumerate(disciplines):
    #             # check if attribute exists (mainly to avoid gems built-in objects
    #             # like mdas)
    #             if hasattr(disc, 'is_parallel'):
    #                 if disc.is_parallel:
    #                     scenarios.append(disc)
    #                     ind.append(i)
    #         if len(scenarios) > 0:
    #             # - build the parallel chain
    #             n_subcouplings_parallel = self.get_inputs_by_name(
    #                 "n_subcouplings_parallel")
    #             LOGGER.info(
    #                 "Detection of %s parallelized disciplines" % str(len(scenarios)))
    #             par_chain = SoSParallelChain(scenarios, use_threading=False,
    #                                          name="SoSParallelChain",
    #                                          grammar_type=grammar_type,
    #                                          n_processes=n_subcouplings_parallel)
    #             # - remove the scenarios from the disciplines list
    #             disciplines[:] = [d for d in disciplines if d not in scenarios]
    #             # - insert the parallel chain in place of the first scenario
    #             if ind[0] > len(disciplines):
    #                 # all scenarios where at the end of the discipline list
    #                 # we put the parallel chain at the end of the list
    #                 disciplines.append(par_chain)
    #             else:
    #                 # we insert the parallel chain in place of the first scenario
    #                 # found
    #                 disciplines.insert(ind[0], par_chain)
    #
    #         return disciplines
    # =========================================================================

    #  METHODS TO DEBUG MDA CHAIN (NEEDED FOR LINEARIZE)
    # ----------------------------------------------------
    # ----------------------------------------------------

    def _check_nan_in_data(self, data):
        """ Using entry data, check if nan value exist in data's

        :params: data
        :type: composite data

        """
        has_nan = self._check_nan_in_data_rec(data, "")
        if has_nan:
            raise ValueError(f'NaN values found in {self.name}')

    def _check_nan_in_data_rec(self, data, parent_key):
        """
        Using entry data, check if nan value exist in data's as recursive method

        :params: data
        :type: composite data

        :params: parent_key, on composite type (dict), reference parent key
        :type: str

        """
        has_nan = False
        import pandas as pd
        for data_key, data_value in data.items():

            nan_found = False
            if isinstance(data_value, DataFrame):
                if data_value.isnull().any():
                    nan_found = True
            elif isinstance(data_value, ndarray):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(list(data_value)).any():
                    nan_found = True
            elif isinstance(data_value, list):
                # None value in list throw an exception when used with isnan
                if sum(1 for _ in filter(None.__ne__, data_value)) != len(data_value):
                    nan_found = True
                elif pd.isnull(data_value).any():
                    nan_found = True
            elif isinstance(data_value, dict):
                self._check_nan_in_data_rec(
                    data_value, f'{parent_key}/{data_key}')
            elif isinstance(data_value, floating):
                if pd.isnull(data_value).any():
                    nan_found = True

            if nan_found:
                full_key = data_key
                if len(parent_key) > 0:
                    full_key = f'{parent_key}/{data_key}'
                self.logger.debug(f'NaN values found in {full_key}')
                self.logger.debug(data_value)
                has_nan = True
        return has_nan

    def _retrieve_diff_inouts(
            self, compute_all_jacobians: bool = False):
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
            inputs, outputs = SoSMDODiscipline._retrieve_diff_inouts(self)

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
        else:
            return filter_variables_to_convert(self.reduced_dm, self.input_grammar.names,
                                               logger=self.logger)

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
            output_data_names.discard(self.RESIDUALS_NORM)

        # Return filtered or unfiltered output data names
        if filtered_outputs:
            return filter_variables_to_convert(self.reduced_dm, output_data_names)

        return list(output_data_names)
