'''
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
'''
from sostrades_core.tools.filter.filter import filter_variables_to_convert

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from gemseo.mda.sequential_mda import MDASequential
import logging
from copy import deepcopy, copy
from multiprocessing import cpu_count

from pandas import DataFrame
from numpy import ndarray, floating
from itertools import repeat
import platform

if platform.system() != 'Windows':
    from sostrades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import PetscKSPAlgos as ksp_lib_petsc

from gemseo.core.chain import MDOChain
# from sostrades_core.execution_engine.parallel_execution.sos_parallel_mdo_chain import SoSParallelChain
from gemseo.mda.mda_chain import MDAChain
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.api import create_mda
from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline

import logging

N_CPUS = cpu_count()


def get_available_linear_solvers():
    '''Get available linear solvers list
    '''
    lsf = LinearSolversFactory()
    algos = lsf.algorithms
    del lsf

    return algos

LOGGER = logging.getLogger(__name__)

class SoSMDAChain(MDAChain): 
    """ GEMSEO Overload
    
    A chain of sub-MDAs.

    The execution sequence is provided by the :class:`.DependencyGraph`.
    """
    #TODO: remove this NUM_DESC_IN (that should be at least in SoSMDODiscipline)
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
                            "linearize_data_change", "min_max_grad", "min_max_couplings", "all"]
    RESIDUALS_HISTORY = "residuals_history"
    NUM_DESC_IN = {
        'linearization_mode': {TYPE: 'string', DEFAULT: 'auto',  # POSSIBLE_VALUES: list(MDODiscipline.AVAILABLE_MODES),
                               NUMERICAL: True},
        CACHE_TYPE: {TYPE: 'string', DEFAULT: 'None',
                     # POSSIBLE_VALUES: ['None', MDODiscipline.SIMPLE_CACHE],
                     # ['None', MDODiscipline.SIMPLE_CACHE, MDODiscipline.HDF5_CACHE, MDODiscipline.MEMORY_FULL_CACHE]
                     NUMERICAL: True,
                     STRUCTURING: True},
        CACHE_FILE_PATH: {TYPE: 'string', DEFAULT: '', NUMERICAL: True, OPTIONAL: True, STRUCTURING: True},
        DEBUG_MODE: {TYPE: 'string', DEFAULT: '', POSSIBLE_VALUES: list(AVAILABLE_DEBUG_MODE),
                       NUMERICAL: True, STRUCTURING: True}
    }


    def __init__(self,
#             ee,
            disciplines,  # type: Sequence[MDODiscipline]
            reduced_dm=None, # type: dict
            sub_mda_class="MDAJacobi",  # type: str
            max_mda_iter=20,  # type: int
            name=None,  # type: Optional[str]
            n_processes=N_CPUS,  # type: int
            chain_linearize=False,  # type: bool
            tolerance=1e-6,  # type: float
            linear_solver_tolerance=1e-12,  # type: float
            use_lu_fact=False,  # type: bool
            grammar_type=MDAChain.JSON_GRAMMAR_TYPE,  # type: str
            coupling_structure=None,  # type: Optional[MDOCouplingStructure]
            sub_coupling_structures=None,  # type: Optional[Iterable[MDOCouplingStructure]]
            log_convergence=False,  # type: bool
            linear_solver="DEFAULT",  # type: str
            linear_solver_options=None,  # type: Mapping[str,Any]
            authorize_self_coupled_disciplines=False, # type: bool
            ** sub_mda_options  # type: Optional[Union[float, int, bool, str]]
            ):
        ''' Constructor
        '''
        self.is_sos_coupling = True
#         self.ee = ee
#         self.dm = ee.dm
        self.authorize_self_coupled_disciplines = authorize_self_coupled_disciplines
        self.reduced_dm = reduced_dm
        
        MDAChain.__init__(self,
            disciplines,  # type: Sequence[MDODiscipline]
            sub_mda_class=sub_mda_class,  # type: str
            max_mda_iter=max_mda_iter,  # type: int
            name=name,  # type: Optional[str]
            n_processes=n_processes,  # type: int
            chain_linearize=chain_linearize,  # type: bool
            tolerance=tolerance,  # type: float
            linear_solver_tolerance=linear_solver_tolerance,  # type: float
            use_lu_fact=use_lu_fact,  # type: bool
            grammar_type=grammar_type,  # type: str
            coupling_structure=coupling_structure,  # type: Optional[MDOCouplingStructure]
            sub_coupling_structures=sub_coupling_structures,  # type: Optional[Iterable[MDOCouplingStructure]]
            log_convergence=log_convergence,  # type: bool
            linear_solver=linear_solver,  # type: str
            linear_solver_options=linear_solver_options,  # type: Mapping[str,Any]
            ** sub_mda_options  # type: Optional[Union[float, int, bool, str]]
        )
    
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
            LOGGER.info(f'{self.name} MDA history')
            LOGGER.info('\tIt.\tRes. norm')

        MDAChain._run(self)

        # save residual history 
        # TODO: to write in data_out after execution
        self.residuals_history = DataFrame(
            {f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.sub_mda_list})
        
        out = {f'{self.name}.{self.RESIDUALS_HISTORY}' : self.residuals_history} #TODO: use a method to get the full name
        self.store_local_data(**out)

        # nothing saved in the DM anymore during execution

#         self.proxy_discipline.store_sos_outputs_values(dict_out, update_dm=True)

#         # store local data in datamanager
#         self.proxy_discipline.update_dm_with_local_data(self.local_data)


    def pre_run_mda(self):
        '''
        Pre run needed if one of the strong coupling variables is None in a MDA 
        No need of prerun otherwise 
        '''
        strong_couplings = [key for key in self.strong_couplings if key in self.local_data] #TODO: replace local_data[key] per key should work
        if len(strong_couplings) < len(self.strong_couplings):
            LOGGER.info(
                f'Execute a pre-run for the coupling ' + self.name)
            self.recreate_order_for_first_execution()
            LOGGER.info(
                f'End of pre-run execution for the coupling ' + self.name)

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
                first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1 or (
                        len(coupled_disciplines) == 1
                        and self.coupling_structure.is_self_coupled(first_disc)
                        and not coupled_disciplines[0].is_sos_coupling
                ):
                    # several disciplines coupled

                    # get the disciplines from self.disciplines
                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for disc in self.disciplines:
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
                         if inputs_values.get(key) is None and not any([key.endswith(num_key) for num_key in self.NUM_DESC_IN])]
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
        input_data_names = self.input_grammar.get_data_names()
        if len(input_data_names) > 0:

            for data_name in input_data_names:
                input_data[data_name] = self.ee.dm.get_value(data_name)

        return input_data

    # -- Protected methods
        
    def linearize(self, input_data=None, force_all=False, force_no_exec=False):
        '''
        Overload the linearize of soscoupling to use the one of sosdiscipline and not the one of MDAChain
        '''
        # LOGGER.info(
        #     f'Computing the gradient for the MDA : {self.get_disc_full_name()}')
        LOGGER.info(
            f'Computing the gradient for the MDA : {self.name}')

        return self._old_discipline_linearize(input_data=input_data,
                                              force_all=force_all,
                                              force_no_exec=force_no_exec)
        
    def _old_discipline_linearize(self, input_data=None, force_all=False, force_no_exec=False, exec_before_linearize=True):
        """ Temporary call to sostrades linearize that was previously in SoSDiscipline
        TODO: see with IRT how we can handle it
        """
        # set GEM's default_inputs for gradient computation purposes
        # to be deleted during GEMS update

        result = SoSMDODiscipline.linearize(
            self, input_data, force_all, force_no_exec)

        return result

    def check_jacobian(self, input_data=None, derr_approx=MDAChain.FINITE_DIFFERENCES,
                       step=1e-7, threshold=1e-8, linearization_mode='auto',
                       inputs=None, outputs=None, parallel=False,
                       n_processes=MDAChain.N_CPUS,
                       use_threading=False, wait_time_between_fork=0,
                       auto_set_step=False, plot_result=False,
                       file_path="jacobian_errors.pdf",
                       show=False, figsize_x=10, figsize_y=10,
                       input_column=None, output_column=None,
                       dump_jac_path=None, load_jac_path=None):
        """
        Overload check jacobian to execute the init_execution
        """
        for disc in self.sos_disciplines:
            disc.init_execution()

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
            output_list = []
            for disc in self.sos_disciplines:
                output_list += disc.get_output_data_names(filtered_outputs=True)
            outputs = self.get_output_data_names(filtered_outputs=True)#list(set(output_list))
        if inputs is None:
            input_list = []
            for disc in self.sos_disciplines:
                input_list += disc.get_input_data_names(filtered_inputs=True)
            inputs = self.get_input_data_names(filtered_inputs=True)#list(set(input_list))
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
                                       figsize_x=figsize_x,
                                       figsize_y=figsize_y,
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

        if self.check_min_max_gradients:
            print("IN CHECK of soscoupling")
            self._check_min_max_gradients(self, self.jac)

    def _create_mdo_chain(
            self,
            disciplines,
            sub_mda_class="MDAJacobi",
            sub_coupling_structures=None,
            **sub_mda_options
    ):
        """
        ** Adapted from MDAChain Class in GEMSEO (overload)**

        Create an MDO chain from the execution sequence of the disciplines.

        Args:
            sub_mda_class: The name of the class of the sub-MDAs.
            disciplines: The disciplines.
            sub_coupling_structures: The coupling structures to be used by the sub-MDAs.
                If None, they are created from the sub-disciplines.
            **sub_mda_options: The options to be used to initialize the sub-MDAs.

        disciplines,  # type: Sequence[MDODiscipline]
        sub_mda_class="MDAJacobi",  # type: str
        # type: Optional[Iterable[MDOCouplingStructure]]
        sub_coupling_structures=None,
        **sub_mda_options  # type: Optional[Union[float,int,bool,str]]
        """
        chained_disciplines = []
        self.sub_mda_list = []

        if sub_coupling_structures is None:
            sub_coupling_structures = repeat(None)

        sub_coupling_structures_iterator = iter(sub_coupling_structures)
        
        for parallel_tasks in self.coupling_structure.sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_disciplines in parallel_tasks:
                first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1 or (
                        len(coupled_disciplines) == 1
                        and self.coupling_structure.is_self_coupled(first_disc)
                        and not coupled_disciplines[0].is_sos_coupling #TODO: replace by "and not isinstance(coupled_disciplines[0], MDA)" as in GEMSEO actual version
                        and self.authorize_self_coupled_disciplines
                ):
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
                    sub_mda_options["use_lu_fact"] = self.use_lu_fact
                    sub_mda_options["linear_solver_tolerance"] = self.linear_solver_tolerance
                    sub_mda_options["linear_solver"] = self.linear_solver
                    sub_mda_options["linear_solver_options"] = self.linear_solver_options
                    if sub_mda_class not in ['MDAGaussSeidel', 'MDAQuasiNewton']:
                        sub_mda_options["n_processes"] = self.n_processes
                    sub_mda = create_mda(
                        sub_mda_class,
                        sub_mda_disciplines,
                        max_mda_iter=self.max_mda_iter,
                        tolerance=self.tolerance,
                        grammar_type=self.grammar_type,
                        coupling_structure=next(
                            sub_coupling_structures_iterator),
                        **sub_mda_options
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
        
    def _parallelize_chained_disciplines(self, disciplines, grammar_type):
        ''' replace the "parallelizable" flagged (eg, scenarios) couplings by one parallel chain
        with all the scenarios inside
        '''
        scenarios = []
        ind = []
        # - get scenario list, if any
        for i, disc in enumerate(disciplines):
            # check if attribute exists (mainly to avoid gems built-in objects
            # like mdas)
            if hasattr(disc, 'is_parallel'):
                if disc.is_parallel:
                    scenarios.append(disc)
                    ind.append(i)
        if len(scenarios) > 0:
            # - build the parallel chain
            n_subcouplings_parallel = self.get_inputs_by_name(
                "n_subcouplings_parallel")
            LOGGER.info(
                "Detection of %s parallelized disciplines" % str(len(scenarios)))
            par_chain = SoSParallelChain(scenarios, use_threading=False,
                                         name="SoSParallelChain",
                                         grammar_type=grammar_type,
                                         n_processes=n_subcouplings_parallel)
            # - remove the scenarios from the disciplines list
            disciplines[:] = [d for d in disciplines if d not in scenarios]
            # - insert the parallel chain in place of the first scenario
            if ind[0] > len(disciplines):
                # all scenarios where at the end of the discipline list
                # we put the parallel chain at the end of the list
                disciplines.append(par_chain)
            else:
                # we insert the parallel chain in place of the first scenario
                # found
                disciplines.insert(ind[0], par_chain)

        return disciplines

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
                LOGGER.debug(f'NaN values found in {full_key}')
                LOGGER.debug(data_value)
                has_nan = True
        return has_nan

    def get_input_data_names(self, filtered_inputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the input variables from the input_grammar.

        Arguments:
            filtered_inputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the input variables.
        """
        if not filtered_inputs:
            return self.input_grammar.get_data_names()
        else:
            return filter_variables_to_convert(self.reduced_dm, self.input_grammar.get_data_names(),
                                                    logger=LOGGER)

    def get_output_data_names(self, filtered_outputs=False):  # type: (...) -> List[str]
        """
        Retrieve the names of the output variables from the output_grammar

        Arguments:
            filtered_outputs (bool): flag whether to filter variables

        Return:
            List[string] The names of the output variables.
        """
        if not filtered_outputs:
            return self.output_grammar.get_data_names()
        else:
            return filter_variables_to_convert(self.reduced_dm, self.output_grammar.get_data_names())

