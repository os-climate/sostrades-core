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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from gemseo.mda.sequential_mda import MDASequential
import logging
from copy import deepcopy, copy
from multiprocessing import cpu_count

from pandas import DataFrame
from itertools import repeat
import platform

if platform.system() != 'Windows':
    from sostrades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import PetscKSPAlgos as ksp_lib_petsc

from gemseo.core.chain import MDOChain
# from sostrades_core.execution_engine.parallel_execution.sos_parallel_mdo_chain import SoSParallelChain
from gemseo.mda.mda_chain import MDAChain
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.api import create_mda

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

    def __init__(self,
            ee,
            disciplines,  # type: Sequence[MDODiscipline]
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
            **sub_mda_options  # type: Optional[Union[float, int, bool, str]]
            ):
        ''' Constructor
        '''
        self.is_sos_coupling = True
        self.ee = ee
        self.dm = ee.dm
        
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
            **sub_mda_options  # type: Optional[Union[float, int, bool, str]]
        )

    def _set_dm_cache_map(self):
        '''
        Update cache_map dict in DM with cache, mdo_chain cache, sub_mda_list caches, and its children recursively
        '''
        if self.cache is not None:
            # store SoSCoupling cache in DM
            self._store_cache_with_hashed_uid(self)
            
            # store mdo_chain cache in DM
            self._store_cache_with_hashed_uid(self.mdo_chain)
        
            # store sub mdas cache recursively
            for mda in self.sub_mda_list:
                self._set_sub_mda_dm_cache_map(mda)
            
        # store children cache recursively
        for disc in self.sos_disciplines:
            disc._set_dm_cache_map() 
            
    def _set_sub_mda_dm_cache_map(self, mda):
        '''
        Update cache_map disc in DM with mda cache and its sub_mdas recursively        
        '''
        # store mda cache in DM
        self._store_cache_with_hashed_uid(mda)
        # store sub mda cache recursively
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self._set_sub_mda_dm_cache_map(sub_mda)   

    def _get_numerical_inputs(self):
        '''
        Get numerical parameters input values for MDAChain init
        '''
        # get input for MDAChain instantiation
        needed_numerical_param = ['sub_mda_class', 'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance',
                                  'use_lu_fact', 'warm_start',
                                  'n_processes']
        num_data = self.get_inputs_by_name(
            needed_numerical_param, in_dict=True)

        if num_data['sub_mda_class'] == 'MDAJacobi':
            num_data['acceleration'] = copy(
                self.get_inputs_by_name('acceleration'))
        if num_data['sub_mda_class'] == 'MDAGaussSeidel':
            num_data['warm_start_threshold'] = copy(self.get_inputs_by_name(
                'warm_start_threshold'))
        if num_data['sub_mda_class'] in ['GSNewtonMDA', 'GSPureNewtonMDA', 'GSorNewtonMDA', 'GSPureNewtonorGSMDA']:
            #             num_data['max_mda_iter_gs'] = copy(self.get_inputs_by_name(
            #                 'max_mda_iter_gs'))
            num_data['tolerance_gs'] = copy(self.get_inputs_by_name(
                'tolerance_gs'))
        if num_data['sub_mda_class'] in ['MDANewtonRaphson', 'PureNewtonRaphson', 'GSPureNewtonMDA', 'GSNewtonMDA',
                                         'GSorNewtonMDA', 'GSPureNewtonorGSMDA']:
            num_data['relax_factor'] = copy(
                self.get_inputs_by_name('relax_factor'))

        # linear solver options MDA
        num_data['linear_solver'] = copy(self.get_inputs_by_name(
            'linear_solver_MDA'))
        linear_solver_options_MDA = deepcopy(self.get_inputs_by_name(
            'linear_solver_MDA_options'))

        if num_data['linear_solver'].endswith('_PETSC'):
            # PETSc case
            linear_solver_options_MDA['solver_type'] = num_data['linear_solver'].split('_PETSC')[
                0].lower()
            preconditioner = copy(self.get_inputs_by_name(
                'linear_solver_MDA_preconditioner'))
            linear_solver_options_MDA['preconditioner_type'] = (
                preconditioner != 'None') * preconditioner or None
        else:
            # Scipy case / gmres
            linear_solver_options_MDA['use_ilu_precond'] = (
                copy(self.get_inputs_by_name('linear_solver_MDA_preconditioner')) == 'ilu')

        num_data['linear_solver_tolerance'] = linear_solver_options_MDA.pop(
            'tol')
        num_data['linear_solver_options'] = linear_solver_options_MDA

        self.linear_solver_MDA = num_data['linear_solver']
        self.linear_solver_tolerance_MDA = num_data['linear_solver_tolerance']
        self.linear_solver_options_MDA = deepcopy(
            num_data['linear_solver_options'])

        # linear solver options MDO
        self.linear_solver_MDO = self.get_inputs_by_name('linear_solver_MDO')
        linear_solver_options_MDO = deepcopy(self.get_inputs_by_name(
            'linear_solver_MDO_options'))

        if self.linear_solver_MDO.endswith('_PETSC'):
            linear_solver_options_MDO['solver_type'] = self.linear_solver_MDO.split('_PETSC')[
                0].lower()
            preconditioner = self.get_inputs_by_name(
                'linear_solver_MDO_preconditioner')
            linear_solver_options_MDO['preconditioner_type'] = (
                preconditioner != 'None') * preconditioner or None
        else:
            linear_solver_options_MDO['use_ilu_precond'] = (
                self.get_inputs_by_name('linear_solver_MDO_preconditioner') == 'ilu')

        self.linear_solver_tolerance_MDO = linear_solver_options_MDO.pop('tol')
        self.linear_solver_options_MDO = linear_solver_options_MDO

        return num_data

    def set_epsilon0_and_cache(self, mda):
        '''
        Set epsilon0 that is not argument of the init of the MDA and need to be set outside of it with MDA attributes
        '''
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self.set_epsilon0_and_cache(sub_mda)
        mda.epsilon0 = copy(self.get_inputs_by_name('epsilon0'))
        self.set_cache(mda, self.get_inputs_by_name(
            'cache_type'), self.get_inputs_by_name('cache_file_path'))

    def check_var_data_mismatch(self):
        '''
        Check if a variable data is not coherent between two coupling disciplines

        The check if a variable that is used in input of multiple disciplines is coherent is made in check_inputs of datamanager
        the list of data_to_check is defined in SoSDiscipline
        '''
        
        #TODO: probably better if moved into proxy discipline
        
        if self.logger.level <= logging.DEBUG:
            coupling_vars = self.coupling_structure.graph.get_disciplines_couplings()
            for from_disc, to_disc, c_vars in coupling_vars:
                for var in c_vars:
                    # from disc is in output
                    from_disc_data = from_disc.proxy_discipline.get_data_with_full_name(
                        from_disc.proxy_discipline.IO_TYPE_OUT, var)
                    # to_disc is in input
                    to_disc_data = to_disc.proxy_discipline.get_data_with_full_name(
                        to_disc.proxy_discipline.IO_TYPE_IN, var)
                    for data_name in to_disc.proxy_discipline.DATA_TO_CHECK:
                        # Check if data_names are different
                        if from_disc_data[data_name] != to_disc_data[data_name]:
                            self.logger.debug(
                                f'The {data_name} of the coupling variable {var} is not the same in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}')
                        # Check if unit is not None
                        elif from_disc_data[data_name] is None and data_name == to_disc.proxy_discipline.UNIT:
                            # if unit is None in a dataframe check if there is a
                            # dataframe descriptor with unit in it
                            if from_disc_data[to_disc.proxy_discipline.TYPE] == 'dataframe':
                                # if no dataframe descriptor and no unit warning
                                if from_disc_data[to_disc.proxy_discipline.DATAFRAME_DESCRIPTOR] is None:
                                    self.logger.debug(
                                        f'The unit and the dataframe descriptor of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]} : cannot find unit for this dataframe')
    # TODO : Check the unit in the dataframe descriptor of both data and check if it is ok : Need to add a new value to the df_descriptor tuple check with WALL-E
    #                             else :
    #                                 from_disc_data[self.DATAFRAME_DESCRIPTOR]
                            else:
                                self.logger.debug(
                                    f'The unit of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}')

    def run(self):
        '''
        Call the _run method of MDAChain in case of SoSCoupling.
        '''
        # set linear solver options for MDA
        self.linear_solver = self.linear_solver_MDA
        self.linear_solver_options = self.linear_solver_options_MDA
        self.linear_solver_tolerance = self.linear_solver_tolerance_MDA

        self.pre_run_mda()

        if len(self.sub_mda_list) > 0:
            self.logger.info(f'{self.get_disc_full_name()} MDA history')
            self.logger.info('\tIt.\tRes. norm')

        self._run()

        # save residual history
        dict_out = {}
        residuals_history = DataFrame(
            {f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.sub_mda_list})
        dict_out[self.RESIDUALS_HISTORY] = residuals_history
        self.store_sos_outputs_values(dict_out, update_dm=True)

        # store local data in datamanager
        self.update_dm_with_local_data()

    def update_dm_with_local_data(self, local_data=None):
        '''
        Update the DM with local data from GEMSEO
        '''
        if local_data is None:
            local_data = self.local_data
        self.dm.set_values_from_dict(local_data)

    def pre_run_mda(self):
        '''
        Pre run needed if one of the strong coupling variables is None in a MDA 
        No need of prerun otherwise 
        '''
        strong_couplings_values = [self.dm.get_value(
            key) for key in self.strong_couplings]
        if any(elem is None for elem in strong_couplings_values):
            self.logger.info(
                f'Execute a pre-run for the coupling ' + self.get_disc_full_name())
            self.recreate_order_for_first_execution()
            self.logger.info(
                f'End of pre-run execution for the coupling ' + self.get_disc_full_name())

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
            # get inputs values of disc with full_name
            inputs_values = disc.get_inputs_by_name(
                in_dict=True, full_name=True)
            # update inputs values with SoSCoupling local_data
            inputs_values.update(disc._filter_inputs(self.local_data))
            keys_none = [key for key, value in inputs_values.items()
                         if value is None and not any([key.endswith(num_key) for num_key in self.NUM_DESC_IN])]
            if keys_none == []:
                ready_disciplines.append(disc)
            else:
                disc_vs_keys_none[disc.sos_name] = keys_none
        if ready_disciplines == []:
            message = '\n'.join(' : '.join([disc, str(keys_none)])
                                for disc, keys_none in disc_vs_keys_none.items())
            raise Exception(
                f'The MDA cannot be pre-runned, some input values are missing to run the MDA \n{message}')
        else:
            return ready_disciplines

    # -- Protected methods

    def _run(self):
        ''' Overloads SoSDiscipline run method.
            In SoSCoupling, self.local_data is updated through MDAChain
            and self._data_out is updated through self.local_data.
        '''
        self._old_sos_discipline_run()

        # logging of residuals of the mdas
        # if len(self.sub_mda_list) > 0:
        # self.logger.info(f'{self.get_disc_full_name()} MDA history')
        # for sub_mda in self.sub_mda_list:
        # self.logger.info('\tIt.\tRes. norm')
        # for res_tuple in sub_mda.residual_history:
        # res_norm = '{:e}'.format(res_tuple[0])
        # self.logger.info(f'\t{res_tuple[1]}\t{res_norm}')
        
    def _old_sos_discipline_run(self):
        """ Temporary call to sostrades run that was previously in SoSDiscipline
        TODO: see with IRT how we can handle it
        """
        # Add an exception handler in order to have the capabilities to log
        # the exception before GEMS (when GEMS manage an error it does not propagate it and does
        # not record the stackstrace)
        try:
            # data conversion GEMS > SosStrades
            self._update_type_metadata()

            # execute model
            self._update_status_dm(self.STATUS_RUNNING)

            if self.check_if_input_change_after_run and not self.is_sos_coupling:
                disc_inputs_before_execution = {key: {'value': value} for key, value in deepcopy(
                    self.local_data).items() if key in self.input_grammar.data_names}

            self.run()
            self.fill_output_value_connector()
            if self.check_if_input_change_after_run and not self.is_sos_coupling:
                disc_inputs_after_execution = {key: {'value': value} for key, value in deepcopy(
                    self.local_data).items() if key in self.input_grammar.data_names}
                is_output_error = True
                output_error = self.check_discipline_data_integrity(disc_inputs_before_execution,
                                                                    disc_inputs_after_execution,
                                                                    'Discipline inputs integrity through run',
                                                                    is_output_error=is_output_error)
                if output_error != '':
                    raise ValueError(output_error)

        except Exception as exc:
            self._update_status_dm(self.STATUS_FAILED)
            self.logger.exception(exc)
            raise exc

        # Make a test regarding discipline children status. With GEMS parallel execution, child discipline
        # can failed but FAILED (without an forward exception) status is not
        # correctly propagate upward
        if len(self.sos_disciplines) > 0:
            failed_list = list(filter(
                lambda d: d.status == self.STATUS_FAILED, self.sos_disciplines))

            if len(failed_list) > 0:
                raise SoSDisciplineException(
                    f'An exception occurs during execution in \'{self.name}\' discipline.')

        self._update_status_dm(self.STATUS_DONE)

    def linearize(self, input_data=None, force_all=False, force_no_exec=False):
        '''
        Overload the linearize of soscoupling to use the one of sosdiscipline and not the one of MDAChain
        '''
        self.logger.info(
            f'Computing the gradient for the MDA : {self.get_disc_full_name()}')

        return self._old_discipline_linearize(input_data=input_data, 
                                              force_all=force_all, 
                                              force_no_exec=force_no_exec)
        
    def _old_discipline_linearize(self, input_data=None, force_all=False, force_no_exec=False, exec_before_linearize=True):
        """ Temporary call to sostrades linearize that was previously in SoSDiscipline
        TODO: see with IRT how we can handle it
        """
        # set GEM's default_inputs for gradient computation purposes
        # to be deleted during GEMS update

        if input_data is not None:
            self.default_inputs = input_data
        else:
            self.default_inputs = {}
            input_data = self.get_input_data_for_gems()
            self.default_inputs = input_data

        if self.linearization_mode == self.COMPLEX_STEP:
            # is complex_step, switch type of inputs variables
            # perturbed to complex
            inputs, _ = self._retreive_diff_inouts(force_all)
            def_inputs = self.default_inputs
            for name in inputs:
                def_inputs[name] = def_inputs[name].astype('complex128')
        else:
            pass

        # need execution before the linearize
        if not force_no_exec and exec_before_linearize:
            self.reset_statuses_for_run()
            self.exec_for_lin = True
            self.execute(input_data)
            self.exec_for_lin = False
            force_no_exec = True
            need_execution_after_lin = False

        # need execution but after linearize, in the NR GEMSEO case an
        # execution is done bfore the while loop which udates the local_data of
        # each discipline
        elif not force_no_exec and not exec_before_linearize:
            force_no_exec = True
            need_execution_after_lin = True

        # no need of any execution
        else:
            need_execution_after_lin = False
            # maybe no exec before the first linearize, GEMSEO needs a
            # local_data with inputs and outputs for the jacobian computation
            # if the local_data is empty
            if self.local_data == {}:
                own_data = {
                    k: v for k, v in input_data.items() if self.is_input_existing(k) or self.is_output_existing(k)}
                self.local_data = own_data

        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_before_linearize = {key: {'value': value} for key, value in deepcopy(
                input_data).items() if key in self.input_grammar.data_names}

        # set LINEARIZE status to get inputs from local_data instead of
        # datamanager
        self._update_status_dm(self.STATUS_LINEARIZE)
        result = MDODiscipline.linearize(
            self, input_data, force_all, force_no_exec)
        # reset DONE status
        self._update_status_dm(self.STATUS_DONE)

        self.__check_nan_in_data(result)
        if self.check_linearize_data_changes and not self.is_sos_coupling:
            disc_data_after_linearize = {key: {'value': value} for key, value in deepcopy(
                input_data).items() if key in disc_data_before_linearize.keys()}
            is_output_error = True
            output_error = self.check_discipline_data_integrity(disc_data_before_linearize,
                                                                disc_data_after_linearize,
                                                                'Discipline data integrity through linearize',
                                                                is_output_error=is_output_error)
            if output_error != '':
                raise ValueError(output_error)

        if need_execution_after_lin:
            self.reset_statuses_for_run()
            self.execute(input_data)

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

        indices = SoSDisciplineBuilder._get_columns_indices(
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

        if inputs is None:
            inputs = self.get_input_data_names(filtered_inputs=True)
        if outputs is None:
            outputs = self.get_output_data_names(filtered_outputs=True)

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
                        and not coupled_disciplines[0].is_sos_coupling
                        and self.get_inputs_by_name('authorize_self_coupled_disciplines')
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
                    if self.get_inputs_by_name("group_mda_disciplines"):
                        sub_mda_disciplines = [MDOChain(sub_mda_disciplines,
                                                        grammar_type=self.grammar_type)]
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
                    self.set_epsilon0_and_cache(sub_mda)

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
            self.logger.info(
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

