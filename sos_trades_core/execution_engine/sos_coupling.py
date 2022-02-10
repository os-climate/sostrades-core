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
from gemseo.mda.sequential_mda import MDASequential
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import deepcopy, copy
from multiprocessing import cpu_count

from pandas import DataFrame
from itertools import repeat
import platform

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline_builder import SoSDisciplineBuilder
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.ns_manager import NS_SEP

if platform.system() != 'Windows':
    from gemseo_petsc.linear_solvers.ksp_lib import PetscKSPAlgos  # pylint: disable-msg=E0401
    from sos_trades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import PetscKSPAlgos as ksp_lib_petsc

from gemseo.core.chain import MDOChain
from sos_trades_core.execution_engine.parallel_execution.sos_parallel_mdo_chain import SoSParallelChain
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.mda import MDA
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


class SoSCoupling(SoSDisciplineBuilder, MDAChain):
    ''' Class that computes a chain of SoSDisciplines
    '''
    SECANT_ACCELERATION = "secant"
    M2D_ACCELERATION = "m2d"
    RESIDUALS_HISTORY = "residuals_history"

    # get list of available linear solvers from LinearSolversFactory
    AVAILABLE_LINEAR_SOLVERS = get_available_linear_solvers()

    # set default value of linear solver according to the operatinh system
    if platform.system() == 'Windows':
        DEFAULT_LINEAR_SOLVER = 'GMRES'
        DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'None'
        POSSIBLE_VALUES_PRECONDITIONER = ['None', 'ilu']
    else:
        DEFAULT_LINEAR_SOLVER = 'GMRES_PETSC'
        DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'gasm'
        POSSIBLE_VALUES_PRECONDITIONER = [
            'None'] + ksp_lib_petsc.AVAILABLE_PRECONDITIONER

    DEFAULT_LINEAR_SOLVER_OPTIONS = {
        'max_iter': 1000,
        'tol': 1.0e-8}

    DESC_IN = {
        # NUMERICAL PARAMETERS
        'sub_mda_class': {SoSDiscipline.TYPE: 'string', SoSDiscipline.POSSIBLE_VALUES: ['MDAJacobi', 'MDAGaussSeidel', 'MDANewtonRaphson', 'PureNewtonRaphson', 'MDAQuasiNewton', 'GSNewtonMDA',  'GSPureNewtonMDA', 'GSorNewtonMDA', 'MDASequential'], SoSDiscipline.DEFAULT: 'MDAJacobi', SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'max_mda_iter': {SoSDiscipline.TYPE: 'int', SoSDiscipline.DEFAULT: 30, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'n_processes': {SoSDiscipline.TYPE: 'int', SoSDiscipline.DEFAULT: 1, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'chain_linearize': {SoSDiscipline.TYPE: 'bool', SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.DEFAULT: False, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'tolerance': {SoSDiscipline.TYPE: 'float', SoSDiscipline.DEFAULT: 1.e-6, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'use_lu_fact': {SoSDiscipline.TYPE: 'bool', SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.DEFAULT: False, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'warm_start': {SoSDiscipline.TYPE: 'bool', SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.DEFAULT: False, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'acceleration': {SoSDiscipline.TYPE: 'string', SoSDiscipline.POSSIBLE_VALUES: [M2D_ACCELERATION, SECANT_ACCELERATION, 'none'], SoSDiscipline.DEFAULT: M2D_ACCELERATION, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'warm_start_threshold': {SoSDiscipline.TYPE: 'float', SoSDiscipline.DEFAULT: -1, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        # parallel sub couplings execution
        'n_subcouplings_parallel': {SoSDiscipline.TYPE: 'int', SoSDiscipline.DEFAULT: 1, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        #'max_mda_iter_gs': {SoSDiscipline.TYPE: 'int', SoSDiscipline.DEFAULT: 5, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'tolerance_gs': {SoSDiscipline.TYPE: 'float', SoSDiscipline.DEFAULT: 10.0, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'relax_factor': {SoSDiscipline.TYPE: 'float', SoSDiscipline.RANGE: [0.0, 1.0], SoSDiscipline.DEFAULT: 0.99, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        # NUMERICAL PARAMETERS OUT OF INIT
        'epsilon0': {SoSDiscipline.TYPE: 'float', SoSDiscipline.DEFAULT: 1.0e-6, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        # Linear solver for MD0
        'linear_solver_MDO': {SoSDiscipline.TYPE: 'string', SoSDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS, SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'linear_solver_MDO_preconditioner': {SoSDiscipline.TYPE: 'string', SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER, SoSDiscipline.POSSIBLE_VALUES: POSSIBLE_VALUES_PRECONDITIONER, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'linear_solver_MDO_options': {SoSDiscipline.TYPE: 'dict', SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        # Linear solver for MDA
        'linear_solver_MDA': {SoSDiscipline.TYPE: 'string',  SoSDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS, SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'linear_solver_MDA_preconditioner': {SoSDiscipline.TYPE: 'string', SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER,  SoSDiscipline.POSSIBLE_VALUES: POSSIBLE_VALUES_PRECONDITIONER, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        'linear_solver_MDA_options': {SoSDiscipline.TYPE: 'dict', SoSDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True},
        # group all disciplines in a MDOChain
        'group_mda_disciplines': {SoSDiscipline.TYPE: 'bool', SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.DEFAULT: False, SoSDiscipline.USER_LEVEL: 3, SoSDiscipline.NUMERICAL: True, SoSDiscipline.STRUCTURING: True}
    }

    DESC_OUT = {}

    eps0 = 1.0e-6
    has_chart = False

    def __init__(self, sos_name, ee, cls_builder=None, with_data_io=False):
        ''' Constructor
        '''
        if cls_builder is None:
            cls_builder = [] = []
        self.cls_builder = cls_builder
        self._reload(sos_name, ee)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Coupling')

        self.with_data_io = with_data_io
        self.residuals_dict = {}

        self.linear_solver_MDA = None
        self.linear_solver_options_MDA = None
        self.linear_solver_tolerance_MDA = None

        self.linear_solver_MDO = None
        self.linear_solver_options_MDO = None
        self.linear_solver_tolerance_MDO = None

        self._set_dm_disc_info()

        # - Disable the possibility to build self-coupled disciplines (MDA with 1 subdisc)
        self.authorize_self_coupled_disciplines = False

    def _reload(self, sos_name, ee):
        ''' reload object
        '''
        self.is_sos_coupling = True
        SoSDiscipline._reload(self, sos_name, ee)

    def build(self):

        old_current_discipline = self.ee.factory.current_discipline
        self.ee.factory.current_discipline = self
        for builder in self.cls_builder:
            disc = builder.build()
            if disc not in self.sos_disciplines:
                self.ee.factory.add_discipline(disc)
        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
        # level coupling and we do not have to restore the current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    #     def clear_cache(self):
    #         self.mdo_chain.cache.clear()
    #         SoSDisciplineBuilder.clear_cache(self)

    # -- Public methods
    def setup_sos_disciplines(self):
        '''
        Set possible values of preconditioner in data manager, according to liner solver MDA/MDO value
        (available preconditioners are different if petsc linear solvers are used)
        And set default value of max_mda_iter_gs according to sub_mda_class
        '''
        # set possible values of linear solver MDA preconditioner
        if 'linear_solver_MDA' in self._data_in:
            linear_solver_MDA = self.get_sosdisc_inputs('linear_solver_MDA')
            if linear_solver_MDA.endswith('_PETSC'):
                self._data_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = ['None'] + \
                    ksp_lib_petsc.AVAILABLE_PRECONDITIONER
                if self.get_sosdisc_inputs('linear_solver_MDA_preconditioner') not in self._data_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]:
                    self._data_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'gasm'
            else:
                self._data_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = [
                    'None', 'ilu']
                if self.get_sosdisc_inputs('linear_solver_MDA_preconditioner') not in self._data_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]:
                    self._data_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'None'

        # set possible values of linear solver MDO preconditioner
        if 'linear_solver_MDO' in self._data_in:
            linear_solver_MDO = self.get_sosdisc_inputs('linear_solver_MDO')
            if linear_solver_MDO.endswith('_PETSC'):
                self._data_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = ['None'] + \
                    ksp_lib_petsc.AVAILABLE_PRECONDITIONER
                if self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') not in self._data_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]:
                    self._data_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'gasm'
            else:
                self._data_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = [
                    'None', 'ilu']
                if self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') not in self._data_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]:
                    self._data_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'None'

#         # set default value of max_mda_iter_gs
#         if 'max_mda_iter_gs' in self._data_in:
#             if self.get_sosdisc_inputs('sub_mda_class') == 'GSorNewtonMDA':
#                 self.update_default_value(
#                     'max_mda_iter_gs', self.IO_TYPE_IN, 200)
#             else:
#                 self.update_default_value(
#                     'max_mda_iter_gs', self.IO_TYPE_IN, 5)

    def configure_io(self):
        '''
        Configure the SoSCoupling by : 
        - setting the discipline in the discipline_dict 
        - configure all children disciplines

        '''
        SoSDiscipline.configure(self)

        disc_to_configure = self.get_disciplines_to_configure()

        if len(disc_to_configure) > 0:
            self.set_configure_status(False)
        else:
            self.set_configure_status(True)

        for disc in disc_to_configure:
            disc.configure()

    def get_disciplines_to_configure(self):
        '''
        Get sub disciplines list to configure
        '''
        disc_to_configure = []
        for disc in self.sos_disciplines:
            if not disc.is_configured():
                disc_to_configure.append(disc)
        return disc_to_configure

    def configure(self):
        # configure SoSTrades objects
        self.configure_io()
        # configure GEMSEO objects (execution sequence)
        self.configure_execution()

    def is_configured(self):
        '''
        Return False if at least one sub discipline needs to be configured, True if not
        '''
        return self.get_configure_status() and not self.check_structuring_variables_changes() and (
            self.get_disciplines_to_configure() == [])

    def configure_execution(self):
        '''
        - configure the GEMSEO MDA with all children disciplines
        - set the data_in and data_out of the coupling with the GEMS grammar computed in configure_mda
        '''
        # update gems grammar with namespaced inputs and outputs
        for disc in self.sos_disciplines:
            disc.update_gems_grammar_with_data_io()

        self.configure_mda()

        self._update_status_dm(self.STATUS_CONFIGURE)
        # Construct the data_in and the data_out of the coupling with the GEMS
        # grammar
        self._set_data_io_with_gems_grammar()

        self._set_residual_history()

        # Update coupling and editable flags in the datamanager for the GUI
        self._update_coupling_flags_in_dm()

    def _set_data_io_with_gems_grammar(self):
        '''
        Construct the data_in and the data_out of the coupling with the GEMS grammar
        '''
        if self.with_data_io:
            # keep numerical inputs in data_in
            self._data_in = {key: value for key, value in self._data_in.items(
            ) if
                key in self.DESC_IN or key in self.NUM_DESC_IN}
            # add coupling inputs in data_in
            gems_grammar_in_keys = self.input_grammar.get_data_names()
            for var_f_name in gems_grammar_in_keys:
                var_name = self.dm.get_data(
                    var_f_name, SoSDisciplineBuilder.VAR_NAME)
                self._data_in[var_name] = self.dm.get_data(var_f_name)

            # keep residuals_history if in data_out
            if self.RESIDUALS_HISTORY in self._data_out:
                self._data_out = {
                    self.RESIDUALS_HISTORY: self._data_out[self.RESIDUALS_HISTORY]}
            else:
                self._data_out = {}
            # add coupling outputs in data_out
            gems_grammar_out_keys = self.output_grammar.get_data_names()
            for var_f_name in gems_grammar_out_keys:
                var_name = self.dm.get_data(
                    var_f_name, SoSDisciplineBuilder.VAR_NAME)
                self._data_out[var_name] = self.dm.get_data(var_f_name)

    def delete_numerical_parameters_for_gems(self, var_name):
        '''
        Delete numerical parameters from input grammar of GEMS
        '''

        if var_name in self.DESC_IN:
            return True
        else:
            return False

    def _update_coupling_flags_in_dm(self):
        ''' 
        Update coupling and editable flags in the datamanager for the GUI
        '''

        def update_flags_of_disc(coupling_key, disc_name, in_or_out):

            disc_list = self.dm.get_disciplines_with_name(disc_name)
            var_name_out = None
            for a_disc in disc_list:
                if in_or_out == 'in':
                    data_io = a_disc._data_in
                else:
                    data_io = a_disc._data_out

                if var_name_k in data_io.keys():
                    var_name_out = var_name_k
                else:
                    var_name_out_list = [
                        key for key in data_io.keys() if coupling_key.endswith(NS_SEP + key)]
                    # To be modified
                    if len(var_name_out_list) != 0:
                        var_name_out = var_name_out_list[0]
                if var_name_out is not None and var_name_out in data_io:
                    data_io[var_name_out][self.COUPLING] = True
                    if self.get_var_full_name(var_name_out, data_io) in self.strong_couplings:
                        data_io[var_name_out][self.EDITABLE] = True
                        data_io[var_name_out][self.OPTIONAL] = True
                    else:
                        data_io[var_name_out][self.EDITABLE] = False

        # END update_flags_of_disc

        # -- update couplings flag into DataManager
        coupl = self.export_couplings()
        couplings = coupl[self.VAR_NAME]
        disc_1 = coupl['disc_1']
        disc_2 = coupl['disc_2']

        # loop on couplings variables and the disciplines linked
        for k, from_disc_name, to_disc_name in zip(
                couplings, disc_1, disc_2):
            self.dm.set_data(k, self.COUPLING, True)
            # Deal with pre run of MDA to enter strong couplings if needed
            if k in self.strong_couplings:
                self.dm.set_data(k, self.IO_TYPE, self.IO_TYPE_IN)
                self.dm.set_data(k, self.EDITABLE, True)
                self.dm.set_data(k, self.OPTIONAL, True)
            else:
                self.dm.set_data(k, self.EDITABLE, False)
            var_name_k = self.dm.get_data(k, self.VAR_NAME)

            # update flags of discipline 1
            update_flags_of_disc(k, from_disc_name, 'out')
            # update flags of discipline 2
            update_flags_of_disc(k, to_disc_name, 'in')

    def configure_mda(self):
        ''' Configuration of SoSCoupling, call to super class MDAChain
        '''
        num_data = self._get_numerical_inputs()

        MDAChain.__init__(self,
                          disciplines=self.sos_disciplines,
                          name=self.sos_name,
                          grammar_type=self.SOS_GRAMMAR_TYPE,
                          **num_data)
        self.logger.info(
            f"The MDA solver of the Coupling {self.get_disc_full_name()} is set to {num_data['sub_mda_class']}")

    def _get_numerical_inputs(self):
        '''
        Get numerical parameters input values for MDAChain init
        '''
        # get input for MDAChain instantiation
        needed_numerical_param = ['sub_mda_class', 'max_mda_iter', 'n_processes',
                                  'chain_linearize', 'tolerance', 'use_lu_fact', 'warm_start']
        num_data = self.get_sosdisc_inputs(
            needed_numerical_param, in_dict=True)

        if num_data['sub_mda_class'] == 'MDAJacobi':
            num_data['acceleration'] = copy(
                self.get_sosdisc_inputs('acceleration'))
        if num_data['sub_mda_class'] == 'MDAGaussSeidel':
            num_data['warm_start_threshold'] = copy(self.get_sosdisc_inputs(
                'warm_start_threshold'))
        if num_data['sub_mda_class'] in ['GSNewtonMDA', 'GSPureNewtonMDA', 'GSorNewtonMDA']:
            #             num_data['max_mda_iter_gs'] = copy(self.get_sosdisc_inputs(
            #                 'max_mda_iter_gs'))
            num_data['tolerance_gs'] = copy(self.get_sosdisc_inputs(
                'tolerance_gs'))
        if num_data['sub_mda_class'] in ['MDANewtonRaphson', 'PureNewtonRaphson', 'GSPureNewtonMDA', 'GSNewtonMDA', 'GSorNewtonMDA']:
            num_data['relax_factor'] = copy(
                self.get_sosdisc_inputs('relax_factor'))

        # linear solver options MDA
        num_data['linear_solver'] = copy(self.get_sosdisc_inputs(
            'linear_solver_MDA'))
        linear_solver_options_MDA = deepcopy(self.get_sosdisc_inputs(
            'linear_solver_MDA_options'))

        if num_data['linear_solver'].endswith('_PETSC'):
            # PETSc case
            linear_solver_options_MDA['solver_type'] = num_data['linear_solver'].split('_PETSC')[
                0].lower()
            preconditioner = copy(self.get_sosdisc_inputs(
                'linear_solver_MDA_preconditioner'))
            linear_solver_options_MDA['preconditioner_type'] = (
                preconditioner != 'None') * preconditioner or None
        else:
            # Scipy case / gmres
            linear_solver_options_MDA['use_ilu_precond'] = (
                copy(self.get_sosdisc_inputs('linear_solver_MDA_preconditioner')) == 'ilu')

        num_data['linear_solver_tolerance'] = linear_solver_options_MDA.pop(
            'tol')
        num_data['linear_solver_options'] = linear_solver_options_MDA

        self.linear_solver_MDA = num_data['linear_solver']
        self.linear_solver_tolerance_MDA = num_data['linear_solver_tolerance']
        self.linear_solver_options_MDA = deepcopy(
            num_data['linear_solver_options'])

        # linear solver options MDO
        self.linear_solver_MDO = self.get_sosdisc_inputs('linear_solver_MDO')
        linear_solver_options_MDO = deepcopy(self.get_sosdisc_inputs(
            'linear_solver_MDO_options'))

        if self.linear_solver_MDO.endswith('_PETSC'):
            linear_solver_options_MDO['solver_type'] = self.linear_solver_MDO.split('_PETSC')[
                0].lower()
            preconditioner = self.get_sosdisc_inputs(
                'linear_solver_MDO_preconditioner')
            linear_solver_options_MDO['preconditioner_type'] = (
                preconditioner != 'None') * preconditioner or None
        else:
            linear_solver_options_MDO['use_ilu_precond'] = (
                self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') == 'ilu')

        self.linear_solver_tolerance_MDO = linear_solver_options_MDO.pop('tol')
        self.linear_solver_options_MDO = linear_solver_options_MDO

        return num_data

    def set_epsilon0(self, mda):
        '''
        Set epsilon0 that is not argument of the init of the MDA and need to be set outside of it with MDA attributes
        '''
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self.set_epsilon0(sub_mda)
        else:
            mda.epsilon0 = copy(self.get_sosdisc_inputs('epsilon0'))

    @property
    def ordered_disc_list(self):
        '''
         Property to obtain the ordered list of disciplines configured by the MDAChain
         Overwrite of sos_discipline property where the order is defined by default
         by the order of sos_disciplines
        '''
        ordered_list = []
        ordered_list = self.ordered_disc_list_rec(self.mdo_chain, ordered_list)

        return ordered_list

    def ordered_disc_list_rec(self, disc, ordered_list):
        '''
         Recursive function to obtain the ordered list of disciplines configured by the MDAChain
        '''
        for subdisc in disc.disciplines:
            if isinstance(subdisc, SoSDiscipline):
                ordered_list.append(subdisc)
            else:  # Means that it is a GEMS class MDAJacobi for example
                ordered_list = self.ordered_disc_list_rec(
                    subdisc, ordered_list)

        return ordered_list

    def export_couplings(self, in_csv=False, f_name=None):
        ''' 
            Export couplings as a csv with
        disc1 | disc2 | var_name
        '''
        # fill in data
        cs = self.coupling_structure
        coupl_tuples = cs.graph.get_disciplines_couplings()
        data = []
        header = ["disc_1", "disc_2", "var_name"]
        for disc1, disc2, c_vars in coupl_tuples:
            for var in c_vars:
                disc1_id = disc1.get_disc_full_name()
                disc2_id = disc2.get_disc_full_name()
                row = [disc1_id, disc2_id, var]
                data.append(row)
        df = DataFrame(data, columns=header)

        for discipline in self.sos_disciplines:
            if isinstance(discipline, SoSCoupling):
                df_couplings = discipline.export_couplings()
                df = df.append(df_couplings, ignore_index=True)

        if in_csv:
            # writing of the file
            if f_name is None:
                f_name = f"{self.get_disc_full_name()}.csv"
            df.to_csv(f_name, index=False)
        else:
            return df

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

        MDAChain._run(self)

        # save residual history
        dict_out = {}
        residuals_history = DataFrame(
            {f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.sub_mda_list})
        dict_out[self.RESIDUALS_HISTORY] = residuals_history

        self.store_sos_outputs_values(dict_out, update_dm=True)

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

                        self.dm.set_values_from_dict(
                            self._convert_array_into_new_type(self.local_data))

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
            inputs_values = disc.get_sosdisc_inputs()
            keys_none = [key for key, value in inputs_values.items()
                         if value is None and key not in self.NUM_DESC_IN]
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
        SoSDisciplineBuilder._run(self, update_local_data=False)

        # convert local_data into new types and store values in data manager
        local_data_sos = self._convert_array_into_new_type(self.local_data)
        self.dm.set_values_from_dict(local_data_sos)

        # logging of residuals of the mdas
        # if len(self.sub_mda_list) > 0:
        # self.logger.info(f'{self.get_disc_full_name()} MDA history')
        # for sub_mda in self.sub_mda_list:
        # self.logger.info('\tIt.\tRes. norm')
        # for res_tuple in sub_mda.residual_history:
        # res_norm = '{:e}'.format(res_tuple[0])
        # self.logger.info(f'\t{res_tuple[1]}\t{res_norm}')

    def linearize(self, input_data=None, force_all=False, force_no_exec=False):
        '''
        Overload the linearize of soscoupling to use the one of sosdiscipline and not the one of MDAChain
        '''
        self.logger.info(
            f'Computing the gradient for the MDA : {self.get_disc_full_name()}')

        return SoSDisciplineBuilder.linearize(
            self, input_data=input_data, force_all=force_all, force_no_exec=force_no_exec)

    def check_jacobian(self, input_data=None, derr_approx=SoSDisciplineBuilder.FINITE_DIFFERENCES,
                       step=1e-7, threshold=1e-8, linearization_mode='auto',
                       inputs=None, outputs=None, parallel=False,
                       n_processes=SoSDisciplineBuilder.N_CPUS,
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
            SoSDiscipline._check_min_max_gradients(self, self.jac)

    def get_input_data_for_gems(self):

        input_data = SoSDisciplineBuilder.get_input_data_for_gems(self)

        # filter numerical inputs
        for key in list(self.DESC_IN.keys()):
            ns_key = self.get_var_full_name(key, self._data_in)
            if ns_key in input_data:
                del input_data[ns_key]

        return input_data

    def get_maturity(self):
        '''
        Get the maturity of the sos_coupling by adding all maturities of children disciplines
        '''
        ref_dict_maturity = deepcopy(self.dict_maturity_ref)
        for discipline in self.sos_disciplines:
            disc_maturity = discipline.get_maturity()
            if isinstance(disc_maturity, dict):
                for m_k in ref_dict_maturity.keys():
                    if m_k in disc_maturity:
                        ref_dict_maturity[m_k] += disc_maturity[m_k]
            elif disc_maturity in ref_dict_maturity:
                ref_dict_maturity[disc_maturity] += 1
        self.set_maturity(ref_dict_maturity, maturity_dict=True)
        return self._maturity

    def remove_discipline(self, disc):
        ''' remove one discipline from coupling
        '''
        disc.clean_dm_from_disc()
        self.sos_disciplines.remove(disc)
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(
            disc, self.disc_id)

    def remove_discipline_list(self, disc_list):
        ''' remove one discipline from coupling
        '''
        for disc in disc_list:
            disc.clean_dm_from_disc()
            self.ee.ns_manager.remove_dependencies_after_disc_deletion(
                disc, self.disc_id)
        self.sos_disciplines = [
            disc for disc in self.sos_disciplines if disc not in disc_list]

    def _set_residual_history(self):
        ''' set residuals history into data_out
        and update DM
        '''
        # dataframe init
        residuals_history = DataFrame(
            {f'{sub_mda.name}': sub_mda.residual_history for sub_mda in self.sub_mda_list})

        # set residual type and value
        rdict = {}
        rdict[self.RESIDUALS_HISTORY] = {}
        rdict[self.RESIDUALS_HISTORY][self.USER_LEVEL] = 3
        rdict[self.RESIDUALS_HISTORY][self.TYPE] = 'dataframe'
        rdict[self.RESIDUALS_HISTORY][self.VALUE] = residuals_history

        # init other fields
        full_out = self._prepare_data_dict(self.IO_TYPE_OUT, rdict)
        self.dm.update_with_discipline_dict(
            disc_id=self.disc_id, disc_dict=full_out)

        # update in loader_out
        self._data_out.update(full_out)

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
                    if self.get_sosdisc_inputs("group_mda_disciplines"):
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
                    self.set_epsilon0(sub_mda)

                    chained_disciplines.append(sub_mda)
                    self.sub_mda_list.append(sub_mda)
                else:
                    # single discipline
                    chained_disciplines.append(first_disc)

        if self.get_sosdisc_inputs("n_subcouplings_parallel") > 1:
            chained_disciplines = self._parallelize_chained_disciplines(
                chained_disciplines, self.grammar_type)

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
            n_subcouplings_parallel = self.get_sosdisc_inputs(
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

    def clean(self):
        """This method cleans a coupling
        We first begin by cleaning all the disciplines executed by the coupling and then the coupling itself
        When cleaning the coupling a particular check  on epsilon0 is done
        """
        for discipline in self.sos_disciplines:
            discipline.clean()

        SoSDiscipline.clean(self)
        # if 'epsilon0' in self._data_in:
        #     self.ee.dm.remove_keys(
        # self.disc_id, [self.get_var_full_name('epsilon0', self._data_in)])
