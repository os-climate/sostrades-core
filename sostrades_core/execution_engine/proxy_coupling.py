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
from gemseo.core.chain import MDOChain
from gemseo.mda.sequential_mda import MDASequential
from sostrades_core.tools.filter.filter import filter_variables_to_convert
from gemseo.mda.mda_chain import MDAChain
from sostrades_core.execution_engine.mdo_discipline_wrapp import MDODisciplineWrapp

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from copy import deepcopy, copy
from multiprocessing import cpu_count

from pandas import DataFrame
import platform
import logging
from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.execution_engine.proxy_discipline_builder import ProxyDisciplineBuilder
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory

if platform.system() != 'Windows':
    from sostrades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import PetscKSPAlgos as ksp_lib_petsc

# from sostrades_core.execution_engine.parallel_execution.sos_parallel_mdo_chain import SoSParallelChain

N_CPUS = cpu_count()
LOGGER = logging.getLogger(__name__)


def get_available_linear_solvers():
    '''Get available linear solvers list
    '''
    lsf = LinearSolversFactory()
    algos = lsf.algorithms
    del lsf

    return algos


class ProxyCoupling(ProxyDisciplineBuilder):
    """
    **ProxyCoupling** is a ProxyDiscipline that represents a coupling and has children sub proxies on the process tree.

    An instance of ProxyCoupling is in one to one aggregation with an instance of MDODisciplineWrapp that has no wrapper,
    but has a GEMSEO MDAChain instantiated at the prepare_execution step.

    Attributes:
        cls_builder (List[SoSBuilder]): list of the sub proxy builders

        mdo_discipline_wrapp (MDODisciplineWrapp): aggregated object that references a GEMSEO MDAChain
    """

    # ontology information
    _ontology_data = {
        'label': 'Coupling',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-cogs fa-fw',
        'version': '',
    }
    SECANT_ACCELERATION = "secant"
    M2D_ACCELERATION = "m2d"
    RESIDUALS_HISTORY = "residuals_history"

    # get list of available linear solvers from LinearSolversFactory
    AVAILABLE_LINEAR_SOLVERS = get_available_linear_solvers()

    # set default value of linear solver according to the operating system
    #     if platform.system() == 'Windows':
    #         DEFAULT_LINEAR_SOLVER = 'GMRES'
    #         DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'None'
    #         POSSIBLE_VALUES_PRECONDITIONER = ['None', 'ilu']
    #     else:
    #         DEFAULT_LINEAR_SOLVER = 'GMRES_PETSC'
    #         DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'gasm'
    #         POSSIBLE_VALUES_PRECONDITIONER = [
    #             'None'] + ksp_lib_petsc.AVAILABLE_PRECONDITIONER

    DEFAULT_LINEAR_SOLVER_OPTIONS = {
        'max_iter': 1000,
        'tol': 1.0e-8}

    DESC_IN = {
        # NUMERICAL PARAMETERS
        'sub_mda_class': {ProxyDiscipline.TYPE: 'string',
                          ProxyDiscipline.POSSIBLE_VALUES: ['MDAJacobi', 'MDAGaussSeidel', 'MDANewtonRaphson',
                                                            'PureNewtonRaphson', 'MDAQuasiNewton', 'GSNewtonMDA',
                                                            'GSPureNewtonMDA', 'GSorNewtonMDA', 'MDASequential',
                                                            'GSPureNewtonorGSMDA'],
                          ProxyDiscipline.DEFAULT: 'MDAJacobi', ProxyDiscipline.NUMERICAL: True,
                          ProxyDiscipline.STRUCTURING: True},
        'max_mda_iter': {ProxyDiscipline.TYPE: 'int', ProxyDiscipline.DEFAULT: 30, ProxyDiscipline.NUMERICAL: True,
                         ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        'n_processes': {ProxyDiscipline.TYPE: 'int', ProxyDiscipline.DEFAULT: 1, ProxyDiscipline.NUMERICAL: True,
                        ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        'chain_linearize': {ProxyDiscipline.TYPE: 'bool', ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                            ProxyDiscipline.DEFAULT: False, ProxyDiscipline.NUMERICAL: True,
                            ProxyDiscipline.STRUCTURING: True},
        'tolerance': {ProxyDiscipline.TYPE: 'float', ProxyDiscipline.DEFAULT: 1.e-6, ProxyDiscipline.NUMERICAL: True,
                      ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        'use_lu_fact': {ProxyDiscipline.TYPE: 'bool', ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                        ProxyDiscipline.DEFAULT: False, ProxyDiscipline.NUMERICAL: True,
                        ProxyDiscipline.STRUCTURING: True},
        'warm_start': {ProxyDiscipline.TYPE: 'bool', ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                       ProxyDiscipline.DEFAULT: False, ProxyDiscipline.NUMERICAL: True,
                       ProxyDiscipline.STRUCTURING: True},
        'acceleration': {ProxyDiscipline.TYPE: 'string',
                         ProxyDiscipline.POSSIBLE_VALUES: [M2D_ACCELERATION, SECANT_ACCELERATION, 'none'],
                         ProxyDiscipline.DEFAULT: M2D_ACCELERATION, ProxyDiscipline.NUMERICAL: True,
                         ProxyDiscipline.STRUCTURING: True},
        'warm_start_threshold': {ProxyDiscipline.TYPE: 'float', ProxyDiscipline.DEFAULT: -1,
                                 ProxyDiscipline.NUMERICAL: True,
                                 ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        # parallel sub couplings execution
        'n_subcouplings_parallel': {ProxyDiscipline.TYPE: 'int', ProxyDiscipline.DEFAULT: 1,
                                    ProxyDiscipline.NUMERICAL: True,
                                    ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        # 'max_mda_iter_gs': {ProxyDiscipline.TYPE: 'int', ProxyDiscipline.DEFAULT: 5, ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True},
        'tolerance_gs': {ProxyDiscipline.TYPE: 'float', ProxyDiscipline.DEFAULT: 10.0, ProxyDiscipline.NUMERICAL: True,
                         ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        'relax_factor': {ProxyDiscipline.TYPE: 'float', ProxyDiscipline.RANGE: [0.0, 1.0],
                         ProxyDiscipline.DEFAULT: 0.99,
                         ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        # NUMERICAL PARAMETERS OUT OF INIT
        'epsilon0': {ProxyDiscipline.TYPE: 'float', ProxyDiscipline.DEFAULT: 1.0e-6, ProxyDiscipline.NUMERICAL: True,
                     ProxyDiscipline.STRUCTURING: True, ProxyDiscipline.UNIT: '-'},
        # Linear solver for MD0
        'linear_solver_MDO': {ProxyDiscipline.TYPE: 'string',
                              ProxyDiscipline.DEFAULT: 'GMRES',
                              #                               ProxyDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS,
                              # ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER,
                              ProxyDiscipline.NUMERICAL: True,
                              ProxyDiscipline.STRUCTURING: True},
        'linear_solver_MDO_preconditioner': {ProxyDiscipline.TYPE: 'string',
                                             ProxyDiscipline.DEFAULT: 'None',
                                             #                                              ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER,
                                             # ProxyDiscipline.POSSIBLE_VALUES:
                                             # POSSIBLE_VALUES_PRECONDITIONER,
                                             ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True},
        'linear_solver_MDO_options': {ProxyDiscipline.TYPE: 'dict',
                                      ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS,
                                      ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True,
                                      ProxyDiscipline.UNIT: '-'},
        # Linear solver for MDA
        'linear_solver_MDA': {ProxyDiscipline.TYPE: 'string',
                              ProxyDiscipline.DEFAULT: 'GMRES',
                              #                               ProxyDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS,
                              # ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER,
                              ProxyDiscipline.NUMERICAL: True,
                              ProxyDiscipline.STRUCTURING: True},
        'linear_solver_MDA_preconditioner': {ProxyDiscipline.TYPE: 'string',
                                             ProxyDiscipline.DEFAULT: 'None',
                                             #                                              ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER,
                                             # ProxyDiscipline.POSSIBLE_VALUES:
                                             # POSSIBLE_VALUES_PRECONDITIONER,
                                             ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True},
        'linear_solver_MDA_options': {ProxyDiscipline.TYPE: 'dict',
                                      ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS,
                                      ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True,
                                      ProxyDiscipline.UNIT: '-'},
        # group all disciplines in a MDOChain
        'group_mda_disciplines': {ProxyDiscipline.TYPE: 'bool', ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                                  ProxyDiscipline.DEFAULT: False, ProxyDiscipline.USER_LEVEL: 3,
                                  ProxyDiscipline.NUMERICAL: True, ProxyDiscipline.STRUCTURING: True},
        'authorize_self_coupled_disciplines': {ProxyDiscipline.TYPE: 'bool',
                                               ProxyDiscipline.POSSIBLE_VALUES: [True, False],
                                               ProxyDiscipline.DEFAULT: False,
                                               ProxyDiscipline.USER_LEVEL: 3,
                                               ProxyDiscipline.STRUCTURING: True}
    }

    DESC_OUT = {
        RESIDUALS_HISTORY: {ProxyDiscipline.USER_LEVEL: 3, ProxyDiscipline.TYPE: 'dataframe',
                            ProxyDiscipline.UNIT: '-'}
    }

    eps0 = 1.0e-6
    has_chart = False

    def __init__(self, sos_name, ee, cls_builder=None, with_data_io=False, associated_namespaces=None):
        '''
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[Class]): list of the sub proxy constructors for the recursive build of the process tree [???]
        '''
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder  # TODO: Move to ProxyDisciplineBuilder?
        self._reload(sos_name, ee, associated_namespaces=associated_namespaces)
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

        self.mdo_discipline_wrapp = MDODisciplineWrapp(name=sos_name)

    def _reload(self, sos_name, ee, associated_namespaces=None):
        '''
        Reload ProxyCoupling with corresponding ProxyDiscipline attributes and set is_sos_coupling.

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
        '''
        self.is_sos_coupling = True
        ProxyDiscipline._reload(
            self, sos_name, ee, associated_namespaces=associated_namespaces)

    # TODO: [and TODISCUSS] move it to mdo_discipline_wrapp, if we want to
    # reduce footprint in GEMSEO
    def _set_dm_cache_map(self):
        '''
        Update cache_map dict in DM with cache, mdo_chain cache, sub_mda_list caches, and its children recursively
        '''
        mda_chain = self.mdo_discipline_wrapp.mdo_discipline
        if mda_chain.cache is not None:
            # store SoSCoupling cache in DM
            self._store_cache_with_hashed_uid(mda_chain)

            # store mdo_chain cache in DM
            self._store_cache_with_hashed_uid(mda_chain.mdo_chain)

            # store sub mdas cache recursively
            for mda in mda_chain.sub_mda_list:
                self._set_sub_mda_dm_cache_map(mda)

        # store children cache recursively
        for disc in self.proxy_disciplines:
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

    # def build(self):
    #     """
    #     Instanciate sub proxies managed by the coupling
    #     """
    #     old_current_discipline = self.ee.factory.current_discipline
    #     self.ee.factory.current_discipline = self
    #     for builder in self.cls_builder:
    #         proxy_disc = builder.build()
    #         if proxy_disc not in self.proxy_disciplines:
    #             self.ee.factory.add_discipline(proxy_disc)
    #     # If the old_current_discipline is None that means that it is the first build of a coupling then self is the
    #     # high level coupling and we do not have to restore the
    #     # current_discipline
    #     if old_current_discipline is not None:
    #         self.ee.factory.current_discipline = old_current_discipline
    #
    # #     def clear_cache(self):
    # #         self.mdo_chain.cache.clear()
    # #         ProxyDisciplineBuilder.clear_cache(self)

    # -- Public methods

    def setup_sos_disciplines(self):
        '''
        Set possible values of preconditioner in data manager, according to liner solver MDA/MDO value
        (available preconditioners are different if petsc linear solvers are used)
        And set default value of max_mda_iter_gs according to sub_mda_class
        '''
        disc_in = self.get_data_in()
        # set possible values of linear solver MDA preconditioner
        if 'linear_solver_MDA' in disc_in:
            linear_solver_MDA = self.get_sosdisc_inputs('linear_solver_MDA')
            if linear_solver_MDA.endswith('_PETSC'):
                if platform.system() == 'Windows':
                    raise Exception(
                        f'Petsc solvers cannot be used on Windows platform, modify linear_solver_MDA option of {self.sos_name} : {linear_solver_MDA}')
                disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = ['None'] + \
                    ksp_lib_petsc.AVAILABLE_PRECONDITIONER
                if self.get_sosdisc_inputs('linear_solver_MDA_preconditioner') not in \
                        disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]:
                    disc_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'gasm'
            else:
                disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = [
                    'None', 'ilu']
                if self.get_sosdisc_inputs('linear_solver_MDA_preconditioner') not in \
                        disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]:
                    disc_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'None'

        # set possible values of linear solver MDO preconditioner
        if 'linear_solver_MDO' in disc_in:
            linear_solver_MDO = self.get_sosdisc_inputs('linear_solver_MDO')
            if linear_solver_MDO.endswith('_PETSC'):
                if platform.system() == 'Windows':
                    raise Exception(
                        f'Petsc solvers cannot be used on Windows platform, modify linear_solver_MDA option of {self.sos_name} : {linear_solver_MDA}')
                disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = ['None'] + \
                    ksp_lib_petsc.AVAILABLE_PRECONDITIONER
                if self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') not in \
                        disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]:
                    disc_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'gasm'
            else:
                disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = [
                    'None', 'ilu']
                if self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') not in \
                        disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]:
                    disc_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'None'

            # set default value of max_mda_iter_gs
            if 'max_mda_iter_gs' in disc_in:
                if self.get_sosdisc_inputs('sub_mda_class') == 'GSorNewtonMDA':
                    self.update_default_value(
                        'max_mda_iter_gs', self.IO_TYPE_IN, 200)
                else:
                    self.update_default_value(
                        'max_mda_iter_gs', self.IO_TYPE_IN, 5)

    def configure_io(self):
        '''
        Configure the ProxyCoupling by :
        - setting the discipline in the discipline_dict 
        - configure all children disciplines
        '''
        ProxyDiscipline.configure(self)

        disc_to_configure = self.get_disciplines_to_configure()

        if len(disc_to_configure) > 0:
            self.set_configure_status(False)
            for disc in disc_to_configure:
                disc.configure()
        else:
            self.set_children_cache_inputs()
            # - all chidren are configured thus proxyCoupling can be configured
            self.set_configure_status(True)
            # - build the coupling structure
            self._build_coupling_structure()
            # - builds data_in/out according to the coupling structure
            self._build_data_io()
            # - Update coupling and editable flags in the datamanager for the GUI
            self._update_coupling_flags_in_dm()

    def _build_data_io(self):
        """
        Build data_in and data_out from sub proxies data_in and out
        we build also data_in_with_full_name and data_out_with_full_name
        to be able to retrieve inputs and outputs with same short name
        in sub proxies
        """
        #- build the data_i/o (sostrades) based on input and output grammar of MDAChain (GEMSEO)
        subprocess_data_in_ns_tuple, subprocess_data_out_ns_tuple = self.__compute_mdachain_gemseo_based_data_io()

        #- data_i/o setup
        #- TODO: check if we can remove _data_in_with_full_name
        # self._data_in_with_full_name = {f'{self.get_disc_full_name()}.{key}': value for key, value in
        #                                 self._data_in.items()
        # if key in self.DESC_IN or key in self.NUM_DESC_IN}
        self._data_in_ns_tuple = {(key, id(value[self.NS_REFERENCE])): value for key, value in
                                  self._data_in.items()
                                  if key in self.DESC_IN or key in self.NUM_DESC_IN}
        self._data_in = {key: value for key, value in self._data_in.items(
        ) if
            key in self.DESC_IN or key in self.NUM_DESC_IN}

        # add inputs - that are not outputs - of all children disciplines in
        # data_in
        self._data_in_ns_tuple.update(subprocess_data_in_ns_tuple)
        # for k_full in data_in:
        #     self._data_in_with_full_name[k_full] = data_in[k_full]
        # if not self.ee.dm.get_data(k_full, self.NUMERICAL): #TODO: check if we can avoid this call to the DM, may be interesting to use data_in directly (perfo improvements)
        #     self._data_in[k] = self.dm.get_data(k_full)

        # self._data_out_with_full_name = {f'{self.get_disc_full_name()}.{key}': value for key, value in
        #                                 self._data_out.items()
        #                                 if key in self.DESC_OUT}
        self._data_out_ns_tuple = {(key, id(value[self.NS_REFERENCE])): value for key, value in
                                   self._data_out.items()
                                   if key in self.DESC_OUT}
        self._data_out = {key: value for key, value in self._data_out.items(
        ) if
            key in self.DESC_OUT}

        # # keep residuals_history if in data_out
        # if self.RESIDUALS_HISTORY in self._data_out:
        #     self._data_out_with_full_name = {
        #         f'{self.get_disc_full_name()}.{self.RESIDUALS_HISTORY}': self._data_out[self.RESIDUALS_HISTORY]}
        #     self._data_out = {
        #         self.RESIDUALS_HISTORY: self._data_out[self.RESIDUALS_HISTORY]} #TODO: shouldn't overwrite data_out
        # else:
        #     self._data_out_with_full_name = {}
        #     self._data_out = {}

        # add outputs of all children disciplines in data_out
        self._data_out_ns_tuple.update(subprocess_data_out_ns_tuple)
        # for k in data_out:
        #     k_full = self.get_var_full_name(k, data_out)
        #     self._data_out_with_full_name[k_full] = data_out[k]
        #     if not self.ee.dm.get_data(k_full, self.NUMERICAL):
        #         self._data_out[k] = self.dm.get_data(k_full)

    def __compute_mdachain_gemseo_based_data_io(self):
        ''' mimics the definition of MDA i/o grammar
        '''
        #- identify i/o grammars like in GEMSEO (like in initialize_grammar method in MDOChain)
        mda_outputs = []
        mda_inputs = []
        get_data = self.dm.get_data
        data_in = {}
        data_out = {}
        for d in self.proxy_disciplines:
            # disc_in = d.get_input_data_names(as_namespaced_tuple=True)
            # disc_out = d.get_output_data_names(as_namespaced_tuple=True)
            # mda_outputs += disc_out
            #
            # # TODO [to discuss]: aren't these zips problematic with repeated short names that are crushed in data_in?
            # # get all inputs that are not in known outputs
            # mda_inputs += list(set(disc_in) - set(mda_outputs))
            # d_data_in = {k_full: get_data(k_full) for k_full in disc_in if k_full not in mda_outputs}
            # data_in.update(d_data_in)
            #
            # # get all outputs
            # d_data_out = {k_full: get_data(k_full) for k_full in disc_out}
            # data_out.update(d_data_out)

            disc_in = d.get_data_io_with_full_name(
                self.IO_TYPE_IN, as_namespaced_tuple=True)
            disc_out = d.get_data_io_with_full_name(
                self.IO_TYPE_OUT, as_namespaced_tuple=True)
            mda_outputs += disc_out
            d_data_in = {key: value for key,
                         value in disc_in.items() if key not in mda_outputs}
            data_in.update(d_data_in)
            data_out.update(disc_out)

        return data_in, data_out

    # def get_input_data_names(self):
    #     '''
    #     Returns:
    #         (List[string]) of input data full names based on i/o and namespaces declarations in the user wrapper
    #     '''
    #     return list(self._data_in_with_full_name.keys())
    #
    # def get_output_data_names(self):
    #     '''
    #     Returns:
    #         (List[string]) outpput data full names based on i/o and namespaces declarations in the user wrapper
    #     '''
    #     return list(self._data_out_with_full_name.keys())

    def _build_coupling_structure(self):
        """
        Build MDOCouplingStructure
        """

        self.coupling_structure = MDOCouplingStructure(self.proxy_disciplines)
        self.strong_couplings = filter_variables_to_convert(self.ee.dm.convert_data_dict_with_full_name(),
                                                            self.coupling_structure.strong_couplings(),
                                                            write_logs=True, logger=LOGGER)

    def configure(self):
        """
        Configure i/o, update status, update status in dm.
        """
        # configure SoSTrades objects
        self.configure_io()

        self._update_status_dm(self.STATUS_CONFIGURE)

    def _update_coupling_flags_in_dm(self):
        ''' 
        Update coupling and editable flags in the datamanager for the GUI
        '''

        def update_flags_of_disc(coupling_key, disc_name, in_or_out):

            disc_list = self.dm.get_disciplines_with_name(disc_name)
            var_name_out = None
            for a_disc in disc_list:
                if in_or_out == 'in':
                    data_io = a_disc.get_data_in()
                else:
                    data_io = a_disc.get_data_out()

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

        for discipline in self.proxy_disciplines:
            if isinstance(discipline, ProxyCoupling):
                df_couplings = discipline.export_couplings()
                df = df.append(df_couplings, ignore_index=True)

        if in_csv:
            # writing of the file
            if f_name is None:
                f_name = f"{self.get_disc_full_name()}.csv"
            df.to_csv(f_name, index=False)
        else:
            return df

    def is_configured(self):
        '''
        Return False if at least one sub discipline needs to be configured, True if not
        '''
        return self.get_configure_status() and not self.check_structuring_variables_changes() and (
            self.get_disciplines_to_configure() == [])

    def prepare_execution(self):
        '''
        Preparation of the GEMSEO process, including GEMSEO objects instanciation
        '''
        # prepare_execution of proxy_disciplines
        sub_mdo_disciplines = []
        for disc in self.proxy_disciplines:
            disc.prepare_execution()
            # Exclude non executable proxy Disciplines
            if disc.mdo_discipline_wrapp is not None:
                sub_mdo_disciplines.append(
                    disc.mdo_discipline_wrapp.mdo_discipline)

        # store cache and n_calls before MDAChain reset, if prepare_execution
        # has already been called
        if self.mdo_discipline_wrapp.mdo_discipline is not None:
            mda_chain_cache = self.mdo_discipline_wrapp.mdo_discipline.cache
            mda_chain_n_calls = self.mdo_discipline_wrapp.mdo_discipline.n_calls
        else:
            mda_chain_cache = None
            mda_chain_n_calls = None

        # create_mda_chain from MDODisciplineWrapp
        self.mdo_discipline_wrapp.create_mda_chain(sub_mdo_disciplines, self)

        # set cache cache of gemseo object
        self.set_gemseo_disciplines_caches(mda_chain_cache, mda_chain_n_calls)

    def set_gemseo_disciplines_caches(self, mda_chain_cache, mda_chain_n_calls):
        '''
        Set cache of MDAChain, MDOChain and sub MDAs
        '''
        cache_type = self.get_sosdisc_inputs('cache_type')
        cache_file_path = self.get_sosdisc_inputs('cache_file_path')
        # set MDAChain cache
        if self._reset_cache:
            # set new cache when cache_type have changed (self._reset_cache == True)
            # TODO: pass cache to MDAChain init to avoid reset cache
            self.set_cache(self.mdo_discipline_wrapp.mdo_discipline,
                           cache_type, cache_file_path)
            self._reset_cache = False
        else:
            # reset stored cache and n_calls of MDAChain
            self.mdo_discipline_wrapp.mdo_discipline.cache = mda_chain_cache
            self.mdo_discipline_wrapp.mdo_discipline.n_calls = mda_chain_n_calls

        # set cache of MDOChain with cache_type and cache_file_path inputs of
        # ProxyCoupling
        self.set_cache(
            self.mdo_discipline_wrapp.mdo_discipline.mdo_chain, cache_type, cache_file_path)

        # set epsilon0 and cache of sub_mda_list
        for sub_mda in self.mdo_discipline_wrapp.mdo_discipline.sub_mda_list:
            self.set_epsilon0_and_cache(sub_mda)

    def check_var_data_mismatch(self):
        '''
        Check if a variable data is not coherent between two coupling disciplines

        The check if a variable that is used in input of multiple disciplines is coherent is made in check_inputs of datamanager
        the list of data_to_check is defined in SoSDiscipline
        '''

        # TODO: probably better if moved into proxy discipline

        if self.logger.level <= logging.DEBUG:
            coupling_vars = self.coupling_structure.graph.get_disciplines_couplings()
            for from_disc, to_disc, c_vars in coupling_vars:
                for var in c_vars:
                    # from disc is in output
                    from_disc_data = from_disc.get_data_with_full_name(
                        from_disc.IO_TYPE_OUT, var)
                    # to_disc is in input
                    to_disc_data = to_disc.get_data_with_full_name(
                        to_disc.IO_TYPE_IN, var)
                    for data_name in to_disc.DATA_TO_CHECK:
                        # Check if data_names are different
                        if from_disc_data[data_name] != to_disc_data[data_name]:
                            self.logger.debug(
                                f'The {data_name} of the coupling variable {var} is not the same in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}')
                        # Check if unit is not None
                        elif from_disc_data[data_name] is None and data_name == to_disc.UNIT:
                            # if unit is None in a dataframe check if there is a
                            # dataframe descriptor with unit in it
                            if from_disc_data[to_disc.TYPE] == 'dataframe':
                                # if no dataframe descriptor and no unit
                                # warning
                                if from_disc_data[to_disc.DATAFRAME_DESCRIPTOR] is None:
                                    self.logger.debug(
                                        f'The unit and the dataframe descriptor of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]} : cannot find unit for this dataframe')
                            # TODO : Check the unit in the dataframe descriptor of both data and check if it is ok : Need to add a new value to the df_descriptor tuple check with WALL-E
                            #                             else :
                            #                                 from_disc_data[self.DATAFRAME_DESCRIPTOR]
                            else:
                                self.logger.debug(
                                    f'The unit of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}')

    def set_epsilon0_and_cache(self, mda):
        '''
        Set epsilon0 that is not argument of the init of the MDA and need to be set outside of it with MDA attributes
        '''
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self.set_epsilon0_and_cache(sub_mda)
        mda.epsilon0 = copy(self.get_sosdisc_inputs('epsilon0'))
        self.set_cache(mda, self.get_sosdisc_inputs(
            'cache_type'), self.get_sosdisc_inputs('cache_file_path'))

    def _get_numerical_inputs(self):
        '''
        Get numerical parameters input values for MDAChain init
        '''
        # get input for MDAChain instantiation
        needed_numerical_param = ['sub_mda_class', 'max_mda_iter', 'n_processes', 'chain_linearize', 'tolerance',
                                  'use_lu_fact', 'warm_start',
                                  'n_processes']
        num_data = self.get_sosdisc_inputs(
            needed_numerical_param, in_dict=True)

        if num_data['sub_mda_class'] == 'MDAJacobi':
            num_data['acceleration'] = copy(
                self.get_sosdisc_inputs('acceleration'))
        if num_data['sub_mda_class'] == 'MDAGaussSeidel':
            num_data['warm_start_threshold'] = copy(self.get_sosdisc_inputs(
                'warm_start_threshold'))
        if num_data['sub_mda_class'] in ['GSNewtonMDA', 'GSPureNewtonMDA', 'GSorNewtonMDA', 'GSPureNewtonorGSMDA']:
            #             num_data['max_mda_iter_gs'] = copy(self.get_sosdisc_inputs(
            #                 'max_mda_iter_gs'))
            num_data['tolerance_gs'] = copy(self.get_sosdisc_inputs(
                'tolerance_gs'))
        if num_data['sub_mda_class'] in ['MDANewtonRaphson', 'PureNewtonRaphson', 'GSPureNewtonMDA', 'GSNewtonMDA',
                                         'GSorNewtonMDA', 'GSPureNewtonorGSMDA']:
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

    def get_maturity(self):
        '''
        Get the maturity of the coupling proxy by adding all maturities of children proxy disciplines
        '''
        ref_dict_maturity = deepcopy(self.dict_maturity_ref)
        for discipline in self.proxy_disciplines:
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
        '''
        Remove one discipline from coupling
        '''
        disc.clean_dm_from_disc()
        self.proxy_disciplines.remove(disc)
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(
            disc, self.disc_id)

    def remove_discipline_list(self, disc_list):
        '''
        Remove several disciplines from coupling
        '''
        for disc in disc_list:
            self.remove_discipline(disc)

    @property
    def ordered_disc_list(self):
        '''
         Property to obtain the ordered list of disciplines configured by the MDAChain
         Overwrite of sos_discipline property where the order is defined by default
         by the order of sos_disciplines
        '''

        ordered_list = self.proxy_disciplines
        self.logger.warning(
            "TODO: fix the order disc list in proxy coupling (set as the top level list of disciplines for debug purpose)")

        return ordered_list

    def ordered_disc_list_rec(self, disc, ordered_list):
        '''
         Recursive function to obtain the ordered list of disciplines configured by the MDAChain
        '''
        for subdisc in disc.disciplines:
            if isinstance(subdisc, ProxyDiscipline):
                ordered_list.append(subdisc)
            else:  # Means that it is a GEMS class MDAJacobi for example
                ordered_list = self.ordered_disc_list_rec(
                    subdisc, ordered_list)

        return ordered_list
