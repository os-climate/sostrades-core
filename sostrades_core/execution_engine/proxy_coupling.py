'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/17-2024/06/28 Copyright 2023 Capgemini

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

import logging
from collections import ChainMap, defaultdict
from copy import copy, deepcopy
from multiprocessing import cpu_count
from os import getenv
from typing import Any, ClassVar

import numpy as np
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.sequential_mda import MDASequential
from gemseo.scenarios.base_scenario import BaseScenario
from numpy import ndarray
from pandas import DataFrame, concat

from sostrades_core.execution_engine.discipline_wrapp import DisciplineWrapp
from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_builder import (
    ProxyDisciplineBuilder,
)
from sostrades_core.tools.filter.filter import filter_variables_to_convert
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

N_CPUS = cpu_count()


def get_available_linear_solvers():
    """Get available linear solvers list"""
    lsf = LinearSolverLibraryFactory()
    algos = lsf.algorithms
    del lsf

    return algos


class ProxyCoupling(ProxyDisciplineBuilder):
    """
    **ProxyCoupling** is a ProxyDiscipline that represents a coupling and has children sub proxies on the process tree.

    An instance of ProxyCoupling is in one to one aggregation with an instance of DisciplineWrapp that has no wrapper,
    but has a GEMSEO MDAChain instantiated at the prepare_execution step.

    Attributes:
        cls_builder (List[SoSBuilder]): list of the sub proxy builders

        discipline_wrapp (DisciplineWrapp): aggregated object that references a GEMSEO MDAChain
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
    RESIDUALS_HISTORY = "residuals_history"
    RESIDUALS_HISTORY_PRETTY = "Residuals History"
    NORMALIZED_RESIDUAL_NORM = 'MDA residuals norm'
    # AUTHORIZE_SELF_COUPLED_DISCIPLINES = "authorize_self_coupled_disciplines"

    # get list of available linear solvers from LinearSolversFactory
    AVAILABLE_LINEAR_SOLVERS = get_available_linear_solvers()
    NEWTON_ALGO_LIST = ('MDANewtonRaphson', 'MDAGSNewton')

    # set default value of linear solver according to the operating system
    if getenv("USE_PETSC", "").lower() in ("true", "1"):
        from sostrades_core.execution_engine.gemseo_addon.linear_solvers.ksp_lib import SoSPetscKSPAlgos

        PETSC_AVAILABLE_LINEAR_SOLVER = (algo for algo in SoSPetscKSPAlgos.ALGORITHM_INFOS)
        PETSC_AVAILABLE_PRECONDITIONER = SoSPetscKSPAlgos.AVAILABLE_PRECONDITIONERS
        DEFAULT_LINEAR_SOLVER = 'GMRES_PETSC'
        DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'gasm'
        POSSIBLE_VALUES_PRECONDITIONER = ["None", *PETSC_AVAILABLE_PRECONDITIONER]
        AVAILABLE_LINEAR_SOLVERS += PETSC_AVAILABLE_LINEAR_SOLVER
    else:
        DEFAULT_LINEAR_SOLVER = 'GMRES'
        DEFAULT_LINEAR_SOLVER_PRECONFITIONER = 'None'
        POSSIBLE_VALUES_PRECONDITIONER = ['None', 'ilu']
    DEFAULT_LINEAR_SOLVER_OPTIONS: ClassVar[dict[str, Any]] = {'max_iter': 1000, 'tol': 1.0e-8}

    DESC_IN = {
        # NUMERICAL PARAMETERS
        'inner_mda_name': {
            ProxyDiscipline.TYPE: 'string',
            ProxyDiscipline.POSSIBLE_VALUES: [
                'MDAJacobi',
                'MDAGaussSeidel',
                'MDANewtonRaphson',
                'MDAQuasiNewton',
                'MDAGSNewton',
                # 'GSorNewtonMDA'# TODO: functionality needs EEv4 update
                # 'PureNewtonRaphson', 'GSPureNewtonMDA', 'GSPureNewtonorGSMDA' PureNewtonRaphson needs update with new gemseo convergence
                'MDASequential',
            ],
            ProxyDiscipline.DEFAULT: 'MDAJacobi',
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'max_mda_iter': {
            ProxyDiscipline.TYPE: 'int',
            ProxyDiscipline.DEFAULT: 30,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'n_processes': {
            ProxyDiscipline.TYPE: 'int',
            ProxyDiscipline.DEFAULT: 1,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'chain_linearize': {
            ProxyDiscipline.TYPE: 'bool',
            ProxyDiscipline.POSSIBLE_VALUES: [True, False],
            ProxyDiscipline.DEFAULT: False,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'tolerance': {
            ProxyDiscipline.TYPE: 'float',
            ProxyDiscipline.DEFAULT: 1.0e-6,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'use_lu_fact': {
            ProxyDiscipline.TYPE: 'bool',
            ProxyDiscipline.POSSIBLE_VALUES: [True, False],
            ProxyDiscipline.DEFAULT: False,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'warm_start': {
            ProxyDiscipline.TYPE: 'bool',
            ProxyDiscipline.POSSIBLE_VALUES: [True, False],
            ProxyDiscipline.DEFAULT: False,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'acceleration_method': {
            ProxyDiscipline.TYPE: 'string',
            ProxyDiscipline.POSSIBLE_VALUES: list(AccelerationMethod),
            ProxyDiscipline.DEFAULT: AccelerationMethod.ALTERNATE_2_DELTA,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        "scaling_method": {
            ProxyDiscipline.TYPE: 'string',
            ProxyDiscipline.POSSIBLE_VALUES: list(BaseMDA.ResidualScaling),
            ProxyDiscipline.DEFAULT: BaseMDA.ResidualScaling.N_COUPLING_VARIABLES,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        # parallel sub couplings execution
        'n_subcouplings_parallel': {
            ProxyDiscipline.TYPE: 'int',
            ProxyDiscipline.DEFAULT: 1,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'max_mda_iter_gs': {
            ProxyDiscipline.TYPE: 'int',
            ProxyDiscipline.DEFAULT: 30,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'tolerance_gs': {
            ProxyDiscipline.TYPE: 'float',
            ProxyDiscipline.DEFAULT: 10.0,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        'over_relaxation_factor': {
            ProxyDiscipline.TYPE: 'float',
            ProxyDiscipline.RANGE: [0.0, 1.0],
            ProxyDiscipline.DEFAULT: 0.99,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        # NUMERICAL PARAMETERS OUT OF INIT
        'epsilon0': {
            ProxyDiscipline.TYPE: 'float',
            ProxyDiscipline.DEFAULT: 1.0e-6,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        # Linear solver for MD0
        'linear_solver_MDO': {
            ProxyDiscipline.TYPE: 'string',
            # ProxyDiscipline.DEFAULT: 'GMRES',
            # ProxyDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS,
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'linear_solver_MDO_preconditioner': {
            ProxyDiscipline.TYPE: 'string',
            # ProxyDiscipline.DEFAULT: 'None',
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER,
            # ProxyDiscipline.POSSIBLE_VALUES: POSSIBLE_VALUES_PRECONDITIONER,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'linear_solver_MDO_options': {
            ProxyDiscipline.TYPE: 'dict',
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        # Linear solver for MDA
        'linear_solver_MDA': {
            ProxyDiscipline.TYPE: 'string',
            # ProxyDiscipline.DEFAULT: 'GMRES',
            # ProxyDiscipline.POSSIBLE_VALUES: AVAILABLE_LINEAR_SOLVERS,
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'linear_solver_MDA_preconditioner': {
            ProxyDiscipline.TYPE: 'string',
            # ProxyDiscipline.DEFAULT: 'None',
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_PRECONFITIONER,
            # ProxyDiscipline.POSSIBLE_VALUES: POSSIBLE_VALUES_PRECONDITIONER,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        'linear_solver_MDA_options': {
            ProxyDiscipline.TYPE: 'dict',
            ProxyDiscipline.DEFAULT: DEFAULT_LINEAR_SOLVER_OPTIONS,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
            ProxyDiscipline.UNIT: '-',
        },
        # group all disciplines in a MDOChain
        'group_mda_disciplines': {
            ProxyDiscipline.TYPE: 'bool',
            ProxyDiscipline.POSSIBLE_VALUES: [True, False],
            ProxyDiscipline.DEFAULT: False,
            ProxyDiscipline.USER_LEVEL: 3,
            ProxyDiscipline.NUMERICAL: True,
            ProxyDiscipline.STRUCTURING: True,
        },
        # AUTHORIZE_SELF_COUPLED_DISCIPLINES: {ProxyDiscipline.TYPE: 'bool',
        #                                      ProxyDiscipline.POSSIBLE_VALUES: [True, False],
        #                                      ProxyDiscipline.DEFAULT: False,
        #                                      ProxyDiscipline.USER_LEVEL: 3,
        #                                      ProxyDiscipline.STRUCTURING: True}
    }

    DESC_OUT = {
        RESIDUALS_HISTORY: {
            ProxyDiscipline.USER_LEVEL: 3,
            ProxyDiscipline.TYPE: 'dataframe',
            ProxyDiscipline.UNIT: '-',
            ProxyDiscipline.NUMERICAL: True,
        },
        # NORMALIZED_RESIDUAL_NORM: {ProxyDiscipline.USER_LEVEL: 3, ProxyDiscipline.TYPE: 'array',
        #                      ProxyDiscipline.UNIT: '-', ProxyDiscipline.NUMERICAL: True}
    }

    eps0 = 1.0e-6
    has_chart = False
    NUMERICAL_VAR_LIST = list(DESC_IN.keys()) + list(ProxyDisciplineBuilder.NUM_DESC_IN.keys())

    def __init__(self, sos_name, ee, cls_builder=None, associated_namespaces=None):
        """
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[Class]): list of the sub proxy constructors for the recursive build of the process tree [???]
        """
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder  # TODO: Move to ProxyDisciplineBuilder?
        self.discipline_wrapp = None
        self._reload(sos_name, ee, associated_namespaces=associated_namespaces)
        self.logger = self.ee.logger.getChild(self.__class__.__name__)

        self.residuals_dict = {}

        self.linear_solver_MDA = None
        self.linear_solver_settings_MDA = None
        self.linear_solver_tolerance_MDA = None

        self.linear_solver_MDO = None
        self.linear_solver_settings_MDO = None
        self.linear_solver_tolerance_MDO = None

        self._set_dm_disc_info()

        self.discipline_wrapp = DisciplineWrapp(name=sos_name, logger=self.logger.getChild("DisciplineWrapp"))

    def _reload(self, sos_name, ee, associated_namespaces=None):
        """
        Reload ProxyCoupling with corresponding ProxyDiscipline attributes and set is_sos_coupling.

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
        """
        self.is_sos_coupling = True
        ProxyDiscipline._reload(self, sos_name, ee, associated_namespaces=associated_namespaces)
        self.logger = ee.logger.getChild(self.__class__.__name__)

    def _set_dm_cache_map(self):
        """Update cache_map dict in DM with cache, mdo_chain cache, inner_mdas caches and its children recursively"""
        mda_chain = self.discipline_wrapp.discipline

        if mda_chain is not None:
            # store SoSCoupling cache in DM
            self._store_cache_with_hashed_uid(mda_chain)

            # store mdo_chain cache in DM
            self._store_cache_with_hashed_uid(mda_chain.mdo_chain)

            # store sub mdas cache recursively
            for mda in mda_chain.inner_mdas:
                self._set_sub_mda_dm_cache_map(mda)
        else:
            msg = 'Can not build the cache map if the prepare execution has not been run because we need GEMSEO objects'
            raise Exception(msg)
        # store children cache recursively
        for disc in self.proxy_disciplines:
            disc._set_dm_cache_map()

    def _set_sub_mda_dm_cache_map(self, mda):
        """Update cache_map disc in DM with mda cache and its sub_mdas recursively"""
        # store mda cache in DM
        self._store_cache_with_hashed_uid(mda)
        # store sub mda cache recursively
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self._set_sub_mda_dm_cache_map(sub_mda)

    # -- Public methods
    def setup_sos_disciplines(self):
        """
        Set possible values of preconditioner in data manager, according to liner solver MDA/MDO value
        (available preconditioners are different if petsc linear solvers are used)
        And set default value of max_mda_iter_gs according to inner_mda_name
        """
        disc_in = self.get_data_in()
        # set possible values of linear solver MDA preconditioner
        if 'linear_solver_MDA' in disc_in:
            linear_solver_mda = self.get_sosdisc_inputs('linear_solver_MDA')
            if linear_solver_mda.endswith('_PETSC'):
                if getenv("USE_PETSC", "").lower() not in ("true", "1"):
                    msg = (
                        f'Trying to use PETSC linear solver with USE_PETSC environment variable undefined or false, '
                        f'modify linear_solver_MDA option of {self.sos_name} : {linear_solver_mda} or activate '
                        f'USE_PETSC'
                    )
                    raise ValueError(msg)
                disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = self.POSSIBLE_VALUES_PRECONDITIONER
                if (
                    self.get_sosdisc_inputs('linear_solver_MDA_preconditioner')
                    not in disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]
                ):
                    disc_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'gasm'
            else:
                disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES] = ['None', 'ilu']
                if (
                    self.get_sosdisc_inputs('linear_solver_MDA_preconditioner')
                    not in disc_in['linear_solver_MDA_preconditioner'][self.POSSIBLE_VALUES]
                ):
                    disc_in['linear_solver_MDA_preconditioner'][self.VALUE] = 'None'

        # set possible values of linear solver MDO preconditioner
        if 'linear_solver_MDO' in disc_in:
            linear_solver_mdo = self.get_sosdisc_inputs('linear_solver_MDO')
            if linear_solver_mdo.endswith('_PETSC'):
                if getenv("USE_PETSC", "").lower() not in ("true", "1"):
                    msg = (
                        f'Trying to use PETSC linear solver with USE_PETSC environment variable undefined or false, '
                        f'modify linear_solver_MDA option of {self.sos_name} : {linear_solver_mda} or activate '
                        f'USE_PETSC'
                    )
                    raise ValueError(msg)
                disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = self.POSSIBLE_VALUES_PRECONDITIONER
                if (
                    self.get_sosdisc_inputs('linear_solver_MDO_preconditioner')
                    not in disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]
                ):
                    disc_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'gasm'
            else:
                disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES] = ['None', 'ilu']
                if (
                    self.get_sosdisc_inputs('linear_solver_MDO_preconditioner')
                    not in disc_in['linear_solver_MDO_preconditioner'][self.POSSIBLE_VALUES]
                ):
                    disc_in['linear_solver_MDO_preconditioner'][self.VALUE] = 'None'

            # set default value of max_mda_iter_gs
            if 'max_mda_iter_gs' in disc_in and self.get_sosdisc_inputs('inner_mda_name') == 'GSorNewtonMDA':
                self.update_default_value('max_mda_iter_gs', self.IO_TYPE_IN, 200)

    def configure_io(self):
        """
        Configure the ProxyCoupling by :
        - setting the discipline in the discipline_dict
        - configure all children disciplines
        """
        ProxyDiscipline.configure(self)
        if self.ee.wrapping_mode == 'GEMSEO':
            # Use the IO of the gemseo object to populate the coupling inputs and outputs
            self.configure_coupling_with_gemseo_object()

        elif self.ee.wrapping_mode == 'SoSTrades':

            disc_to_configure = self.get_disciplines_to_configure()

            if len(disc_to_configure) > 0:
                self.set_configure_status(False)
                for disc in disc_to_configure:
                    # possibly some one else like the driver has already configured the discipline
                    # not working for multiscenario of architecture builder ...
                    # if not disc.is_configured():
                    disc.configure()
            else:
                self.set_children_numerical_inputs()
                # - all chidren are configured thus proxyCoupling can be configured
                self.set_configure_status(True)
                # - build the coupling structure

                self._build_coupling_structure()
                # - builds data_in/out according to the coupling structure
                self._build_data_io()
                # - Update coupling and editable flags in the datamanager for the GUI
                self._update_coupling_flags_in_dm()

    def configure_coupling_with_gemseo_object(self):
        '''
        Use the IO of the gemseo object to populate the coupling inputs and outputs

        '''

        gemseo_inputs = self.build_gemseo_io(self.IO_TYPE_IN)
        gemseo_outputs = self.build_gemseo_io(self.IO_TYPE_OUT)

        self.update_data_io_and_nsmap(gemseo_inputs, self.IO_TYPE_IN)
        self.update_data_io_and_nsmap(gemseo_outputs, self.IO_TYPE_OUT)

    def build_gemseo_io(self, io_type):
        '''

        Transform the gemseo grammars into understandable grammardict for sostrades
        '''
        gemseo_io_dict = {}
        if isinstance(self.cls_builder, BaseDiscipline):
            gemseo_disc = self.cls_builder
            grammar = self.retrieve_gemseo_grammar(gemseo_disc, io_type)
            gemseo_io_dict = self.convert_gemseo_grammar_into_io_dict(grammar, io_type)
        elif isinstance(self.cls_builder, BaseScenario):
            gemseo_discs = self.cls_builder.disciplines

            for gemseo_disc in gemseo_discs:
                disc_grammar = self.retrieve_gemseo_grammar(gemseo_disc, io_type)
                gemseo_io_dict.update(self.convert_gemseo_grammar_into_io_dict(disc_grammar, io_type))

        return gemseo_io_dict

    def convert_gemseo_grammar_into_io_dict(self, gemseo_grammar, io_type):

        io_dict = {gemseo_key: {self.TYPE: gemseo_type} for
                   gemseo_key, gemseo_type
                   in gemseo_grammar._get_names_to_types().items() if gemseo_type in self.VAR_TYPE_MAP}
        if io_type == self.IO_TYPE_IN:
            # for key in self.cls_builder.io.output_grammar:
            #     if key in gemseo_io_dict:
            #         gemseo_io_dict.pop(key)
            for gemseo_key, gemseo_dict in io_dict.items():
                gemseo_dict[self.DEFAULT] = gemseo_grammar.defaults[gemseo_key]

        return io_dict

    def retrieve_gemseo_grammar(self, disc, io_type):
        '''

        Args:
            disc: the gemseo discipline where ther grammar is
            io_type: the type of the grammar input or output

        Returns: the grammar

        '''
        grammar = None
        if io_type == self.IO_TYPE_IN:
            grammar = disc.io.input_grammar
        elif io_type == self.IO_TYPE_OUT:
            grammar = disc.io.output_grammar

        return grammar

    def _build_data_io(self):
        """
        Build data_in and data_out from sub proxies data_in and out
        we build also data_in_with_full_name and data_out_with_full_name
        to be able to retrieve inputs and outputs with same short name
        in sub proxies
        """
        # - build the data_i/o (sostrades) based on input and output grammar of MDAChain (GEMSEO)
        subprocess_data_in, subprocess_data_out = self.__compute_mdachain_gemseo_based_data_io()
        self._restart_data_io_to_disc_io()
        self.update_data_io_with_child(subprocess_data_in, subprocess_data_out)

    def __compute_mdachain_gemseo_based_data_io(self):
        """Mimics the definition of MDAChain i/o grammar"""
        # - identify i/o grammars like in GEMSEO
        # get discipline structure of the MDAChain
        # e.g, in Sellar : [[Sellar1, Sellar2], SellarProblem]
        disciplines = self._get_mda_structure_as_in_gemseo()
        # build associated i/o grammar
        chain_inputs = []
        chain_outputs = []
        for group in disciplines:
            if isinstance(group, list):
                # if MDA, i.e. group of disciplines (e.g. [Sellar1, Sellar2])
                # we gather all the i/o of the sub-disciplines (MDA-like i/o grammar)
                list_of_data_in = [
                    d.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple=True) for d in group
                ]
                list_of_data_out = [
                    d.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple=True) for d in group
                ]

                mda_inputs, mda_outputs = self.__get_MDA_io(list_of_data_in, list_of_data_out)
                chain_inputs.append(mda_inputs)
                chain_outputs.append(mda_outputs)
            else:
                # if the group is composed of single discipline (e.g. SellarProblem])
                # we add the i/o to the chain i/o
                disc_inputs = group.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple=True)
                disc_outputs = group.get_data_io_with_full_name(self.IO_TYPE_OUT, as_namespaced_tuple=True)
                chain_inputs.append(disc_inputs)
                chain_outputs.append(disc_outputs)

        # compute MDOChain-like i/o grammar
        return self.__get_MDOChain_io(chain_inputs, chain_outputs)

    def _build_coupling_structure(self):
        """Build CouplingStructure"""
        for proxy_disc in self.proxy_disciplines:
            proxy_disc.initialize_gemseo_io()
        self.coupling_structure = CouplingStructure(self.proxy_disciplines)
        self.strong_couplings = filter_variables_to_convert(
            self.ee.dm.convert_data_dict_with_full_name(),
            self.coupling_structure.strong_couplings,
            write_logs=True,
            logger=self.logger,
        )

    def configure(self):
        """Configure i/o, update status, update status in dm."""
        # configure SoSTrades objects
        self.configure_io()

        self._update_status_dm(self.STATUS_CONFIGURE)

    def _update_coupling_flags_in_dm(self):
        """Update coupling and editable flags in the datamanager for the GUI"""

        def update_flags_of_disc(coupling_key, disc_name, in_or_out):
            disc_list = self.dm.get_disciplines_with_name(disc_name)
            var_name_out = None
            for a_disc in disc_list:
                data_io = a_disc.get_data_in() if in_or_out == "in" else a_disc.get_data_out()

                if var_name_k in data_io:
                    var_name_out = var_name_k
                else:
                    var_name_out_list = [
                        key
                        for key, data_dict in data_io.items()
                        if coupling_key.endswith(NS_SEP + data_dict[self.VAR_NAME])
                    ]
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
        for k, from_disc_name, to_disc_name in zip(couplings, disc_1, disc_2):
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
        """
        Export couplings as a csv with
        disc1 | disc2 | var_name
        """
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
                df = concat([df, df_couplings], ignore_index=True)

        if in_csv:
            # writing of the file
            if f_name is None:
                f_name = f"{self.get_disc_full_name()}.csv"
            df.to_csv(f_name, index=False)
        else:
            return df
        return None

    def is_configured(self):
        """Return False if at least one sub discipline needs to be configured, True if not"""
        return (
            self.get_configure_status()
            and not self.check_structuring_variables_changes()
            and (self.get_disciplines_to_configure() == [])
        )

    def prepare_execution(self):
        """Preparation of the GEMSEO process, including GEMSEO objects instanciation"""
        # prepare_execution of proxy_disciplines
        if isinstance(self.cls_builder, ProxyDiscipline.GEMSEO_OBJECTS):
            self.discipline_wrapp.wrapping_mode = 'GEMSEO'

        gemseo_disciplines = self.create_gemseo_disciplines()
        # store cache and n_calls before MDAChain reset, if prepare_execution
        # has already been called
        if self.discipline_wrapp.discipline is not None:
            mda_chain_cache = self.discipline_wrapp.discipline.cache
        else:
            mda_chain_cache = None

        reduced_dm = defaultdict(dict) if self.ee.dm.reduced_dm is None else self.ee.dm.reduced_dm
        # create_mda_chain from DisciplineWrapp
        self.discipline_wrapp.create_mda_chain(gemseo_disciplines, self, reduced_dm=reduced_dm)

        # set cache cache of gemseo object
        if self.discipline_wrapp.wrapping_mode != 'GEMSEO':
            self.set_gemseo_disciplines_caches(mda_chain_cache)

    def create_gemseo_disciplines(self):
        gemseo_disciplines = []
        for disc in self.proxy_disciplines:
            disc.prepare_execution()

            # Exclude non executable proxy Disciplines
            if disc.discipline_wrapp is not None and disc.discipline_wrapp.discipline is not None:
                gemseo_disc = disc.discipline_wrapp.discipline
                gemseo_disciplines.append(gemseo_disc)
        return gemseo_disciplines

    def set_gemseo_disciplines_caches(self, mda_chain_cache):
        """Set cache of MDAChain, MDOChain and sub MDAs"""
        cache_type = self.get_sosdisc_inputs('cache_type')

        # if warm_start then cache_type must be activated
        if self.discipline_wrapp.discipline.settings.warm_start and cache_type == self.discipline_wrapp.discipline.CacheType.NONE:
            cache_type = self.discipline_wrapp.discipline.CacheType.SIMPLE
        # set MDAChain cache
        if self._reset_cache:
            # set new cache when cache_type have changed (self._reset_cache == True)
            # TODO: pass cache to MDAChain init to avoid reset cache
            self.set_cache(self.discipline_wrapp.discipline, cache_type)
            self._reset_cache = False
        else:
            # reset stored cache and n_calls of MDAChain
            self.discipline_wrapp.discipline.cache = mda_chain_cache

        # set cache of MDOChain with cache_type and cache_file_path inputs of
        # ProxyCoupling
        self.set_cache(self.discipline_wrapp.discipline.mdo_chain, cache_type)

        # set epsilon0 and cache of inner_mdas
        for sub_mda in self.discipline_wrapp.discipline.inner_mdas:
            self.set_epsilon0_and_cache(sub_mda)

    def check_var_data_mismatch(self):
        """
        Check if a variable data is not coherent between two coupling disciplines

        The check if a variable that is used in input of multiple disciplines is coherent is made in check_inputs of datamanager
        the list of data_to_check is defined in SoSDiscipline
        """
        self._build_coupling_structure()
        if self.logger.level <= logging.DEBUG:
            coupling_vars = self.coupling_structure.graph.get_disciplines_couplings()
            for from_disc, to_disc, c_vars in coupling_vars:
                for var in c_vars:
                    # from disc is in output
                    from_disc_data = from_disc.get_data_with_full_name(self.IO_TYPE_OUT, var)
                    # to_disc is in input
                    to_disc_data = to_disc.get_data_with_full_name(self.IO_TYPE_IN, var)
                    for data_name in self.DATA_TO_CHECK:
                        # Check if data_names are different
                        if from_disc_data[data_name] != to_disc_data[data_name]:
                            self.logger.debug(
                                f'The {data_name} of the coupling variable {var} is not the same in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}'
                            )
                        # Check if unit is not None
                        elif from_disc_data[data_name] is None and data_name == self.UNIT:
                            # if unit is None in a dataframe check if there is a
                            # dataframe descriptor with unit in it
                            if from_disc_data[self.TYPE] == 'dataframe':
                                # if no dataframe descriptor and no unit
                                # warning
                                if from_disc_data[self.DATAFRAME_DESCRIPTOR] is None:
                                    self.logger.debug(
                                        f'The unit and the dataframe descriptor of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]} : cannot find unit for this dataframe'
                                    )
                            else:
                                self.logger.debug(
                                    f'The unit of the coupling variable {var} is None in input of {to_disc.__class__} : {to_disc_data[data_name]} and in output of {from_disc.__class__} : {from_disc_data[data_name]}'
                                )

    def set_epsilon0_and_cache(self, mda):
        """Set epsilon0 that is not argument of the init of the MDA and need to be set outside of it with MDA attributes"""
        if isinstance(mda, MDASequential):
            for sub_mda in mda.mda_sequence:
                self.set_epsilon0_and_cache(sub_mda)
        mda.epsilon0 = copy(self.get_sosdisc_inputs('epsilon0'))
        self.set_cache(mda, self.get_sosdisc_inputs('cache_type'))

    def _get_numerical_inputs(self):
        """Get numerical parameters input values for MDAChain init"""
        # get input for MDAChain instantiation
        needed_numerical_param = [
            'inner_mda_name',
            'max_mda_iter',
            'chain_linearize',
            'tolerance',
            'use_lu_fact',
            'warm_start',
            'n_processes',
            "scaling_method",
        ]
        num_data = self.get_sosdisc_inputs(needed_numerical_param, in_dict=True)

        if num_data['inner_mda_name'] == 'MDAJacobi':
            num_data['acceleration_method'] = copy(self.get_sosdisc_inputs('acceleration_method'))
        if num_data['inner_mda_name'] in ['MDAGSNewton', 'GSorNewtonMDA']:
            # , 'GSPureNewtonMDA', 'GSPureNewtonorGSMDA'
            num_data['max_mda_iter_gs'] = copy(self.get_sosdisc_inputs('max_mda_iter_gs'))
            num_data['tolerance_gs'] = copy(self.get_sosdisc_inputs('tolerance_gs'))
        if num_data['inner_mda_name'] in self.NEWTON_ALGO_LIST:
            # 'PureNewtonRaphson', 'GSPureNewtonMDA', 'GSPureNewtonorGSMDA'
            num_data['over_relaxation_factor'] = copy(self.get_sosdisc_inputs('over_relaxation_factor'))

        # linear solver options MDA
        num_data['linear_solver'] = copy(self.get_sosdisc_inputs('linear_solver_MDA'))
        linear_solver_settings_MDA = deepcopy(self.get_sosdisc_inputs('linear_solver_MDA_options'))

        if num_data['linear_solver'].endswith('_PETSC'):
            # PETSc case
            preconditioner = copy(self.get_sosdisc_inputs('linear_solver_MDA_preconditioner'))
            linear_solver_settings_MDA['preconditioner_type'] = (preconditioner != 'None') * preconditioner or None
        else:
            # Scipy case / gmres
            linear_solver_settings_MDA['use_ilu_precond'] = (
                copy(self.get_sosdisc_inputs('linear_solver_MDA_preconditioner')) == 'ilu'
            )

        num_data['linear_solver_tolerance'] = linear_solver_settings_MDA.pop('tol')
        num_data['linear_solver_settings'] = linear_solver_settings_MDA

        self.linear_solver_MDA = num_data['linear_solver']
        self.linear_solver_tolerance_MDA = num_data['linear_solver_tolerance']
        self.linear_solver_settings_MDA = deepcopy(num_data['linear_solver_settings'])

        # linear solver options MDO
        self.linear_solver_MDO = self.get_sosdisc_inputs('linear_solver_MDO')
        linear_solver_settings_MDO = deepcopy(self.get_sosdisc_inputs('linear_solver_MDO_options'))

        if self.linear_solver_MDO.endswith('_PETSC'):
            preconditioner = self.get_sosdisc_inputs('linear_solver_MDO_preconditioner')
            linear_solver_settings_MDO['preconditioner_type'] = (preconditioner != 'None') * preconditioner or None
        else:
            linear_solver_settings_MDO['use_ilu_precond'] = (
                self.get_sosdisc_inputs('linear_solver_MDO_preconditioner') == 'ilu'
            )

        self.linear_solver_tolerance_MDO = linear_solver_settings_MDO.pop('tol')
        self.linear_solver_settings_MDO = linear_solver_settings_MDO

        return num_data

    def get_maturity(self):
        """Get the maturity of the coupling proxy by adding all maturities of children proxy disciplines"""
        ref_dict_maturity = deepcopy(self.dict_maturity_ref)
        for discipline in self.proxy_disciplines:
            disc_maturity = discipline.get_maturity()
            if isinstance(disc_maturity, dict):
                for m_k in ref_dict_maturity:
                    if m_k in disc_maturity:
                        ref_dict_maturity[m_k] += disc_maturity[m_k]
            elif disc_maturity in ref_dict_maturity:
                ref_dict_maturity[disc_maturity] += 1
        self.set_maturity(ref_dict_maturity, maturity_dict=True)
        return self._maturity

    def remove_discipline(self, disc):
        """Remove one discipline from coupling"""
        disc.clean_dm_from_disc()
        self.proxy_disciplines.remove(disc)
        self.ee.ns_manager.remove_dependencies_after_disc_deletion(disc, self.disc_id)

    def remove_discipline_list(self, disc_list):
        """Remove several disciplines from coupling"""
        for disc in disc_list:
            self.remove_discipline(disc)

    @property
    def ordered_disc_list(self):
        """
        Property to obtain the ordered list of disciplines configured by the MDAChain
        Overwrite of sos_discipline property where the order is defined by default
        by the order of sos_disciplines
        """
        return self.proxy_disciplines
        #         self.logger.warning(
        #             "TODO: fix the order disc list in proxy coupling (set as the top level list of disciplines for debug purpose)")

    @property
    def is_prepared(self):
        return self.discipline_wrapp.discipline is not None

    def ordered_disc_list_rec(self, disc, ordered_list):
        """Recursive function to obtain the ordered list of disciplines configured by the MDAChain"""
        for subdisc in disc.disciplines:
            if isinstance(subdisc, ProxyDiscipline):
                ordered_list.append(subdisc)
            else:  # Means that it is a GEMS class MDAJacobi for example
                ordered_list = self.ordered_disc_list_rec(subdisc, ordered_list)

        return ordered_list

    def _get_mda_structure_as_in_gemseo(self):
        """Based on GEMSEO create_mdo_chain function, in MDAChain.py
        returns a list of disciplines (weak couplings) or list (strong couplings / MDAs) :
        - if a MDA loop is detected, put disciplines in a sub-list
        - if no MDA loop, the current discipline is added in the main list
        """
        chained_disciplines = []

        for parallel_tasks in self.coupling_structure.sequence:
            for coupled_disciplines in parallel_tasks:
                first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1 or (
                    len(coupled_disciplines) == 1 and self.coupling_structure.is_self_coupled(first_disc)
                    #                     and not coupled_disciplines[0].is_sos_coupling #TODO: replace by "and not isinstance(coupled_disciplines[0], MDA)" as in GEMSEO actual version
                    # and self.get_sosdisc_inputs(self.AUTHORIZE_SELF_COUPLED_DISCIPLINES) #Deactivate AUTHORIZE_SELF_COUPLED_DISCIPLINES which is not correctly implemented regarding strong couplings
                    # option that is never used in SoSTrades applications for now
                ):
                    # - MDA detection
                    # in this case, mda i/o is the union of all i/o (different from MDOChain)
                    # order the MDA disciplines the same way as the
                    # original disciplines
                    # -> works only if the disciplines are built following the same order than proxy ones
                    sub_mda_disciplines = [
                        disc for disc in self.coupling_structure.disciplines if disc in coupled_disciplines
                    ]

                    chained_disciplines.append(sub_mda_disciplines)
                else:
                    # single discipline
                    chained_disciplines.append(first_disc)

        return chained_disciplines

    def __get_MDA_io(self, data_in_list, data_out_list):
        """Returns a tuple of the i/o dict {(local_name, ns ID) : value} (data_io formatting) of provided list of data_io,
        according to GEMSEO rules for MDAs grammar creation :
        MDA input grammar is built as the union of inputs of all sub-disciplines, same for outputs.

        Args:
        data_in_list : list of data_in (one per discipline)
        data_out_list : list of data_out (one per discipline)
        """
        data_in = ChainMap(*data_in_list)  # merge list of dict in 1 dict
        data_out = ChainMap(*data_out_list)
        return data_in, data_out

    def __get_MDOChain_io(self, data_in_list, data_out_list):
        """Returns a tuple of dictionaries (data_in, data_out) of the provided disciplines,
        according to GEMSEO convention for MDOChain grammar creation :

        Args:
        data_in_list : list of data_in (one per discipline)
        data_out_list : list of data_out (one per discipline)
        """
        mdo_inputs = {}
        mdo_outputs = {}

        for d_in, d_out in zip(data_in_list, data_out_list):
            # add discipline input tuple (name, id) if tuple not already in outputs
            mdo_inputs.update({t: v for (t, v) in d_in.items() if t not in mdo_outputs})
            # add discipline output name in outputs
            mdo_outputs.update(d_out)

        return mdo_inputs, mdo_outputs

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = [self.RESIDUALS_HISTORY_PRETTY]

        chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        instanciated_charts = []
        # Overload default value with chart filter
        select_all = False
        chart_list = []
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values
        else:
            select_all = True

        post_processing_mda_data = self.get_sosdisc_outputs(self.RESIDUALS_HISTORY)

        # TODO: utility function consider moving to tool?
        def to_series(varname: str, x: list, y: ndarray) -> list[InstanciatedSeries]:
            dim = y.shape[1]
            series = []
            for d in range(dim):
                series_name = varname if dim == 1 else rf"{varname}\{d}"
                new_series = InstanciatedSeries(x, list(y[:, d]), series_name, 'lines', True)
                series.append(new_series)
            return series

        if select_all or self.RESIDUALS_HISTORY_PRETTY in chart_list:
            inner_mda_name = self.get_sosdisc_inputs('inner_mda_name')
            if post_processing_mda_data is not None and inner_mda_name in post_processing_mda_data.columns:
                residuals_through_iterations = np.asarray([[x[0]] for x in post_processing_mda_data[inner_mda_name]])
                iterations = list(range(len(residuals_through_iterations)))
                chart_name = self.RESIDUALS_HISTORY_PRETTY

                new_chart = TwoAxesInstanciatedChart('Iterations', 'Residuals', chart_name=chart_name, y_axis_log=True)

                if iterations:  # TODO: quickfix to avoid post-proc crash when coupling is cached, will give empty plot
                    for series in to_series(varname="Residuals", x=iterations, y=residuals_through_iterations):
                        new_chart.series.append(series)

                instanciated_charts.append(new_chart)

        return instanciated_charts
