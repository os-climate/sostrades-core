'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/06-2023/11/03 Copyright 2023 Capgemini

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
from typing import List
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import deepcopy
from multiprocessing import cpu_count

import pandas as pd
from numpy import array, ndarray, delete, inf

from gemseo.algos.design_space import DesignSpace
from gemseo.core.scenario import Scenario
from gemseo.core.function import MDOFunction
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.core.jacobian_assembly import JacobianAssembly
from sostrades_core.execution_engine.data_manager import POSSIBLE_VALUES
from sostrades_core.execution_engine.ns_manager import NS_SEP, NamespaceManager
from sostrades_core.execution_engine.mdo_discipline_wrapp import MDODisciplineWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.design_space import design_space as dspace_tool


class ProxyOptim(ProxyDriverEvaluator):
    """
        **ProxyOptim** is a class proxy for an optim on the SoSTrades side.

        It contains the information and methonds necessary for i/o configuration (static or dynamic).



        Leaves of the process tree are direct instances of ProxyDiscipline. Other nodes are instances that inherit from
        ProxyDiscipline (e.g. ProxyCoupling).

        An instance of ProxyDiscipline is in one-to-one aggregation with an instance of MDODisciplineWrapp, which allows the
        use of different wrapping modes to provide the model run.

        During the prepare_execution step, the ProxyDiscipline coordinates the instantiation of the GEMSEO objects that
        manage the model run.

        Attributes:
            mdo_discipline_wrapp (MDODisciplineWrapp): aggregated object that references the wrapper and GEMSEO discipline

            proxy_disciplines (List[ProxyDiscipline]): children in the process tree
            status (property,<<associated with string _status>>): status in the current process,either CONFIGURATION or
             provided by the GEMSEO objects during run

            disc_id (string): anonymized discipline identifier in the data manager
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the process
            dm (DataManager): data manager of the process


            is_sos_coupling (bool): type of node flag
            is_optim_scenario (bool): type of node flag
            is_parallel (bool): type of node flag
            is_specific_driver (bool): type of node flag

            _is_configured (bool): flag for configuration relaying on children configuration and structuring vars changes
            _reset_cache (bool): flag to reset cache

            inst_desc_in (Dict[Dict]): desc_in of instance used to add dynamic inputs
            inst_desc_out (Dict[Dict]): desc_out of instance used to add dynamic outputs
            _data_in (Dict[Dict]): instance variable for input data handling containing description of variables in disc and subprocess
            _data_out (Dict[Dict]): instance variable for output data handling containing description of variables in disc and subprocess

            _io_ns_map_in(Dict[int]): map of short names to namespace object id of discipline DESC_IN+NUM_DESC_IN+inst_desc_in
            _io_ns_map_out(Dict[int]): map of short names to namespace object id of discipline DESC_OUT+inst_desc_out

            _structuring_variables (Dict[Any]): stored values of variables whose changes force revert of the configured status
            _maturity (string): maturity of the user-defined model


            cls (Class): constructor of the model wrapper with user-defin ed run (or None)
        """
    # Default values of algorithms

    # ontology information
    _ontology_data = {
        'label': ' Optimization Driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    default_algo_options = {"ftol_rel": 3e-16,
                            "normalize_design_space": True,
                            "maxls": 100,
                            "maxcor": 50,
                            "pg_tol": 1.e-8,
                            "max_iter": 500,
                            "disp": 30}
    default_parallel_options = {'parallel': False,
                                'n_processes': cpu_count(),
                                'use_threading': False,
                                'wait_time_between_fork': 0}
    USER_GRAD = 'user'
    # Design space dataframe headers
    TYPE = "type"
    VARIABLES = dspace_tool.VARIABLES
    VALUES = dspace_tool.VALUES
    UPPER_BOUND = dspace_tool.UPPER_BOUND
    LOWER_BOUND = dspace_tool.LOWER_BOUND
    ENABLE_VARIABLE_BOOL = dspace_tool.ENABLE_VARIABLE_BOOL
    LIST_ACTIVATED_ELEM = dspace_tool.LIST_ACTIVATED_ELEM
    VARIABLE_TYPE = dspace_tool.VARIABLE_TYPE
    ALGO = "algo"
    MAX_ITER = "max_iter"
    ALGO_OPTIONS = "algo_options"
    FINITE_DIFFERENCES = "finite_differences"
    COMPLEX_STEP = "complex_step"
    APPROX_MODES = [FINITE_DIFFERENCES, COMPLEX_STEP]
    AVAILABLE_MODES = (
        JacobianAssembly.AUTO_MODE,
        JacobianAssembly.DIRECT_MODE,
        JacobianAssembly.ADJOINT_MODE,
        JacobianAssembly.REVERSE_MODE,
        FINITE_DIFFERENCES,
        COMPLEX_STEP,
    )
    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'
    # FD step
    FD_STEP = "fd_step"

    DESIGN_SPACE = dspace_tool.DESIGN_SPACE
    FORMULATION = 'formulation'
    MAXIMIZE_OBJECTIVE = 'maximize_objective'
    OBJECTIVE_NAME = 'objective_name'
    FORMULATION_OPTIONS = 'formulation_options'

    #        self.SEARCH_PATHS = 'search_paths'
    SCENARIO_MANDATORY_FIELDS = [
        DESIGN_SPACE,
        FORMULATION,
        MAXIMIZE_OBJECTIVE,
        OBJECTIVE_NAME]
    #            self.SEARCH_PATHS]
    OPTIMAL_OBJNAME_SUFFIX = "opt"
    ALGO_MANDATORY_FIELDS = [ALGO, MAX_ITER]

    DIFFERENTIATION_METHOD = 'differentiation_method'
    EVAL_JAC = 'eval_jac'
    EVAL_MODE = 'eval_mode'
    EXECUTE_AT_XOPT = 'execute_at_xopt'

    default_algo_options = {'max_iter': 999, 'ftol_rel': 1e-9,
                            'ftol_abs': 1e-9, 'xtol_rel': 1e-9,
                            'xtol_abs': 1e-9, 'max_ls_step_size': 0.,
                            'max_ls_step_nb': 20, 'max_fun_eval': 999999, 'max_time': 0,
                            'pg_tol': 1e-5, 'disp': 0, 'maxCGit': -1, 'eta': -1.,
                            'factr': 1e7, 'maxcor': 20, 'normalize_design_space': True,
                            'eq_tolerance': 1e-2, 'ineq_tolerance': 1e-4,
                            'stepmx': 0., 'minfev': 0., 'sigma': 10.0, 'bounds': [0.0, 10.0], 'population_size': 20}

    default_algo_options_plbfgsb = {'max_iter': 999, 'ftol_rel': 1e-9,
                                    'ftol_abs': 1e-9, 'xtol_rel': 1e-9,
                                    'xtol_abs': 1e-9, 'max_ls_step_size': 0.,
                                    'max_ls_step_nb': 20, 'max_fun_eval': 999999, 'max_time': 0,
                                    'pg_tol': 1e-5, 'disp': 0, 'maxCGit': -1, 'eta': -1.,
                                    'factr': 1e7, 'maxcor': 20, 'normalize_design_space': True,
                                    'eq_tolerance': 1e-2, 'ineq_tolerance': 1e-4,
                                    'stepmx': 0., 'minfev': 0., 'linesearch': 'lnsrlb', 'lnsrlb_xtol': 0.1,
                                    'projection': 'proj_bound', 'func_target': None, 'ln_step_init': 1.0,
                                    'max_ln_step': 1e99,
                                    'lmem': 10, 'precond': None, 'precond_file': None, 'use_cauchy_linesearch': None,
                                    'zero_tol': 1.0e-15,
                                    'primal_epsilon': 1e-10, 'bound_tol': 1e-10, 'gcp_precond_space': None,
                                    'lnsrlb_max_fg_calls': 21, 'lnsrlb_stpmin': 0.0,
                                    'lnsrlb_ftol': 1e-3, 'lnsrlb_gtol': 0.9, 'lnsrlb_xtrapl': 1.1, 'lnsrlb_xtrapu': 4.0,
                                    'unfeas_comp_exeption': None, 'epsmch': 1e-16}

    default_algo_options_nlopt = {'ftol_abs': 1e-14,
                                  'xtol_abs': 1e-14, 'max_iter': 999,
                                  'ftol_rel': 1e-8, 'xtol_rel': 1e-8, 'max_time': 0., 'ctol_abs': 1e-6,
                                  'stopval': None, 'normalize_design_space': True,
                                  'eq_tolerance': 1e-2, 'ineq_tolerance': 1e-4, 'init_step': 0.25}

    default_algo_options_openopt = {'max_iter': 999,  # pylint: disable=W0221
                                    'ftol_abs': 1e-12, 'xtol_abs': 1e-12, 'iprint': 1000,
                                    'max_time': float("inf"), 'max_cpu_time': float("inf"),
                                    'max_ls_step_nb': 500, 'max_fun_eval': 100000,
                                    'normalize_design_space': True, 'eq_tolerance': 1e-2,
                                    'ineq_tolerance': 1e-4, 'scale': None, 'pg_tol': 0.0}

    default_algo_options_oa = {'max_iter': 999,  # pylint: disable=W0221
                               'ftol_abs': 1e-12,
                               'algo_options_MILP': {},
                               'algo_options_NLP': default_algo_options_nlopt,
                               'algo_NLP': 'SLSQP',
                               'normalize_design_space': False,
                               }

    algo_dict = {"NLOPT": default_algo_options_nlopt,
                 "OPENOPT": default_algo_options_openopt,
                 "P-L-BFGS-B": default_algo_options_plbfgsb,
                 "OuterApproximation": default_algo_options_oa,
                 "PYMOO_GA": {"normalize_design_space": False},
                 "PYMOO_NSGA2": {"normalize_design_space": False},
                 "PYMOO_NSGA3": {"normalize_design_space": False, "ref_dirs_name": "energy"},
                 "PYMOO_UNSGA3": {"normalize_design_space": False, "ref_dirs_name": "energy"},
                 }

    DESC_IN = {ALGO: {'type': 'string', 'structuring': True},
               DESIGN_SPACE: {'type': 'dataframe', 'structuring': True,
                                'dataframe_descriptor': {VARIABLES: ('string', None, True),
                                                         VALUES: ('multiple', None, True),
                                                         LOWER_BOUND: ('multiple', None, True),
                                                         UPPER_BOUND: ('multiple', None, True),
                                                         ENABLE_VARIABLE_BOOL: ('bool', None, True),
                                                         LIST_ACTIVATED_ELEM: ('list', None, True), }},

               FORMULATION: {'type': 'string', 'structuring': True},
               MAXIMIZE_OBJECTIVE: {'type': 'bool', 'structuring': True, 'default': False},
               OBJECTIVE_NAME: {'type': 'string', 'structuring': True},
               DIFFERENTIATION_METHOD: {'type': 'string', 'default': Scenario.FINITE_DIFFERENCES,
                                          'possible_values': [USER_GRAD, Scenario.FINITE_DIFFERENCES,
                                                              Scenario.COMPLEX_STEP],
                                          'structuring': True},
               FD_STEP: {'type': 'float', 'structuring': True, 'default': 1e-6},
               ALGO_OPTIONS: {'type': 'dict', 'dataframe_descriptor': {VARIABLES: ('string', None, False),
                                                                         VALUES: ('string', None, True)},
                                'dataframe_edition_locked': False,
                                'default': default_algo_options,
                                'structuring': True},
               PARALLEL_OPTIONS: {'type': 'dict',  # SoSDisciplineBuilder.OPTIONAL: True,
                                  'dataframe_descriptor': {VARIABLES: ('string', None, False),  # bool
                                                           VALUES: ('string', None, True)},
                                  #                                   'dataframe_descriptor': {'parallel': ('int', None, True), #bool
                                  #                                                          'n_processes': ('int', None, True),
                                  #                                                          'use_threading': ('int', None, True),#bool
                                  #                                                          'wait_time_between_fork': ('int', None, True)},
                                  'dataframe_edition_locked': False,
                                  'default': default_parallel_options,
                                  'structuring': True},
               EVAL_MODE: {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False], 'structuring': True},
               EVAL_JAC: {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False]},
               EXECUTE_AT_XOPT: {'type': 'bool', 'default': True},
               MAX_ITER: {'type': 'float'},
               INEQ_CONSTRAINTS: {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                  'structuring': True},
               EQ_CONSTRAINTS: {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                'structuring': True},
               }

    DESC_OUT = {'design_space_out': {'type': 'dataframe'},
                'post_processing_mdo_data': {'type': 'dict'}}

    def __init__(self, sos_name, ee, cls_builder, with_data_io=True, associated_namespaces=None):
        """
        Constructor
        """
        super().__init__(sos_name, ee, cls_builder, associated_namespaces=associated_namespaces)
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder
        self.mdo_discipline_wrapp = None

        self.with_data_io = with_data_io
        self.formulation = None
        self.maximize_objective = None
        self.algo_name = None
        self.algo_options = None
        self.max_iter = None

        self.objective_name = None
        self.design_space = None

        self.opt_problem = None
        self.eval_mode = False
        self.eval_jac = False

        self.dict_desactivated_elem = {}
        self.activated_variables = []
        self.is_optim_scenario = True
        self.functions_before_run = []

        self.mdo_discipline_wrapp = MDODisciplineWrapp(name=sos_name, logger=self.logger.getChild("MDODisciplineWrapp"))

        self.check_integrity_msg_list = []
        self.opt_data_integrity = True

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        """
        if self.ALGO_OPTIONS in self.get_sosdisc_inputs().keys():
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            if algo_name is not None:
                default_dict = self.get_algo_options(algo_name)
                if algo_options is not None:
                    values_dict = deepcopy(default_dict)
                    for k in algo_options.keys():
                        if algo_options[k] is not None and algo_options[k] != 'None':
                            values_dict.update({k: algo_options[k]})
                    self.inst_desc_in[self.ALGO_OPTIONS] = values_dict
                    for key in self._data_in.keys():
                        if self.ALGO_OPTIONS == key[0]:
                            self._data_in[key][self.VALUE] = values_dict
        self.set_edition_inputs_if_eval_mode()

    def prepare_build(self):
        """
        To be overload by subclasses with special builds.
        """
        if not isinstance(self.cls_builder, list):
            builder_list = [self.cls_builder]
        else:
            builder_list = self.cls_builder
        return builder_list

    def set_edition_inputs_if_eval_mode(self):
        '''
        if eval mode then algo and algo options will turn to not editable
        '''

        if 'eval_mode' in [key[0] for key in self._data_in.keys()]:
            eval_mode = self.get_sosdisc_inputs(self.EVAL_MODE)
            if eval_mode:
                data_in = self.get_data_in()
                self.eval_mode = True
                self.eval_jac = self.get_sosdisc_inputs(self.EVAL_JAC)
                self._data_in[(self.ALGO, id(data_in[self.ALGO][self.NS_REFERENCE]))][self.EDITABLE] = False
                self._data_in[(self.ALGO_OPTIONS, id(data_in[self.ALGO_OPTIONS][self.NS_REFERENCE]))][
                    self.EDITABLE] = False
                self._data_in[(self.FORMULATION, id(data_in[self.FORMULATION][self.NS_REFERENCE]))][
                    self.EDITABLE] = False
                self._data_in[(self.MAXIMIZE_OBJECTIVE, id(data_in[self.MAXIMIZE_OBJECTIVE][self.NS_REFERENCE]))][
                    self.EDITABLE] = False
                self._data_in[(self.PARALLEL_OPTIONS, id(data_in[self.PARALLEL_OPTIONS][self.NS_REFERENCE]))][
                    self.EDITABLE] = False
                self._data_in[(self.MAX_ITER, id(data_in[self.MAX_ITER][self.NS_REFERENCE]))][self.EDITABLE] = False

                self._data_in[(self.ALGO, id(data_in[self.ALGO][self.NS_REFERENCE]))][self.OPTIONAL] = True
                self._data_in[(self.ALGO_OPTIONS, id(data_in[self.ALGO_OPTIONS][self.NS_REFERENCE]))][
                    self.OPTIONAL] = True
                self._data_in[(self.FORMULATION, id(data_in[self.FORMULATION][self.NS_REFERENCE]))][
                    self.OPTIONAL] = True
                self._data_in[(self.MAXIMIZE_OBJECTIVE, id(data_in[self.MAXIMIZE_OBJECTIVE][self.NS_REFERENCE]))][
                    self.OPTIONAL] = True
                self._data_in[(self.PARALLEL_OPTIONS, id(data_in[self.PARALLEL_OPTIONS][self.NS_REFERENCE]))][
                    self.OPTIONAL] = True
                self._data_in[(self.MAX_ITER, id(data_in[self.MAX_ITER][self.NS_REFERENCE]))][self.OPTIONAL] = True
            else:
                self.eval_jac = False

    def prepare_execution(self):
        '''
        Preparation of the GEMSEO process, including GEMSEO objects instanciation
        '''
        self.algo_name, self.algo_options, self.max_iter = self.get_sosdisc_inputs([self.ALGO,
                                                                                    self.ALGO_OPTIONS,
                                                                                    self.MAX_ITER])
        self.formulation, self.objective_name, self.design_space, self.maximize_objective = self.pre_set_scenario()

        # prepare_execution of proxy_disciplines and extract GEMSEO objects
        sub_mdo_disciplines = []
        for disc in self.proxy_disciplines:
            disc.prepare_execution()
            # Exclude non executable proxy Disciplines
            if disc.mdo_discipline_wrapp is not None:
                sub_mdo_disciplines.append(
                    disc.mdo_discipline_wrapp.mdo_discipline)

        # create_mdo_scenario from MDODisciplineWrapp
        self.mdo_discipline_wrapp.create_mdo_scenario(sub_mdo_disciplines, proxy=self, reduced_dm=self.ee.dm.reduced_dm)
        self.set_constraints()
        self.set_diff_method()
        self.set_design_space_for_complex_step()
        self.set_parallel_options()

        self.set_formulation_for_func_manager(sub_mdo_disciplines)

        # update MDA flag to flush residuals between each mda run
        self._set_flush_submdas_to_true()

    def set_formulation_for_func_manager(self, sub_mdo_disciplines):
        """

        If a func manager exists in the coupling then
        we associate the formulation of the optim in order to retrieve current iter in the func_manager

        """
        # formulation can be found in the GEMSEO mdo_discipline
        formulation = self.mdo_discipline_wrapp.mdo_discipline.formulation

        # Check that only 1 discipline is below the proxy optim
        if len(sub_mdo_disciplines) == 1:
            coupling = sub_mdo_disciplines[0]
            # gather all disciplines under the coupling that are FunctionManagerDisc disicpline
            func_manager_list = [disc.sos_wrapp for disc in coupling.sos_disciplines if
                                 isinstance(disc.sos_wrapp, FunctionManagerDisc)]
            # Normally only one FunctionManagerDisc should be under the optim
            # if multiple do nothing
            if len(func_manager_list) == 1:
                func_manager = func_manager_list[0]
                func_manager.set_optim_formulation(formulation)

    def pre_set_scenario(self):
        """
        prepare the set GEMS set_scenario method
        """
        design_space = None
        formulation = None
        obj_full_name = None
        maximize_objective = False
        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        if dspace is not None:
            # build design space
            design_space = self.set_design_space()
            if design_space.variables_names:
                _, formulation, maximize_objective, obj_name = self.get_sosdisc_inputs(
                    self.SCENARIO_MANDATORY_FIELDS)

                # get full objective ids
                obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
                obj_full_name = self._update_names([obj_name], self.IO_TYPE_OUT)[0]

        return formulation, obj_full_name, design_space, maximize_objective

    def set_design_space(self) -> DesignSpace:
        """
        reads design space (set_design_space)
        """
        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE).copy()
        dspace_df[self.VARIABLES] = self._update_names(dspace_df[self.VARIABLES], self.IO_TYPE_IN)
        design_space, self.dict_desactivated_elem = dspace_tool.create_gemseo_dspace_from_dspace_df(dspace_df)
        return design_space

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = ['Fitness function',
                      'Design variables']

        post_processing_mdo_data = self.get_sosdisc_outputs("post_processing_mdo_data")

        if len(post_processing_mdo_data["constraints"]) > 0:
            chart_list.append("Constraints variables")

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        instanciated_charts = []
        # Overload default value with chart filter
        # Overload default value with chart filter
        chart_list = []
        select_all = False
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values
        else:
            select_all = True

        post_processing_mdo_data = self.get_sosdisc_outputs("post_processing_mdo_data")

        def to_series(varname: str, x: List, y: ndarray) -> List[InstanciatedSeries]:
            dim = y.shape[1]
            series = []
            for d in range(dim):
                series_name = varname if dim == 1 else f"{varname}[{d}]"
                new_series = InstanciatedSeries(
                    x, list(y[:, d]),
                    series_name, 'lines', True)
                series.append(new_series)
            return series

        if select_all or "Fitness function" in chart_list:
            fitness_func_through_iterations = post_processing_mdo_data["objective"]
            iterations = list(range(len(fitness_func_through_iterations)))

            chart_name = 'Objective function optimization'

            new_chart = TwoAxesInstanciatedChart('Iterations', 'Fitness function',
                                                 chart_name=chart_name)

            for series in to_series(
                    varname="Fitness function", x=iterations, y=fitness_func_through_iterations):
                new_chart.series.append(series)

            instanciated_charts.append(new_chart)

        if select_all or 'Design variables' in chart_list:
            for variable_name, history_values_variable in post_processing_mdo_data['variables'].items():
                dim_var = history_values_variable.shape[1]
                shortened_var_name = '.'.join(variable_name.split('.')[2:])
                chart_name = f"Design var '{shortened_var_name}' evolution"
                iterations = list(range(history_values_variable.shape[0]))
                new_chart = TwoAxesInstanciatedChart('Iterations', "Value", chart_name=chart_name)
                for i in range(dim_var):
                    new_series = InstanciatedSeries(
                        iterations, list(history_values_variable[:, i]),
                        f"{shortened_var_name}[{i}]", 'lines', True)
                    new_chart.add_series(new_series)
                instanciated_charts.append(new_chart)

        if len(post_processing_mdo_data["constraints"]) > 0 and (select_all or 'Constraints variables' in chart_list):
            dict_variables_history = post_processing_mdo_data["constraints"]
            min_y, max_y = inf, - inf
            all_series = []
            for variable_name, history in dict_variables_history.items():

                iterations = list(range(len(history)))
                min_value, max_value = history.min(), history.max()
                if max_value > max_y: max_y = max_value
                if min_value < min_y: min_y = min_value
                for series in to_series(varname=variable_name, x=iterations, y=history):
                    all_series.append(series)

            chart_name = 'Constraints variables evolution'
            new_chart = TwoAxesInstanciatedChart('Iterations', 'Constraints variables',
                                                 [min(iterations), max(iterations)], [
                                                     min_y - (max_y - min_y) * 0.1
                                                     , max_y + (max_y - min_y) * 0.1],
                                                 chart_name)
            for series in all_series:
                new_chart.series.append(series)
            instanciated_charts.append(new_chart)

        return instanciated_charts

    def set_design_space_for_complex_step(self):
        '''
        Set design space values to complex if the differentiation method is complex_step
        '''
        diff_method = self.get_sosdisc_inputs(self.DIFFERENTIATION_METHOD)
        if diff_method == self.COMPLEX_STEP:
            dspace = deepcopy(self.mdo_discipline_wrapp.mdo_discipline.formulation.opt_problem.design_space)
            curr_x = dspace._current_x
            for var in curr_x:
                curr_x[var] = curr_x[var].astype('complex128')
            self.mdo_discipline_wrapp.mdo_discipline.formulation.opt_problem.design_space = dspace

    def get_algo_options(self, algo_name):
        """
        Create default dict for algo options
        :param algo_name: the name of the algorithm
        :returns: dictionary with algo options default values
        """
        # TODO : add warning and log algo options

        default_dict = {}
        driver_lib = OptimizersFactory().create(algo_name)
        driver_lib.init_options_grammar(algo_name)
        schema_dict = driver_lib.opt_grammar.schema.to_dict()
        properties = schema_dict.get(driver_lib.opt_grammar.PROPERTIES_FIELD)
        algo_options_keys = list(properties.keys())

        found_algo_names = [
            key for key in self.algo_dict.keys() if key in algo_name]
        if len(found_algo_names) == 1:
            key = found_algo_names[0]
            for algo_option in algo_options_keys:
                default_val = self.algo_dict[key].get(algo_option)
                if default_val is not None:
                    default_dict[algo_option] = default_val
        else:
            for algo_option in algo_options_keys:
                if algo_option in self.default_algo_options:
                    algo_default_val = self.default_algo_options.get(algo_option)
                    if algo_default_val is not None:
                        default_dict[algo_option] = algo_default_val

        return default_dict

    def configure_driver(self):
        """
        Specific configuration actions for the optimisation driver do be done after the subprocess is configured:
        extraction of the subprocess inputs and outputs, GEMSEO algorithms and formulations.
        """
        self.set_eval_possible_values(strip_first_ns=False)

        # TODO: with the short name logic we cannot check directly the data integrity w/ POSSIBLE VALUES, reminder
        #  to re-activate possible values setting when short name logic is abolished
        # # Fill the possible_values of obj and constraints
        # self.dm.set_data(f'{self.get_disc_full_name()}.{self.OBJECTIVE_NAME}',
        #                  self.POSSIBLE_VALUES, self.eval_out_possible_values)
        # if self.is_constraints:
        #     self.dm.set_data(f'{self.get_disc_full_name()}.{self.INEQ_CONSTRAINTS}',
        #                      self.POSSIBLE_VALUES, self.eval_out_possible_values)
        #     self.dm.set_data(f'{self.get_disc_full_name()}.{self.EQ_CONSTRAINTS}',
        #                      self.POSSIBLE_VALUES, self.eval_out_possible_values)

        # fill the possible values of algos
        _algo_factory = OptimizersFactory()
        avail_algos = _algo_factory.algorithms
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.ALGO}', self.POSSIBLE_VALUES, avail_algos)

        # fill the possible values of formulations
        _form_factory = MDOFormulationsFactory()
        avail_formulations = _form_factory.formulations
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.FORMULATION}', self.POSSIBLE_VALUES, avail_formulations)

        # fill the possible values of maximize_objective
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.MAXIMIZE_OBJECTIVE}',  self.POSSIBLE_VALUES, [False, True])

    def _update_eval_output_with_possible_out_values(self, possible_out_values, disc_in):
        pass

    def set_diff_method(self):
        """
        Set differentiation method and send a WARNING
        if some linearization_mode are not coherent with diff_method
        """
        diff_method = self.get_sosdisc_inputs('differentiation_method')

        if diff_method in self.APPROX_MODES:
            for disc in self.proxy_disciplines:
                if disc.linearization_mode != diff_method:
                    self.logger.warning(
                        f'The differentiation method "{diff_method}" will overload the linearization mode "{disc.linearization_mode}" ')

        fd_step = self.get_sosdisc_inputs(self.FD_STEP)
        self.mdo_discipline_wrapp.mdo_discipline.set_differentiation_method(diff_method, fd_step)

    def set_parallel_options(self):
        """
        sets parallel options for jacobian approximation
        """

        # update default options with user options
        user_options = self.get_sosdisc_inputs(self.PARALLEL_OPTIONS)
        if user_options is None:
            user_options = {}
        options = deepcopy(self.default_parallel_options)
        options.update(user_options)
        parallel = options.pop("parallel")
        # update problem options
        self.mdo_discipline_wrapp.mdo_discipline.formulation.opt_problem.parallel_differentiation = parallel
        self.mdo_discipline_wrapp.mdo_discipline.formulation.opt_problem.parallel_differentiation_options = options

    def set_constraints(self):
        # -- inequality constraints
        # NB: ineq constraint definition as [(var_name, cstr_sign), ...] deprecated
        ineq_names = self.get_sosdisc_inputs(self.INEQ_CONSTRAINTS)
        is_positive = [False for _ in ineq_names]
        ineq_full_names = self._update_names(ineq_names, self.IO_TYPE_OUT)
        for ineq, is_pos in zip(ineq_full_names, is_positive):
            self.mdo_discipline_wrapp.mdo_discipline.add_constraint(
                ineq, MDOFunction.TYPE_INEQ, ineq, value=None, positive=is_pos)

        # -- equality constraints
        eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
        eq_full_names = self._update_names(eq_names, self.IO_TYPE_OUT)
        for eq in eq_full_names:
            self.mdo_discipline_wrapp.mdo_discipline.add_constraint(
                self, eq, MDOFunction.TYPE_EQ, eq, value=None,
                positive=False)

    def _set_flush_submdas_to_true(self):
        # update MDA flag to flush residuals between each mda run
        for disc in self.proxy_disciplines:
            if disc.is_sos_coupling:
                if len(disc.mdo_discipline_wrapp.mdo_discipline.sub_mda_list) > 0:
                    for sub_mda in disc.mdo_discipline_wrapp.mdo_discipline.sub_mda_list:
                        sub_mda.reset_history_each_run = True

    def __str__(self):
        """
        Summarize results for display

        :returns: string summarizing results
        """
        msg = ""
        if hasattr(self, "disciplines"):
            msg = self.__class__.__name__ + ":\nDisciplines: "
            disc_names = [disc.name
                          for disc in self.disciplines]  # pylint: disable=E1101
            msg += " ".join(disc_names)
            msg += "\nMDOFormulation: "  # We keep MDO here has is done in gemseo
            msg += self.formulation.__class__.__name__
            msg += "\nAlgorithm: "
            msg += str(self.get_sosdisc_inputs(self.ALGO)) + "\n"
        return msg

    def check_data_integrity(self):
        """
        Specific check data integrity of the ProxyOptim.
        """
        super().check_data_integrity()

        self.opt_data_integrity = True
        disc_in = self.get_data_in()

        # DESIGN SPACE -------------------------------------------------------------------------
        design_space_integrity_msg = []
        if self.DESIGN_SPACE in disc_in:
            design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
            if design_space is not None:
                # TODO: no check of possible values because of the short name, specific check below that needs to be
                #  changed if changing name rule
                design_space_integrity_msg = dspace_tool.check_design_space_data_integrity(design_space,
                                                                                           possible_variables_types=None)
                # specific check for the short names
                var_names = design_space[self.VARIABLES].tolist()
                _, out_errors = self._get_subprocess_var_names(var_names,
                                                               io_type=self.IO_TYPE_IN)
                design_space_integrity_msg.extend(out_errors)

                # type checks based on design space value  # TODO: type checks based on DM
                for var_name, var_value in zip(var_names, design_space[self.VALUE].tolist()):
                    var_type = type(var_value).__name__
                    if var_type not in ['array', 'list', 'ndarray']:
                        design_space_integrity_msg.append(
                            f"A design variable must obligatory be an array, {var_name} is of type {var_type}.")

        if design_space_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.DESIGN_SPACE, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(design_space_integrity_msg))
        # OBJECTIVE ---------------------------------------------------------------------------
        obj_integrity_msg = []
        if self.OBJECTIVE_NAME in disc_in:
            obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
            if obj_name is not None:
                # specific checks that the objective short name can be identified
                _, obj_out_err = self._get_subprocess_var_names([obj_name],
                                                                io_type=self.IO_TYPE_OUT)
                obj_integrity_msg.extend(obj_out_err)
        if obj_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.OBJECTIVE_NAME, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(obj_integrity_msg))
        # INEQ CONSTRAINTS --------------------------------------------------------------------
        ineq_integrity_msg = []
        if self.INEQ_CONSTRAINTS in disc_in:
            ineq_names = self.get_sosdisc_inputs(self.INEQ_CONSTRAINTS)
            # specific checks that the short names can be identified
            _, ineq_out_err = self._get_subprocess_var_names(ineq_names,
                                                             io_type=self.IO_TYPE_OUT)
            ineq_integrity_msg.extend(ineq_out_err)
        if ineq_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.INEQ_CONSTRAINTS, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(ineq_integrity_msg))
        # EQ CONSTRAINTS ----------------------------------------------------------------------
        eq_integrity_msg = []
        if self.INEQ_CONSTRAINTS in disc_in:
            eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
            # specific checks that the short names can be identified
            _, eq_out_err = self._get_subprocess_var_names(eq_names,
                                                           io_type=self.IO_TYPE_OUT)
            eq_integrity_msg.extend(eq_out_err)
        if eq_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(self.get_var_full_name(self.EQ_CONSTRAINTS, disc_in),
                                self.CHECK_INTEGRITY_MSG, '\n'.join(eq_integrity_msg))

    def _get_subprocess_var_names(self, var_names, io_type):
        """
        Method that searches for a list of variable names inside the eval_in/out_possible_values attribute. If the entry
        exists (actual full name of a variable anonymized wrt. driver node), then the entry is added to _out_names.
        If a short name is used that cannot be associated to one and only one variable in the subprocess, an error
        string is added to _out_errors.

        Arguments:
            var_names (list[string]): list of variable names to query
            io_type (string): 'in'/'out' for input/output subprocess variables resp.
        Returns:
            _out_names (list[string]): output list of variable full names anonymized wrt. driver node.
            _out_errors (list[string]): list of error strings obtained from the queries for data integrity (empty if OK)
        """
        if io_type == self.IO_TYPE_IN:
            subpr_vars = self.eval_in_possible_types
        elif io_type == self.IO_TYPE_OUT:
            subpr_vars = set(self.eval_out_possible_values)
        else:
            raise ValueError(f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]')
        # TODO: related to short names logic
        _out_names = []
        _out_errors = []
        for var_name in var_names:
            if var_name in subpr_vars:
                _out_names.append(var_name)
            else:
                subpr_var_names = [var for var in subpr_vars if var.endswith(f".{var_name}")]
                if not subpr_var_names:
                    _out_names.append(None)
                    _out_errors.append(f'Variable {var_name} is not among subprocess {io_type}puts.')
                elif len(subpr_var_names) > 1:
                    _out_names.append(None)
                    _out_errors.append(f'Variable {var_name} appears more than once among optimisation subprocess '
                                       f'{io_type}puts, please use a non-ambiguous variable name.')
                else:
                    _out_names.append(subpr_var_names[0])
        return _out_names, _out_errors

    def _update_names(self, var_names, io_type):
        """
        Utility function for getting full name from a short name of a variable in the subprocess. Should not be called
        before data integrity checks as it will raise an error if no solution found, or if ambiguity determining
        the variable full name.
        Arguments:
            var_names (list[string]): list of variable names to query
            io_type (string): 'in'/'out' for input/output subprocess variables resp.
        Returns:
            f_names (list[string]): output list of variable full names (absolute, with study name too)
        """
        f_names, query_err = self._get_subprocess_var_names(var_names, io_type)
        if query_err:
            raise ValueError(" ".join(query_err))  # this already has been checked on data_integrity.
        f_names = self._compose_with_driver_ns(f_names)
        return f_names