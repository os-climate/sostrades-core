'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/06-2025/02/14 Copyright 2025 Capgemini

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

from copy import deepcopy
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.scenarios.base_scenario import BaseScenario
from numpy import inf, ndarray, size

from sostrades_core.execution_engine.data_manager import POSSIBLE_VALUES
from sostrades_core.execution_engine.discipline_wrapp import DisciplineWrapp
from sostrades_core.execution_engine.optim_manager_disc import OptimManagerDisc
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.tools.design_space import design_space as dspace_tool
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace


class ProxyOptim(ProxyDriverEvaluator):
    """
    **ProxyOptim** is a class proxy for an optim on the SoSTrades side.

    It contains the information and methonds necessary for i/o configuration (static or dynamic).



    Leaves of the process tree are direct instances of ProxyDiscipline. Other nodes are instances that inherit from
    ProxyDiscipline (e.g. ProxyCoupling).

    An instance of ProxyDiscipline is in one-to-one aggregation with an instance of DisciplineWrapp, which allows the

    use of different wrapping modes to provide the model run.

    During the prepare_execution step, the ProxyDiscipline coordinates the instantiation of the GEMSEO objects that
    manage the model run.

    Attributes:
        discipline_wrapp (DisciplineWrapp): aggregated object that references the wrapper and GEMSEO discipline

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
    default_algo_options = {
        "ftol_rel": 3e-16,
        "normalize_design_space": True,
        "maxls": 100,
        "maxcor": 50,
        "pg_tol": 1.0e-8,
        "max_iter": 500,
        "disp": False,
    }
    default_parallel_options = {
        'parallel': False,
        'n_processes': cpu_count(),
        'use_threading': False,
        'wait_time_between_fork': 0,
    }
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
    APPROX_MODES = (FINITE_DIFFERENCES, COMPLEX_STEP)
    AVAILABLE_MODES = (
        JacobianAssembly.DerivationMode.AUTO,
        JacobianAssembly.DerivationMode.DIRECT,
        JacobianAssembly.DerivationMode.ADJOINT,
        JacobianAssembly.DerivationMode.REVERSE,
        FINITE_DIFFERENCES,
        COMPLEX_STEP,
    )
    POST_PROC_MDO_DATA = 'post_processing_mdo_data'
    DESIGN_SPACE_OUT = 'design_space_out'
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

    SCENARIO_MANDATORY_FIELDS = (DESIGN_SPACE, FORMULATION, MAXIMIZE_OBJECTIVE, OBJECTIVE_NAME)

    OPTIMAL_OBJNAME_SUFFIX = "opt"
    ALGO_MANDATORY_FIELDS = (ALGO, MAX_ITER)

    DIFFERENTIATION_METHOD = 'differentiation_method'
    EVAL_JAC = 'eval_jac'
    EVAL_MODE = 'eval_mode'
    EXECUTE_AT_XOPT = 'execute_at_xopt'

    DESACTIVATE_OPTIM_OUT_STORAGE = 'desactivate_optim_out_storage'

    default_algo_options_plbfgsb = {
        'max_iter': 999,
        'ftol_rel': 1e-9,
        'ftol_abs': 1e-9,
        'xtol_rel': 1e-9,
        'xtol_abs': 1e-9,
        'max_ls_step_size': 0.0,
        'maxls': 20,
        'max_fun_eval': 999999,
        'max_time': 0,
        'pg_tol': 1e-5,
        'disp': False,
        'maxCGit': -1,
        'eta': -1.0,
        'factr': 1e7,
        'maxcor': 20,
        'normalize_design_space': True,
        'eq_tolerance': 1e-2,
        'ineq_tolerance': 1e-4,
        'stepmx': 0.0,
        'minfev': 0.0,
        'linesearch': 'lnsrlb',
        'lnsrlb_xtol': 0.1,
        'projection': 'proj_bound',
        'func_target': None,
        'ln_step_init': 1.0,
        'max_ln_step': 1e99,
        'lmem': 10,
        'precond': None,
        'precond_file': None,
        'use_cauchy_linesearch': None,
        'zero_tol': 1.0e-15,
        'primal_epsilon': 1e-10,
        'bound_tol': 1e-10,
        'gcp_precond_space': None,
        'lnsrlb_max_fg_calls': 21,
        'lnsrlb_stpmin': 0.0,
        'lnsrlb_ftol': 1e-3,
        'lnsrlb_gtol': 0.9,
        'lnsrlb_xtrapl': 1.1,
        'lnsrlb_xtrapu': 4.0,
        'unfeas_comp_exeption': None,
        'epsmch': 1e-16,
    }

    default_algo_options_nlopt = {
        'ftol_abs': 1e-14,
        'xtol_abs': 1e-14,
        'max_iter': 999,
        'ftol_rel': 1e-8,
        'xtol_rel': 1e-8,
        'max_time': 0.0,
        'ctol_abs': 1e-6,
        'stopval': None,
        'normalize_design_space': True,
        'eq_tolerance': 1e-2,
        'ineq_tolerance': 1e-4,
        'init_step': 0.25,
    }

    default_algo_options_openopt = {
        'max_iter': 999,  # pylint: disable=W0221
        'ftol_abs': 1e-12,
        'xtol_abs': 1e-12,
        'iprint': 1000,
        'max_time': float("inf"),
        'max_cpu_time': float("inf"),
        'max_ls_step_nb': 500,
        'max_fun_eval': 100000,
        'normalize_design_space': True,
        'eq_tolerance': 1e-2,
        'ineq_tolerance': 1e-4,
        'scale': None,
        'pg_tol': 0.0,
    }

    default_algo_options_oa = {
        'max_iter': 999,  # pylint: disable=W0221
        'ftol_abs': 1e-12,
        'algo_options_MILP': {},
        'algo_options_NLP': default_algo_options_nlopt,
        'algo_NLP': 'SLSQP',
        'normalize_design_space': False,
    }

    algo_dict = {
        "NLOPT": default_algo_options_nlopt,
        "OPENOPT": default_algo_options_openopt,
        "P-L-BFGS-B": default_algo_options_plbfgsb,
        "OuterApproximation": default_algo_options_oa,
        "PYMOO_GA": {"normalize_design_space": False},
        "PYMOO_NSGA2": {"normalize_design_space": False},
        "PYMOO_NSGA3": {"normalize_design_space": False, "ref_dirs_name": "energy"},
        "PYMOO_UNSGA3": {"normalize_design_space": False, "ref_dirs_name": "energy"},
    }

    DESC_IN = {
        ALGO: {'type': 'string', 'structuring': True, 'numerical': True, 'default': 'SLSQP'},
        DESIGN_SPACE: {
            'type': 'dataframe',
            'structuring': True,
            'numerical': True,
            'dataframe_descriptor': {
                VARIABLES: ('string', None, True),
                VALUES: ('multiple', None, True),
                LOWER_BOUND: ('multiple', None, True),
                UPPER_BOUND: ('multiple', None, True),
                ENABLE_VARIABLE_BOOL: ('bool', None, True),
                LIST_ACTIVATED_ELEM: ('list', None, True),
            },
        },
        FORMULATION: {'type': 'string', 'numerical': True, 'structuring': True},
        MAXIMIZE_OBJECTIVE: {'type': 'bool', 'structuring': True, 'numerical': True, 'default': False},
        OBJECTIVE_NAME: {'type': 'string', 'numerical': True, 'structuring': True},
        DIFFERENTIATION_METHOD: {
            'type': 'string',
            'default': BaseScenario.DifferentiationMethod.FINITE_DIFFERENCES,
            'numerical': True,
            'possible_values': [
                USER_GRAD,
                BaseScenario.DifferentiationMethod.FINITE_DIFFERENCES,
                BaseScenario.DifferentiationMethod.COMPLEX_STEP,
            ],
            'structuring': True,
        },
        FD_STEP: {'type': 'float', 'structuring': True, 'numerical': True, 'default': 1e-6},
        ALGO_OPTIONS: {
            'type': 'dict',
            'dataframe_descriptor': {VARIABLES: ('string', None, False), VALUES: ('string', None, True)},
            'dataframe_edition_locked': False,
            'default': default_algo_options,
            'structuring': True,
            'numerical': True,
        },
        PARALLEL_OPTIONS: {
            'type': 'dict',  # SoSDisciplineBuilder.OPTIONAL: True,
            'dataframe_descriptor': {
                VARIABLES: ('string', None, False),  # bool
                VALUES: ('string', None, True),
            },
            #                                   'dataframe_descriptor': {'parallel': ('int', None, True), #bool
            #                                                          'n_processes': ('int', None, True),
            #                                                          'use_threading': ('int', None, True),#bool
            #                                                          'wait_time_between_fork': ('int', None, True)},
            'dataframe_edition_locked': False,
            'default': default_parallel_options,
            'structuring': True,
            'numerical': True,
        },
        EVAL_MODE: {'type': 'bool', 'numerical': True, 'default': False, POSSIBLE_VALUES: [True, False],
                    'structuring': True},
        EVAL_JAC: {'type': 'bool', 'numerical': True, 'default': False, POSSIBLE_VALUES: [True, False]},
        EXECUTE_AT_XOPT: {'type': 'bool', 'numerical': True, 'default': True},
        MAX_ITER: {'type': 'float', 'numerical': True},
        INEQ_CONSTRAINTS: {
            'type': 'list',
            'subtype_descriptor': {'list': 'string'},
            'default': [],
            'structuring': True,
            'numerical': True,
        },
        EQ_CONSTRAINTS: {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [], 'numerical': True,
                         'structuring': True},
        DESACTIVATE_OPTIM_OUT_STORAGE: {'type': 'bool', 'default': True, 'numerical': True,
                                        POSSIBLE_VALUES: [True, False]},
    }

    DESC_OUT = {}

    def __init__(self, sos_name, ee, cls_builder, with_data_io=True, associated_namespaces=None):
        """Constructor"""
        super().__init__(sos_name, ee, cls_builder, associated_namespaces=associated_namespaces)
        if cls_builder is None:
            cls_builder = []
        self.cls_builder = cls_builder
        self.discipline_wrapp = None

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

        self.discipline_wrapp = DisciplineWrapp(name=sos_name, logger=self.logger.getChild("DisciplineWrapp"))

        self.check_integrity_msg_list = []
        self.opt_data_integrity = True

    def setup_sos_disciplines(self):
        """Overload setup_sos_disciplines to create a dynamic desc_in"""
        data_in = self.get_sosdisc_inputs()
        if self.ALGO_OPTIONS in data_in:
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            if algo_name is not None:
                default_dict = self.get_algo_options(algo_name)
                if algo_options is not None:
                    values_dict = deepcopy(default_dict)
                    for k in algo_options:
                        if algo_options[k] is not None and algo_options[k] != 'None':
                            values_dict.update({k: algo_options[k]})
                    self.inst_desc_in[self.ALGO_OPTIONS] = values_dict
                    for key in self._data_in:
                        if key[0] == self.ALGO_OPTIONS:
                            self._data_in[key][self.VALUE] = values_dict
        dynamic_outputs = {}
        if self.DESACTIVATE_OPTIM_OUT_STORAGE in data_in:
            desactivate_optim_storage = self.get_sosdisc_inputs(self.DESACTIVATE_OPTIM_OUT_STORAGE)
            if not desactivate_optim_storage:
                dynamic_outputs[self.POST_PROC_MDO_DATA] = {self.TYPE: 'dict',
                                                            self.NUMERICAL: True}
                dynamic_outputs[self.DESIGN_SPACE_OUT] = {self.TYPE: 'dataframe',
                                                          self.NUMERICAL: True}

        self.add_outputs(dynamic_outputs)

        self.set_edition_inputs_if_eval_mode()

    def prepare_build(self):
        """To be overload by subclasses with special builds."""
        return [self.cls_builder] if not isinstance(self.cls_builder, list) else self.cls_builder

    def set_edition_inputs_if_eval_mode(self):
        """If eval mode then algo and algo options will turn to not editable"""
        if 'eval_mode' in [key[0] for key in self._data_in]:
            eval_mode = self.get_sosdisc_inputs(self.EVAL_MODE)
            if eval_mode:
                data_in = self.get_data_in()
                self.eval_mode = True
                self.eval_jac = self.get_sosdisc_inputs(self.EVAL_JAC)
                self._data_in[self.ALGO, id(data_in[self.ALGO][self.NS_REFERENCE])][self.EDITABLE] = False
                self._data_in[self.ALGO_OPTIONS, id(data_in[self.ALGO_OPTIONS][self.NS_REFERENCE])][self.EDITABLE] = (
                    False
                )
                self._data_in[self.FORMULATION, id(data_in[self.FORMULATION][self.NS_REFERENCE])][self.EDITABLE] = False
                self._data_in[self.MAXIMIZE_OBJECTIVE, id(data_in[self.MAXIMIZE_OBJECTIVE][self.NS_REFERENCE])][
                    self.EDITABLE
                ] = False
                self._data_in[self.PARALLEL_OPTIONS, id(data_in[self.PARALLEL_OPTIONS][self.NS_REFERENCE])][
                    self.EDITABLE
                ] = False
                self._data_in[self.MAX_ITER, id(data_in[self.MAX_ITER][self.NS_REFERENCE])][self.EDITABLE] = False

                self._data_in[self.ALGO, id(data_in[self.ALGO][self.NS_REFERENCE])][self.OPTIONAL] = True
                self._data_in[self.ALGO_OPTIONS, id(data_in[self.ALGO_OPTIONS][self.NS_REFERENCE])][self.OPTIONAL] = (
                    True
                )
                self._data_in[self.FORMULATION, id(data_in[self.FORMULATION][self.NS_REFERENCE])][self.OPTIONAL] = True
                self._data_in[self.MAXIMIZE_OBJECTIVE, id(data_in[self.MAXIMIZE_OBJECTIVE][self.NS_REFERENCE])][
                    self.OPTIONAL
                ] = True
                self._data_in[self.PARALLEL_OPTIONS, id(data_in[self.PARALLEL_OPTIONS][self.NS_REFERENCE])][
                    self.OPTIONAL
                ] = True
                self._data_in[self.MAX_ITER, id(data_in[self.MAX_ITER][self.NS_REFERENCE])][self.OPTIONAL] = True
            else:
                self.eval_jac = False

    def prepare_execution(self):
        """Preparation of the GEMSEO process, including GEMSEO objects instanciation"""
        self.algo_name, self.algo_options, self.max_iter = self.get_sosdisc_inputs([
            self.ALGO,
            self.ALGO_OPTIONS,
            self.MAX_ITER,
        ])
        self.formulation, self.objective_name, self.design_space, self.maximize_objective = self.pre_set_scenario()

        # prepare_execution of proxy_disciplines and extract GEMSEO objects
        if self.formulation:
            sub_disciplines = []
            self.set_diff_mode_under_optim()
            for disc in self.proxy_disciplines:
                disc.prepare_execution()
                # Exclude non executable proxy Disciplines
                if disc.discipline_wrapp is not None:
                    sub_disciplines.append(disc.discipline_wrapp.discipline)

            # create_mdo_scenario from DisciplineWrapp
            self.discipline_wrapp.create_mdo_scenario(sub_disciplines, proxy=self, reduced_dm=self.ee.dm.reduced_dm)
            self.scenario = self.discipline_wrapp.discipline.scenario
            self.set_constraints()
            self.set_diff_method()
            self.set_design_space_for_complex_step()
            self.set_parallel_options()

            self.set_formulation_for_func_manager(sub_disciplines)

            # update MDA flag to flush residuals between each mda run
            self._set_flush_submdas_to_true()

    def set_formulation_for_func_manager(self, sub_disciplines):
        """

        If a func manager exists in the coupling then
        we associate the formulation of the optim in order to retrieve current iter in the func_manager

        """
        # formulation can be found in the GEMSEO discipline
        formulation = self.scenario.formulation

        # Check that only 1 discipline is below the proxy optim
        if len(sub_disciplines) == 1:
            coupling = sub_disciplines[0]
            # gather all disciplines under the coupling that are FunctionManagerDisc disicpline
            func_manager_list = [
                disc.sos_wrapp for disc in coupling.disciplines if isinstance(disc.sos_wrapp, OptimManagerDisc)
            ]
            # Normally only one OptimManagerDisc should be under the optim
            # if multiple do nothing
            if len(func_manager_list) == 1:
                func_manager = func_manager_list[0]
                func_manager.set_optim_formulation(formulation)

    def pre_set_scenario(self):
        """Prepare the set GEMS set_scenario method"""
        design_space = None
        formulation = None
        obj_full_name = None
        maximize_objective = False
        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        if dspace is not None:
            # build design space
            design_space = self.set_design_space()
            if design_space.variable_names:
                _, formulation, maximize_objective, obj_name = self.get_sosdisc_inputs(self.SCENARIO_MANDATORY_FIELDS)

                # get full objective ids
                obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
                obj_full_name = self._update_names([obj_name], self.IO_TYPE_OUT)[0]

        return formulation, obj_full_name, design_space, maximize_objective

    def set_design_space(self) -> DesignSpace:
        """Reads design space (set_design_space)"""
        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE).copy()
        dspace_df[self.VARIABLES] = self._update_names(dspace_df[self.VARIABLES], self.IO_TYPE_IN)
        design_space, self.dict_desactivated_elem = dspace_tool.create_gemseo_dspace_from_dspace_df(dspace_df)
        return design_space

    def get_chart_filter_list(self):
        chart_filters = []

        chart_list = ['Fitness function', 'Design variables']

        desactivate_post_processing_mdo_data = self.get_sosdisc_inputs(self.DESACTIVATE_OPTIM_OUT_STORAGE)
        if not desactivate_post_processing_mdo_data:
            post_processing_mdo_data = self.get_sosdisc_outputs(self.POST_PROC_MDO_DATA)

            if len(post_processing_mdo_data["constraints"]) > 0:
                chart_list.append("Constraints variables")

            chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'charts'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        instanciated_charts = []

        desactivate_post_processing_mdo_data = self.get_sosdisc_inputs(self.DESACTIVATE_OPTIM_OUT_STORAGE)
        if not desactivate_post_processing_mdo_data:
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

            post_processing_mdo_data = self.get_sosdisc_outputs(self.POST_PROC_MDO_DATA)

            def to_series(varname: str, x: list, y: ndarray) -> list[InstanciatedSeries]:
                dim = y.shape[1]
                series = []
                for d in range(dim):
                    series_name = varname if dim == 1 else f"{varname}[{d}]"
                    new_series = InstanciatedSeries(x, list(y[:, d]), series_name, 'lines', True)
                    series.append(new_series)
                return series

            if select_all or "Fitness function" in chart_list:
                fitness_func_through_iterations = post_processing_mdo_data["objective"]
                iterations = list(range(len(fitness_func_through_iterations)))

                chart_name = 'Objective function optimization'

                new_chart = TwoAxesInstanciatedChart('Iterations', 'Fitness function', chart_name=chart_name)

                for series in to_series(varname="Fitness function", x=iterations, y=fitness_func_through_iterations):
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
                            iterations, list(history_values_variable[:, i]), f"{shortened_var_name}[{i}]", 'lines', True
                        )
                        new_chart.add_series(new_series)
                    instanciated_charts.append(new_chart)

            if len(post_processing_mdo_data["constraints"]) > 0 and (
                select_all or 'Constraints variables' in chart_list
            ):
                dict_variables_history = post_processing_mdo_data["constraints"]
                min_y, max_y = inf, -inf
                all_series = []
                for variable_name, history in dict_variables_history.items():
                    iterations = list(range(len(history)))
                    min_value, max_value = history.min(), history.max()
                    if max_value > max_y:
                        max_y = max_value
                    if min_value < min_y:
                        min_y = min_value
                    all_series += list(to_series(varname=variable_name, x=iterations, y=history))

                chart_name = 'Constraints variables evolution'
                new_chart = TwoAxesInstanciatedChart(
                    'Iterations',
                    'Constraints variables',
                    [min(iterations), max(iterations)],
                    [min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1],
                    chart_name,
                )
                for series in all_series:
                    new_chart.series.append(series)
                instanciated_charts.append(new_chart)

        return instanciated_charts

    def set_design_space_for_complex_step(self):
        """Set design space values to complex if the differentiation method is complex_step"""
        diff_method = self.get_sosdisc_inputs(self.DIFFERENTIATION_METHOD)
        if diff_method == self.COMPLEX_STEP:
            dspace = deepcopy(self.scenario.formulation.optimization_problem.design_space)
            curr_x = dspace._current_x
            for var in curr_x:
                curr_x[var] = curr_x[var].astype('complex128')
            self.scenario.formulation.optimization_problem.design_space = dspace

    def get_algo_options(self, algo_name: str):
        """
        Create default dict for algo options.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            A dictionary with algo options default values.

        """
        # TODO : add warning and log algo options
        default_dict = {}
        driver_lib = OptimizationLibraryFactory().create(algo_name)
        algo_options = driver_lib.ALGORITHM_INFOS[algo_name].Settings.model_fields
        algo_options_keys = list(algo_options.keys())

        found_algo_names = [key for key in self.algo_dict if key in algo_name]

        if found_algo_names:
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
        _algo_factory = OptimizationLibraryFactory()
        avail_algos = _algo_factory.algorithms
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.ALGO}', self.POSSIBLE_VALUES, avail_algos)

        # fill the possible values of formulations
        _form_factory = MDOFormulationFactory()
        avail_formulations = _form_factory.class_names
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.FORMULATION}', self.POSSIBLE_VALUES, avail_formulations)

        # fill the possible values of maximize_objective
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.MAXIMIZE_OBJECTIVE}', self.POSSIBLE_VALUES, [False, True])

    def _update_eval_output_with_possible_out_values(self, possible_out_values, disc_in):
        pass

    def set_diff_mode_under_optim(self):
        """Set linearization_mode under optim with respect to differentiation_method or send a warning"""
        diff_method = self.get_sosdisc_inputs(self.DIFFERENTIATION_METHOD)

        if diff_method in self.APPROX_MODES:
            for disc in self.proxy_disciplines:
                if disc.linearization_mode != diff_method:
                    self.logger.warning(
                        "The differentiation method `%s` will overload the linearization mode `%s`",
                        diff_method,
                        disc.linearization_mode,
                    )
        elif diff_method == 'user':
            for disc in self.proxy_disciplines:
                if disc.linearization_mode in self.APPROX_MODES:
                    self.logger.warning(
                        "The differentiation method `%s` will overload the linearization mode `%s by default with auto linearization mode`",
                        diff_method,
                        disc.linearization_mode,
                    )
                    disc.linearization_mode = 'auto'

    def set_diff_method(self):
        """Set differentiation method"""
        diff_method = self.get_sosdisc_inputs(self.DIFFERENTIATION_METHOD)
        fd_step = self.get_sosdisc_inputs(self.FD_STEP)
        self.scenario.set_differentiation_method(diff_method, fd_step)

    def set_parallel_options(self):
        """Sets parallel options for jacobian approximation"""
        # update default options with user options
        user_options = self.get_sosdisc_inputs(self.PARALLEL_OPTIONS)
        if user_options is None:
            user_options = {}
        options = deepcopy(self.default_parallel_options)
        options.update(user_options)
        parallel = options.pop("parallel")
        # update problem options
        self.scenario.formulation.optimization_problem.parallel_differentiation = parallel
        self.scenario.formulation.optimization_problem.parallel_differentiation_options = (
            options
        )

    def set_constraints(self):
        # -- inequality constraints
        # NB: ineq constraint definition as [(var_name, cstr_sign), ...] deprecated
        ineq_names = self.get_sosdisc_inputs(self.INEQ_CONSTRAINTS)
        is_positive = [False for _ in ineq_names]
        ineq_full_names = self._update_names(ineq_names, self.IO_TYPE_OUT)
        for ineq, is_pos in zip(ineq_full_names, is_positive):
            self.scenario.add_constraint(
                ineq, MDOFunction.ConstraintType.INEQ, ineq, positive=is_pos
            )

        # -- equality constraints
        eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
        eq_full_names = self._update_names(eq_names, self.IO_TYPE_OUT)
        for eq in eq_full_names:
            self.scenario.add_constraint(
                eq, MDOFunction.ConstraintType.EQ, eq, positive=False
            )

    def _set_flush_submdas_to_true(self):
        # update MDA flag to flush residuals between each mda run
        for disc in self.proxy_disciplines:
            if disc.is_sos_coupling and len(disc.discipline_wrapp.discipline.inner_mdas) > 0:
                for sub_mda in disc.discipline_wrapp.discipline.inner_mdas:
                    sub_mda.reset_history_each_run = True

    def __str__(self):
        """
        Summarize results for display

        :returns: string summarizing results
        """
        msg = ""
        if hasattr(self, "disciplines"):
            msg = self.__class__.__name__ + ":\nDisciplines: "
            disc_names = [disc.name for disc in self._disciplines]  # pylint: disable=E1101
            msg += " ".join(disc_names)
            msg += "\nMDOFormulation: "  # We keep MDO here has is done in gemseo
            msg += self.formulation.__class__.__name__
            msg += "\nAlgorithm: "
            msg += str(self.get_sosdisc_inputs(self.ALGO)) + "\n"
        return msg

    def check_data_integrity(self):
        """Specific check data integrity of the ProxyOptim."""
        super().check_data_integrity()

        self.opt_data_integrity = True
        disc_in = self.get_data_in()

        # DESIGN SPACE -------------------------------------------------------------------------
        design_space_integrity_msg = []
        if self.DESIGN_SPACE in disc_in:
            design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
            if design_space is not None:
                # specific check for the short names
                var_names = design_space[self.VARIABLES].tolist()
                _, out_errors = self._get_subprocess_var_names(var_names, io_type=self.IO_TYPE_IN)
                design_space_integrity_msg.extend(out_errors)

                # type checks based on design space value  # TODO: type checks based on DM
                for var_name, var_value, var_lb, var_ub in zip(
                    var_names,
                    design_space[self.VALUE].tolist(),
                    design_space[self.LOWER_BOUND].tolist(),
                    design_space[self.UPPER_BOUND].tolist(),
                ):
                    if not (size(var_value) > 0 and size(var_lb) > 0 and size(var_ub) > 0):
                        design_space_integrity_msg.append(
                            f"Please fill columns {self.VALUE}, {self.LOWER_BOUND} and {self.UPPER_BOUND} "
                            f"for variable {var_name}."
                        )
                    else:
                        var_type = type(var_value).__name__
                        var_lb_type = type(var_lb).__name__
                        var_ub_type = type(var_ub).__name__
                        ok_types = {'array', 'list', 'ndarray'}
                        if var_type not in ok_types or var_lb_type not in ok_types or var_ub_type not in ok_types:
                            design_space_integrity_msg.append(
                                f"Columns {self.VALUE}, {self.LOWER_BOUND} and {self.UPPER_BOUND} must "
                                f"be arrays or lists for variable {var_name}."
                            )
                        elif len(var_value) == 0 or len(var_lb) == 0 or len(var_ub) == 0:
                            design_space_integrity_msg.append(
                                f"Please fill columns {self.VALUE}, {self.LOWER_BOUND} and {self.UPPER_BOUND} "
                                f"for variable {var_name}."
                            )

                # TODO: no check of possible values because of the short name, specific check below that needs to be
                #  changed if changing name rule
                design_space_integrity_msg.extend(
                    dspace_tool.check_design_space_data_integrity(design_space, possible_variables_types=None)
                )

        if design_space_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(
                self.get_var_full_name(self.DESIGN_SPACE, disc_in),
                self.CHECK_INTEGRITY_MSG,
                '\n'.join(design_space_integrity_msg),
            )
        # OBJECTIVE ---------------------------------------------------------------------------
        obj_integrity_msg = []
        if self.OBJECTIVE_NAME in disc_in:
            obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
            if obj_name is not None:
                # specific checks that the objective short name can be identified
                _, obj_out_err = self._get_subprocess_var_names([obj_name], io_type=self.IO_TYPE_OUT)
                obj_integrity_msg.extend(obj_out_err)
        if obj_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(
                self.get_var_full_name(self.OBJECTIVE_NAME, disc_in),
                self.CHECK_INTEGRITY_MSG,
                '\n'.join(obj_integrity_msg),
            )
        # INEQ CONSTRAINTS --------------------------------------------------------------------
        ineq_integrity_msg = []
        if self.INEQ_CONSTRAINTS in disc_in:
            ineq_names = self.get_sosdisc_inputs(self.INEQ_CONSTRAINTS)
            # specific checks that the short names can be identified
            _, ineq_out_err = self._get_subprocess_var_names(ineq_names, io_type=self.IO_TYPE_OUT)
            ineq_integrity_msg.extend(ineq_out_err)
        if ineq_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(
                self.get_var_full_name(self.INEQ_CONSTRAINTS, disc_in),
                self.CHECK_INTEGRITY_MSG,
                '\n'.join(ineq_integrity_msg),
            )
        # EQ CONSTRAINTS ----------------------------------------------------------------------
        eq_integrity_msg = []
        if self.INEQ_CONSTRAINTS in disc_in:
            eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
            # specific checks that the short names can be identified
            _, eq_out_err = self._get_subprocess_var_names(eq_names, io_type=self.IO_TYPE_OUT)
            eq_integrity_msg.extend(eq_out_err)
        if eq_integrity_msg:
            self.opt_data_integrity = False
            self.ee.dm.set_data(
                self.get_var_full_name(self.EQ_CONSTRAINTS, disc_in),
                self.CHECK_INTEGRITY_MSG,
                '\n'.join(eq_integrity_msg),
            )

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
            msg = f'data type {io_type} not recognized [{self.IO_TYPE_IN}/{self.IO_TYPE_OUT}]'
            raise ValueError(msg)
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
                    _out_errors.append(
                        f'Variable {var_name} appears more than once among optimisation subprocess '
                        f'{io_type}puts, please use a non-ambiguous variable name.'
                    )
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
        return self._compose_with_driver_ns(f_names)
