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
from gemseo.core.scenario import Scenario
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from sostrades_core.tools.eval_possible_values.eval_possible_values import find_possible_output_values

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import deepcopy
from multiprocessing import cpu_count

from numpy import array, ndarray, delete, inf

from gemseo.algos.design_space import DesignSpace
from gemseo.core.scenario import Scenario
from gemseo.core.function import MDOFunction
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.core.jacobian_assembly import JacobianAssembly
from sostrades_core.execution_engine.data_manager import POSSIBLE_VALUES
from sostrades_core.execution_engine.ns_manager import NamespaceManager
from sostrades_core.execution_engine.mdo_discipline_wrapp import MDODisciplineWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter


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
        'label': 'sostrades_core.execution_engine.sos_scenario',
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
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    VARIABLE_TYPE = "variable_type"
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

    DESC_IN = {'algo': {'type': 'string', 'structuring': True},
               'design_space': {'type': 'dataframe', 'structuring': True,
                                'dataframe_descriptor': {VARIABLES: ('string', None, True),
                                                         VALUES: ('multiple', None, True),
                                                         LOWER_BOUND: ('multiple', None, True),
                                                         UPPER_BOUND: ('multiple', None, True),
                                                         ENABLE_VARIABLE_BOOL: ('bool', None, True),
                                                         LIST_ACTIVATED_ELEM: ('list', None, True), }},

               'formulation': {'type': 'string', 'structuring': True},
               'maximize_objective': {'type': 'bool', 'structuring': True, 'default': False},
               'objective_name': {'type': 'string', 'structuring': True},
               'differentiation_method': {'type': 'string', 'default': Scenario.FINITE_DIFFERENCES,
                                          'possible_values': [USER_GRAD, Scenario.FINITE_DIFFERENCES,
                                                              Scenario.COMPLEX_STEP],
                                          'structuring': True},
               'fd_step': {'type': 'float', 'structuring': True, 'default': 1e-6},
               'algo_options': {'type': 'dict', 'dataframe_descriptor': {VARIABLES: ('string', None, False),
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
               'eval_mode': {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False], 'structuring': True},
               'eval_jac': {'type': 'bool', 'default': False, POSSIBLE_VALUES: [True, False]},
               'execute_at_xopt': {'type': 'bool', 'default': True},
               'max_iter': {'type': 'float'},
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
        self.opt_problem = None
        self.eval_mode = False
        self.eval_jac = False

        self.DESIGN_SPACE = 'design_space'
        self.FORMULATION = 'formulation'
        self.MAXIMIZE_OBJECTIVE = 'maximize_objective'
        self.OBJECTIVE_NAME = 'objective_name'
        self.FORMULATION_OPTIONS = 'formulation_options'

        #        self.SEARCH_PATHS = 'search_paths'
        self.SCENARIO_MANDATORY_FIELDS = [
            self.DESIGN_SPACE,
            self.FORMULATION,
            self.MAXIMIZE_OBJECTIVE,
            self.OBJECTIVE_NAME]
        #            self.SEARCH_PATHS]
        self.OPTIMAL_OBJNAME_SUFFIX = "opt"
        self.dict_desactivated_elem = {}
        self.activated_variables = []
        self.ALGO_MANDATORY_FIELDS = [self.ALGO, self.MAX_ITER]
        self.is_optim_scenario = True
        self.functions_before_run = []

        self.mdo_discipline_wrapp = MDODisciplineWrapp(name=sos_name, logger=self.logger.getChild("MDODisciplineWrapp"))

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        """
        # super().setup_sos_disciplines()
        if self.ALGO_OPTIONS in self.get_sosdisc_inputs().keys():
            algo_name = self.get_sosdisc_inputs(self.ALGO)
            algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
            if algo_name is not None:
                default_dict = self.get_algo_options(algo_name)
                if algo_options is not None:
                    values_dict = deepcopy(default_dict)

                    for k in algo_options.keys():
                        if algo_options[k] != 'None' or not isinstance(algo_options[k], type(None)):
                            values_dict.update({k: algo_options[k]})
                    self.inst_desc_in[self.ALGO_OPTIONS] = values_dict

                    # {(key, id(
                    #     value[self.NS_REFERENCE])): value for key, value in self.get_data_in().items()}
                    # id(
                    #     value[self.NS_REFERENCE])
                    for key in self._data_in.keys():
                        if self.ALGO_OPTIONS == key[0]:
                            self._data_in[key]['value'] = values_dict
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

    def configure(self):
        """
        Configuration of SoSScenario, call to super Class and
        """
        self.configure_io()
        self._update_status_dm(self.STATUS_CONFIGURE)

    def configure_io(self):
        """
        Configure discipline  and all sub-disciplines
        """
        if self._data_in == {} or self.check_structuring_variables_changes():
            super().configure()

        disc_to_configure = self.get_disciplines_to_configure()

        if len(disc_to_configure) > 0:
            self.set_configure_status(False)
        else:
            self.set_children_numerical_inputs()
            self.set_configure_status(True)

        for disc in disc_to_configure:
            disc.configure()

    def get_disciplines_to_configure(self):
        """
        Get sub disciplines list to configure
        """
        # TODO: not yet adapted ProxyOptim to case self.flatten_subprocess == True
        return self._get_disciplines_to_configure(self.proxy_disciplines)

    def set_edition_inputs_if_eval_mode(self):
        '''
        if eval mode then algo and algo options will turn to not editable
        '''

        if 'eval_mode' in [key[0] for key in self._data_in.keys()]:
            eval_mode = self.get_sosdisc_inputs('eval_mode')
            if eval_mode:
                data_in = self.get_data_in()
                self.eval_mode = True
                self.eval_jac = self.get_sosdisc_inputs('eval_jac')
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

        self.ee.dm.create_reduced_dm()
        # prepare_execution of proxy_disciplines
        sub_mdo_disciplines = []
        for disc in self.proxy_disciplines:
            disc.prepare_execution()
            # Exclude non executable proxy Disciplines
            if disc.mdo_discipline_wrapp is not None:
                sub_mdo_disciplines.append(
                    disc.mdo_discipline_wrapp.mdo_discipline)

        self.setup_sos_disciplines()

        self.algo_name, self.algo_options, self.max_iter = self.get_sosdisc_inputs(self.ALGO), self.get_sosdisc_inputs(
            self.ALGO_OPTIONS), self.get_sosdisc_inputs(self.MAX_ITER)
        self.formulation, self.objective_name, self.design_space, self.maximize_objective = self.pre_set_scenario()

        # create_mdo_scenario from MDODisciplineWrapp
        self.mdo_discipline_wrapp.create_mdo_scenario(
            sub_mdo_disciplines, proxy=self, reduced_dm=self.ee.dm.reduced_dm)
        self.set_constraints()
        self.set_diff_method()
        self.set_design_space_for_complex_step()
        self.set_parallel_options()

        self.set_formulation_for_func_manager(sub_mdo_disciplines)

        # Extract variables for eval analysis
        if self.proxy_disciplines is not None and len(self.proxy_disciplines) > 0:
            self.set_eval_possible_values()

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
            if any(type(design_variable).__name__ not in ['array', 'list', 'ndarray'] for design_variable in
                   dspace['value'].tolist()):
                raise ValueError(
                    f"A design variable must obligatory be an array {[type(design_variable).__name__ for design_variable in dspace['value'].tolist()]}")

            # build design space
            design_space = self.set_design_space()
            if design_space.variables_names:
                _, formulation, maximize_objective, obj_name = self.get_sosdisc_inputs(
                    self.SCENARIO_MANDATORY_FIELDS)

                # get full objective ids
                obj_name = self.get_sosdisc_inputs(self.OBJECTIVE_NAME)
                obj_full_name = self._update_names([obj_name])[0]

        return formulation, obj_full_name, design_space, maximize_objective

    def set_design_space(self) -> DesignSpace:
        """
        reads design space (set_design_space)
        """

        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        # update design space dv with full names
        dvs = list(dspace_df[self.VARIABLES])
        full_dvs = []
        dspace_dict_updated = {}

        for key in dvs:

            full_key_l = self.get_full_names([key])
            if len(full_key_l) > 0:
                full_key = full_key_l[0]
                full_dvs.append(full_key)
                # dspace_dict_updated[full_key] = dspace_df[key]
            else:
                raise Exception(f" The design variable {key} is not in the dm : {key}")

        dspace_dict_updated = dspace_df.copy()
        dspace_dict_updated[self.VARIABLES] = full_dvs

        design_space = self.read_from_dataframe(dspace_dict_updated)

        return design_space

    def get_full_names(self, names):
        '''
        get full names of variables
        '''
        full_names = []
        for i_name in names:
            full_id_l = self.dm.get_all_namespaces_from_var_name(i_name)
            if full_id_l != []:
                if len(full_id_l) > 1:
                    # full_id = full_id_l[0]
                    full_id = self.get_scenario_lagr(full_id_l)
                else:
                    full_id = full_id_l[0]
                full_names.append(full_id)

        return full_names

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

    def get_scenario_lagr(self, full_id_l):
        """
        get the corresponding lagrangian formulation of a given
        optimization scenario
        """

        possible_full_id_list = [ns for ns in full_id_l if f'{self.sos_name}.' in ns]

        if len(possible_full_id_list) == 1:
            return possible_full_id_list[0]
        else:
            raise Exception(f'Cannot find the only objective of the optim {self.sos_name} ')

    def set_design_space_for_complex_step(self):
        '''
        Set design space values to complex if the differentiation method is complex_step
        '''
        diff_method = self.get_sosdisc_inputs('differentiation_method')
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

    def _update_names(self, names):
        """
        if no dot in the name, it looks for the full name in the dm
        else we suppose that this is a full name that needs to be updated with current
        study name
        |!| it will NOT work for names with a dot in data_io...
        """
        local_names = []
        full_names = []
        for name in names:
            if NamespaceManager.NS_SEP not in name:
                local_names.append(name)
            else:
                full_names.append(name)
        return self.get_full_names(local_names) + \
            self._update_study_ns_in_varname(full_names)

    def set_eval_possible_values(self):

        possible_out_values = find_possible_output_values(self, strip_first_ns=True)

        # Fill the possible_values of obj and constraints
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.OBJECTIVE_NAME}',
                         self.POSSIBLE_VALUES, possible_out_values)

        if self.is_constraints:
            self.dm.set_data(f'{self.get_disc_full_name()}.{self.INEQ_CONSTRAINTS}',
                             self.POSSIBLE_VALUES, possible_out_values)
            self.dm.set_data(f'{self.get_disc_full_name()}.{self.EQ_CONSTRAINTS}',
                             self.POSSIBLE_VALUES, possible_out_values)
        # fill the possible values of algos
        self.mdo_discipline_wrapp.mdo_discipline._init_algo_factory()
        avail_algos = self.mdo_discipline_wrapp.mdo_discipline._algo_factory.algorithms
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.ALGO}',
                         self.POSSIBLE_VALUES, avail_algos)
        # fill the possible values of formulations
        self._form_factory = MDOFormulationsFactory()
        avail_formulations = self._form_factory.formulations
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.FORMULATION}',
                         self.POSSIBLE_VALUES, avail_formulations)
        # fill the possible values of maximize_objective
        self.dm.set_data(f'{self.get_disc_full_name()}.{self.MAXIMIZE_OBJECTIVE}',
                         self.POSSIBLE_VALUES, [False, True])

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
        # -- equality constraints
        # retrieve ineq_constraints data
        # the data is either a string "variable name" or
        # a tuple with the variable name and the ineq sign
        ineq_data = self.get_sosdisc_inputs(self.INEQ_CONSTRAINTS)
        ineq_names = []
        is_positive = []
        for data in ineq_data:
            if type(data) == str:
                # if no tuple, the default value of ineq sign is
                # negative
                name = data
                is_pos = False
            else:
                name = data[0]
                sign = data[1]
                if sign == self.INEQ_POSITIVE:
                    is_pos = True
                elif sign == self.INEQ_NEGATIVE:
                    is_pos = False
                else:
                    msg = "Sign of constraint %s is not among %s" % (
                        name, self.INEQ_SIGNS)
                    raise ValueError(msg)
            ineq_names.append(name)
            is_positive.append(is_pos)

        ineq_full_names = self._update_names(ineq_names)

        for ineq, is_pos in zip(ineq_full_names, is_positive):
            self.mdo_discipline_wrapp.mdo_discipline.add_constraint(
                ineq, MDOFunction.TYPE_INEQ, ineq, value=None, positive=is_pos)

        # -- equality constraints
        eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
        eq_full_names = self._update_names(eq_names)
        for eq in eq_full_names:
            self.mdo_discipline_wrapp.mdo_discipline.add_constraint(
                self, eq, MDOFunction.TYPE_EQ, eq, value=None,
                positive=False)

    def read_from_dict(self, dp_dict):
        """Parses a dictionary to read the DesignSpace

        :param dp_dict : design space dictionary
        :returns:  the design space
        """
        design_space = DesignSpace()
        for key in dp_dict:
            print(key)
            if type(dp_dict[key]['value']) != list and type(dp_dict[key]['value']) != ndarray:
                name = key
                var_type = ['float']

                size = 1
                l_b = array([dp_dict[key]['lower_bnd']])
                u_b = array([dp_dict[key]['upper_bnd']])
                value = array([dp_dict[key]['value']])
            else:
                size = len(dp_dict[key]['value'])
                var_type = ['float'] * size

                name = key
                l_b = array(dp_dict[key]['lower_bnd'])
                u_b = array(dp_dict[key]['upper_bnd'])
                value = array(dp_dict[key]['value'])

            design_space.add_variable(name, size, var_type, l_b, u_b, value)
        return design_space

    def read_from_dataframe(self, df):
        """Parses a DataFrame to read the DesignSpace

        :param df : design space df
        :returns:  the design space
        """
        names = list(df[self.VARIABLES])
        values = list(df[self.VALUES])
        l_bounds = list(df[self.LOWER_BOUND])
        u_bounds = list(df[self.UPPER_BOUND])
        enabled_variable = list(df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(df[self.LIST_ACTIVATED_ELEM])

        # looking for the optionnal variable type in the design space
        if self.VARIABLE_TYPE in df:
            var_types = df[self.VARIABLE_TYPE]
        else:
            # set to None for all variables if not exists
            var_types = [None] * len(names)

        design_space = DesignSpace()

        for dv, val, lb, ub, l_activated, enable_var, vtype in zip(names, values, l_bounds, u_bounds,
                                                                   list_activated_elem, enabled_variable, var_types):

            # check if variable is enabled to add it or not in the design var
            if enable_var:
                self.dict_desactivated_elem[dv] = {}

                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.dict_desactivated_elem[dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)

                # 'automatic' var_type values are overwritten if filled by the user
                if vtype is not None:
                    var_type = vtype

                design_space.add_variable(
                    dv, size, var_type, l_b, u_b, value)
        return design_space

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
