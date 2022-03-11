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

from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.function import MDOFunction

from sos_trades_core.execution_engine.sos_scenario import SoSScenario
from sos_trades_core.api import get_sos_logger


class SoSOptimScenario(SoSScenario, MDOScenario):
    """
    Generic implementation of Optimization Scenario
    """
    # Default values of algorithms

    # ontology information
    _ontology_data = {
        'label': 'Scenario Optimization Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-bezier-curve fa-fw',
        'version': '',
    }
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
                                    'projection': 'proj_bound', 'func_target': None, 'ln_step_init': 1.0, 'max_ln_step': 1e99,
                                    'lmem': 10, 'precond': None, 'precond_file': None, 'use_cauchy_linesearch': None, 'zero_tol': 1.0e-15,
                                    'primal_epsilon': 1e-10, 'bound_tol': 1e-10, 'gcp_precond_space': None, 'lnsrlb_max_fg_calls': 21, 'lnsrlb_stpmin': 0.0,
                                    'lnsrlb_ftol': 1e-3, 'lnsrlb_gtol': 0.9, 'lnsrlb_xtrapl': 1.1, 'lnsrlb_xtrapu': 4.0, 'unfeas_comp_exeption': None, 'epsmch': 1e-16}

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

    algo_dict = {"NLOPT": default_algo_options_nlopt,
                 "OPENOPT": default_algo_options_openopt,
                 "P-L-BFGS-B": default_algo_options_plbfgsb,
                 }
    is_constraints = True
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    INEQ_POSITIVE = "positive_ineq"
    INEQ_NEGATIVE = "negative_ineq"
    INEQ_SIGNS = [INEQ_POSITIVE, INEQ_NEGATIVE]

    # DESC_I/O
    DESC_IN = {'max_iter': {'type': 'float'},
               INEQ_CONSTRAINTS: {'type': 'string_list', 'default': [], 'structuring': True},
               EQ_CONSTRAINTS: {'type': 'string_list', 'default': [], 'structuring': True},
               }
    DESC_IN.update(SoSScenario.DESC_IN)

    DESC_OUT = {}
    DESC_OUT.update(SoSScenario.DESC_OUT)

    def __init__(self, sos_name, ee, cls_builder):
        """
        Constructor
        """
        SoSScenario.__init__(self, sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.SoSOptimScenario')

        self.ALGO_MANDATORY_FIELDS = [self.ALGO, self.MAX_ITER]
        self.is_optim_scenario = True
        self.functions_before_run = []

    def set_edition_inputs_if_eval_mode(self):
        '''
        If eval_mode max_iter is not editable and optional
        '''
        SoSScenario.set_edition_inputs_if_eval_mode(self)
        if 'eval_mode' in self._data_in:
            eval_mode = self.get_sosdisc_inputs('eval_mode')
            if eval_mode:
                self._data_in[self.MAX_ITER][self.EDITABLE] = False
                self._data_in[self.MAX_ITER][self.OPTIONAL] = True

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
            MDOScenario.add_constraint(
                self, ineq, MDOFunction.TYPE_INEQ, ineq, value=None, positive=is_pos)

        # -- equality constraints
        eq_names = self.get_sosdisc_inputs(self.EQ_CONSTRAINTS)
        eq_full_names = self._update_names(eq_names)
        for eq in eq_full_names:
            MDOScenario.add_constraint(
                self, eq, MDOFunction.TYPE_EQ, eq, value=None,
                positive=False)

    def set_scenario(self):
        '''
        Init the MDO Scenario wirh design space formulation and objectives
        '''
        # pre-set scenario
        design_space, formulation, obj_full_name = self.pre_set_scenario()

        if None not in [design_space, formulation, obj_full_name]:
            # MDOScenario creation (GEMS object)
            MDOScenario.__init__(self, self.sos_disciplines, formulation,
                                 obj_full_name, design_space, name=self.sos_name,
                                 grammar_type=SoSScenario.SOS_GRAMMAR_TYPE)

            self.activated_variables = self.formulation.design_space.variables_names
            self.set_diff_method()

            # add constraints
            self.set_constraints()

    def run_scenario(self):
        '''
        Call to the GEMSEO MDOScenario run and update design_space_out
        Post run is possible if execute_at_xopt is activated
        '''
        MDOScenario._run(self)
        self.update_design_space_out()

        self.execute_at_xopt()

    def execute_at_xopt(self):
        '''
        trigger post run if execute at optimum is activated
        '''
        #
        execute_at_opt = self.get_sosdisc_inputs('execute_at_xopt')

        if execute_at_opt:
            self.logger.info("Post run at xopt")
            self._post_run()

    def _run_algorithm(self):
        '''
        Run the chosen algorithm with algo options and max_iter
        '''
        problem = self.formulation.opt_problem
        # Clears the database when multiple runs are performed (bi level)
        if self.clear_history_before_run:
            problem.database.clear()
        algo_name = self.get_sosdisc_inputs(self.ALGO)
        max_iter = self.get_sosdisc_inputs(self.MAX_ITER)
        options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
        if options is None:
            options = {}
        if self.MAX_ITER in options:
            self.logger.warning("Double definition of algorithm option " +
                                "max_iter, keeping value: " + str(max_iter))
            options.pop(self.MAX_ITER)
        lib = self._algo_factory.create(algo_name)
        self.logger.info(options)
        self.preprocess_functions()

        self.optimization_result = lib.execute(problem, algo_name=algo_name,
                                               max_iter=max_iter,
                                               **options)
        return self.optimization_result

    def preprocess_functions(self):
        """
        preprocess functions to store functions list 
        """

        problem = self.formulation.opt_problem
        algo_options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
        normalize = algo_options['normalize_design_space']

        # preprocess functions
        problem.preprocess_functions(normalize=normalize)
        functions = problem.nonproc_constraints + \
            [problem.nonproc_objective]

        self.functions_before_run = functions

    def _post_run(self):
        """
        Post-processes the scenario.
        """
        formulation = self.formulation
        problem = formulation.opt_problem
        design_space = problem.design_space
        normalize = self.get_sosdisc_inputs(self.ALGO_OPTIONS)[
            'normalize_design_space']
        # Test if the last evaluation is the optimum
        x_opt = design_space.get_current_x()
        try:
            # get xopt from x_opt
            x_opt_result = problem.solution.x_opt
            self.logger.info(f"Executing at xopt point {x_opt}")
            self.logger.info(f"x_opt from problem solution is {x_opt_result}")
        except:
            self.logger.info(f"Exception {problem.solution}")
            pass
        # Revaluate all functions at optimum
        # To re execute all disciplines and get the right data

        # self.logger.info(
        #    f"problem database {problem.database._Database__dict}")
        try:

            self.evaluate_functions(problem, x_opt)

        except:
            self.logger.warning(
                "Warning: executing the functions in the except after nominal execution of post run failed")

            for func in self.functions_before_run:
                func(x_opt)

    def evaluate_functions(self,
                           problem,
                           x_vect=None,  # type: ndarray
                           ):  # type: (...) -> Tuple[Dict[str,Union[float,ndarray]],Dict[str,ndarray]]
        """Compute the objective and the constraints.

        amples.

        Args:
            x_vect: The input vector at which the functions must be evaluated;
                if None, x_0 is used.
            problem: opt problem object 


        """
        functions = problem.nonproc_constraints + \
            [problem.nonproc_objective]
        self.logger.info(f'list of functions to evaluate {functions}')

        for func in functions:
            try:
                func(x_vect)
            except ValueError:
                self.logger.error("Failed to evaluate function %s", func.name)
                raise
            except TypeError:
                self.logger.error("Failed to evaluate function %s", func)
                raise

    def get_scenario_lagr(self, full_id_l):
        """
        get the corresponding lagrangian formulation of a given
        optimization scenario
        """
        from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
        list_coupl = self.ee.root_process.sos_disciplines
        for i in list_coupl:
            if isinstance(i, SoSCoupling):
                if id(self) == id(i.sos_disciplines[0]):
                    scenario_name = i.sos_name
        for j in full_id_l:
            if scenario_name + '.' in j:
                full_id = j
        return full_id
