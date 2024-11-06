'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/10/17 Copyright 2023 Capgemini

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
import logging

from gemseo.scenarios.mdo_scenario import MDOScenario


class SoSMDOScenario(MDOScenario):
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

    POST_PROC_MDO_DATA = 'post_processing_mdo_data'

    def __init__(self,
                 disciplines,
                 formulation,
                 objective_name,
                 design_space,
                 name,
                 maximize_objective,
                 logger: logging.Logger,
                 reduced_dm=None):

        """
        Constructor
        """

        super().__init__(disciplines,
                         objective_name,
                         design_space,
                         name=name,
                         maximize_objective=maximize_objective,
                         formulation_name=formulation
                         )

        self.logger = logger
        self.algo_name = None
        self.algo_options = {}
        self.max_iter = None
        self.eval_mode = False
        self.eval_jac = False
        self.dict_desactivated_elem = {}
        self.input_design_space = None
        self.reduced_dm = reduced_dm
        self.activated_variables = self.formulation.design_space.variable_names
        self.is_sos_coupling = False
        self.mdo_options = {}

    def _run(self):
        '''

        '''

        if self.eval_mode:
            self.run_eval_mode()
        else:
            self.run_scenario()

    # def execute_at_xopt(self):
    #     '''
    #     trigger post run if execute at optimum is activated
    #     '''
    #     self.logger.info("Post run at xopt")
    #     self._post_run()

    # def _run_algorithm(self):
    #     '''
    #     Run the chosen algorithm with algo options and max_iter
    #     '''
    #     problem = self.formulation.optimization_problem
    #     # Clears the database when multiple runs are performed (bi level)
    #     if self.clear_history_before_run:
    #         problem.database.clear()
    #     algo_name = self.algo_name
    #     max_iter = self.max_iter
    #     options = self.algo_options
    #     if options is None:
    #         options = {}
    #     if "max_iter" in options:
    #         self.logger.warning("Double definition of algorithm option " +
    #                             "max_iter, keeping value: " + str(max_iter))
    #         options.pop("max_iter")
    #     lib = self._algo_factory.create(algo_name)
    #     self.logger.info(options)
    #
    #     self.preprocess_functions()
    #
    #     self.optimization_result = lib.execute(problem,
    #                                            max_iter=max_iter,
    #                                            **options)
    #     self.clear_jacobian()
    #     return self.optimization_result

    # def clear_jacobian(self):
    #     return SoSDiscipline.clear_jacobian(self)  # should rather be double inheritance

    def run_scenario(self):
        '''
        Call to the GEMSEO MDOScenario run and update design_space_out
        Post run is possible if execute_at_xopt is activated
        '''
        # I think it is already in GEMSEO
        # self.execute_at_xopt()

    def run_eval_mode(self):
        '''
        Run evaluate functions with the initial x
        jacobian_functions: The functions computing the Jacobians.
                If ``None``, evaluate all the functions computing Jacobians.
                If empty, do not evaluate functions computing Jacobians.
        '''
        output_functions, _ = self.formulation.optimization_problem.get_functions()
        jacobian_functions = []
        if self.eval_jac:
            jacobian_functions = output_functions

        outputs, jacobians = self.formulation.optimization_problem.evaluate_functions(output_functions=output_functions,
                                                                 jacobian_functions=jacobian_functions)

    # def preprocess_functions(self):
    #     """
    #     preprocess functions to store functions list
    #     """
    #
    #     problem = self.formulation.optimization_problem
    #     normalize = self.algo_options['normalize_design_space']
    #
    #     # preprocess functions
    #     problem.preprocess_functions(is_function_input_normalized=normalize)
    #     functions = list(problem.constraints.get_originals()) + [problem.objective.original]
    #
    #     self.functions_before_run = functions

    # def set_design_space_for_complex_step(self):
    #     '''
    #     Set design space values to complex if the differentiation method is complex_step
    #     '''
    #
    #     if self.formulation.optimization_problem.differentiation_method == self.COMPLEX_STEP:
    #         dspace = deepcopy(self.opt_problem.design_space)
    #         curr_x = dspace._current_x
    #         for var in curr_x:
    #             curr_x[var] = curr_x[var].astype('complex128')
    #         self.formulation.optimization_problem.design_space = dspace

    # def _post_run(self):
    #     """
    #     Post-processes the scenario.
    #     """
    #     formulation = self.formulation
    #     problem = formulation.optimization_problem
    #     design_space = problem.design_space
    #     normalize = self.algo_options[
    #         'normalize_design_space']
    #     # Test if the last evaluation is the optimum
    #     x_opt = design_space.get_current_value()
    #     try:
    #         # get xopt from x_opt
    #         x_opt_result = problem.solution.x_opt
    #         self.logger.info(f"Executing at xopt point {x_opt}")
    #         self.logger.info(f"x_opt from problem solution is {x_opt_result}")
    #     except:
    #         self.logger.info(f"Exception {problem.solution}")
    #         pass
    #     # Revaluate all functions at optimum
    #     # To re execute all disciplines and get the right data
    #
    #     # self.logger.info(
    #     #    f"problem database {problem.database._Database__dict}")
    #     try:
    #
    #         self.evaluate_functions(problem, x_opt)
    #
    #     except:
    #         self.logger.warning(
    #             "Warning: executing the functions in the except after nominal execution of post run failed")
    #
    #         for func in self.functions_before_run:
    #             func.evaluate(x_opt)
    #
