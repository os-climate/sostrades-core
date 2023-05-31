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
import numpy as np

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from typing import Union
from copy import deepcopy
import logging
import pandas as pd
from numpy import ndarray

from gemseo.core.mdo_scenario import MDOScenario


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

    def __init__(self,
                 disciplines, name,
                 formulation, objective_name,
                 design_space,
                 grammar_type=None,
                 reduced_dm=None,
                 logger:logging.Logger=None):
        """
        Constructor
        """
        self.logger = logger
        self.formulation = formulation
        self.objective_name = objective_name
        self.name = name
        super().__init__(disciplines,
            self.formulation,
            self.objective_name,
            design_space,
            name=self.name,
            grammar_type=grammar_type)
        self.maximize_objective = None
        self.algo_name = None
        self.algo_options = None
        self.max_iter = None
        self.eval_mode = False
        self.eval_jac = False
        self.dict_desactivated_elem = None
        self.input_design_space = None
        self.reduced_dm = reduced_dm
        self.activated_variables = self.formulation.design_space.variables_names
        self.is_sos_coupling=False

    def _run(self):
        '''

        '''
        self.status = self.STATUS_RUNNING
        self.update_default_coupling_inputs()

        if self.eval_mode:
            self.run_eval_mode()
        else:
            self.run_scenario()
        outputs = [discipline.get_output_data()
                   for discipline in self.disciplines]
        for data in outputs:
            self.local_data.update(data)
        self.update_design_space_out()
        if not self.eval_mode:
            self.update_post_processing_df()

    def update_post_processing_df(self):
        """Gathers the data for plotting the MDO graphs"""
        dataset = self.export_to_dataset()
        dataframe = dataset.export_to_dataframe()

        #dataframe = dataframe.rename(columns=rename_func)

        constraints_names = [constraint.name for constraint in
                             self.formulation.opt_problem.constraints]
        objective_name = self.formulation.opt_problem.objective.name

        def correct_var_name(varname: str) -> str:
            """removes study name from variable name"""
            corrected_var_name = ".".join(varname.split(".")[1:])
            return corrected_var_name

        out = {
            "objective": np.array(dataframe["functions"][objective_name].values),
            "variables": {correct_var_name(var): np.array(dataframe["design_parameters"][var].values) for var in self.design_space.variables_names},
            "constraints": {correct_var_name(var): np.array(dataframe["functions"][var].values) for var in constraints_names}
        }

        self.local_data.update({
            [key for key in self.get_output_data_names() if 'post_processing_mdo_data' in key][
                0]: out})


    def execute_at_xopt(self):
        '''
        trigger post run if execute at optimum is activated
        '''
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
        algo_name = self.algo_name
        max_iter = self.max_iter
        options = self.algo_options
        if options is None:
            options = {}
        if "max_iter" in options:
            self.logger.warning("Double definition of algorithm option " +
                                "max_iter, keeping value: " + str(max_iter))
            options.pop("max_iter")
        lib = self._algo_factory.create(algo_name)
        self.logger.info(options)

        self.preprocess_functions()

        self.optimization_result = lib.execute(problem, algo_name=algo_name,
                                               max_iter=max_iter,
                                               **options)
        return self.optimization_result

    def run_scenario(self):
        '''
        Call to the GEMSEO MDOScenario run and update design_space_out
        Post run is possible if execute_at_xopt is activated
        '''
        MDOScenario._run(self)

        self.execute_at_xopt()

    def run_eval_mode(self):
        '''
        Run evaluate functions with the initial x
        '''

        self.formulation.opt_problem.evaluate_functions(
            eval_jac=self.eval_jac, normalize=False)

        #self.store_local_data(**local_data)
        # if eval mode design space was not modified
        # self.store_sos_outputs_values(
        #     {'design_space_out': self.formulation.design_space}, update_dm=True)

    def preprocess_functions(self):
        """
        preprocess functions to store functions list 
        """

        problem = self.formulation.opt_problem
        normalize = self.algo_options['normalize_design_space']

        # preprocess functions
        problem.preprocess_functions(normalize=normalize)
        functions = problem.nonproc_constraints + \
            [problem.nonproc_objective]

        self.functions_before_run = functions

    def set_design_space_for_complex_step(self):
        '''
        Set design space values to complex if the differentiation method is complex_step
        '''

        if self.formulation.opt_problem.differentiation_method == self.COMPLEX_STEP:
            dspace = deepcopy(self.opt_problem.design_space)
            curr_x = dspace._current_x
            for var in curr_x:
                curr_x[var] = curr_x[var].astype('complex128')
            self.formulation.opt_problem.design_space = dspace

    def _post_run(self):
        """
        Post-processes the scenario.
        """
        formulation = self.formulation
        problem = formulation.opt_problem
        design_space = problem.design_space
        normalize = self.algo_options[
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
                           ):  # type: (...) -> tuple[dict[str,Union[float,ndarray]],dict[str,ndarray]]
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
        current_idx=0
        for k,v in problem.design_space.items():
            k_size=v.size
            self.local_data.update({k: x_vect[current_idx:current_idx+k_size]})
            current_idx+=k_size

    def update_default_coupling_inputs(self):
        '''
        Update default inputs of the couplings
        '''
        for disc in self.disciplines:
            self._set_default_inputs_from_local_data(disc)

    def _set_default_inputs_from_local_data(self, disc):
        """
        Based on dm values, default_inputs are set to mdachains,
        and default_inputs dtype is set to complex in case of complex_step gradient computation.
        """
        input_data = {}
        input_data_names = disc.get_input_data_names()
        for data_name in input_data_names:
            if data_name in self.local_data.keys():
                val = self.local_data[data_name]
            else:
                val = None
            # for cases of early configure steps
            if val is not None:
                input_data[data_name] = val

        # store mdo_chain default inputs
        if disc.is_sos_coupling:
            disc.mdo_chain.default_inputs.update(input_data)
        disc.default_inputs.update(input_data)

        if hasattr(disc, 'disciplines'):
            for disc in disc.disciplines:
                self._set_default_inputs_from_local_data(disc)

    def update_design_space_out(self):
        """
        Method to update design space with opt value
        """
        design_space = deepcopy(self.input_design_space)
        l_variables = design_space['variable']
        for var in l_variables:
            var = var.split('.')[-1]
            full_name_var = [full_name for full_name in self.get_input_data_names() if
                             (var == full_name.split('.')[-1] or var == full_name)][0]
            if full_name_var in self.activated_variables:
                value_x_opt = list(self.formulation.design_space._current_x.get(
                    full_name_var))
                if self.dict_desactivated_elem[full_name_var] != {}:
                    # insert a desactivated element
                    value_x_opt.insert(
                        self.dict_desactivated_elem[full_name_var]['position'],
                        self.dict_desactivated_elem[full_name_var]['value'])

                design_space.loc[design_space['variable'] == var, 'value'] = pd.Series(
                    [value_x_opt] * len(design_space))
        self.local_data.update({
            [key for key in self.get_output_data_names() if 'design_space_out' in key][
                0]: design_space})
