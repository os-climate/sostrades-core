'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/06/07 Copyright 2023 Capgemini

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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter

from sostrades_core.execution_engine.sos_mdo_scenario import SoSMDOScenario

if TYPE_CHECKING:
    import logging


class SoSMDOScenarioAdapter(MDOScenarioAdapter):
    """Generic implementation of Optimization Scenario"""

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

    scenario: SoSMDOScenario
    """The wrapped SoS scenario."""

    def __init__(
        self,
        disciplines,
        name,
        formulation,
        objective_name,
        design_space,
        maximize_objective,
        input_names,
        output_names,
        logger: logging.Logger,
        reduced_dm=None,
        mdo_options=None,
    ):
        """Constructor"""
        if mdo_options is None:
            mdo_options = {}
        self.logger = logger
        self.formulation = formulation
        self.objective_name = objective_name
        self.name = name
        self.scenario = SoSMDOScenario(
            disciplines, self.formulation, self.objective_name, design_space, self.name, maximize_objective, logger
        )
        # NOTE: adding the design variables as output names of the scenario adapter, so that the upper mda can retrieve
        # the inputs that are design variables in order to push them into the dm after the optimisation
        _output_names = list(set(output_names + self.scenario.formulation.optimization_problem.design_space.variable_names))
        super().__init__(self.scenario, input_names=[], output_names=_output_names, name=f'{self.name}_adapter')
        self.scenario.eval_mode = mdo_options.pop('eval_mode')
        self.scenario.eval_jac = mdo_options.pop('eval_jac')
        self.scenario.dict_desactivated_elem = mdo_options.pop('dict_desactivated_elem')
        self.scenario.input_design_space = mdo_options.pop('input_design_space')
        self.desactivate_optim_out_storage = mdo_options.pop('desactivate_optim_out_storage')
        self.design_space_out = None
        self.post_processing_mdo_data = None

        self.mdo_options = mdo_options

        self.reduced_dm = reduced_dm
        self.activated_variables = self.scenario.formulation.design_space.variable_names
        self.is_sos_coupling = False

    def _update_input_grammar(self) -> None:
        # desactivate designspace outputs for post processings
        self.desactivate_optim_out_storage = False

    def _execute(self) -> None:
        self._pre_run()
        # with LoggingContext(LOGGING_SETTINGS.logger, level=self.__scenario_log_level):
        self.scenario.execute(**self.mdo_options)
        if self.scenario.eval_jac:
            mda_chain = self.scenario.disciplines[0]
            mda_chain.linearize(mda_chain.io.data, execute=False)
            self.jac = mda_chain.jac
        if self.scenario.eval_mode:
            self._retrieve_top_level_outputs()
        else:
            self.optimization_result = self.scenario.optimization_result
            self._post_run()

        # I think everything is in the post run
        #        self.scenario_outputs_dict =
        # self.scenario_outputs = [discipline.get_output_data()
        #                          for discipline in self.disciplines]
        # for data in outputs:
        #     self.io.data.update(data)
        #
        self.add_design_space_inputs_to_local_data()
        # # save or not the output of design space for post processings
        if not self.desactivate_optim_out_storage:
            self.update_design_space_out()
            post_processing_mdo_data = {}
            if not self.scenario.eval_mode:
                self.post_processing_mdo_data = self.update_post_processing_df()
            # self.io.data.update({
            #     [key for key in self.get_output_data_names() if self.POST_PROC_MDO_DATA in key][
            #         0]: post_processing_mdo_data})

    def update_design_space_out(self):
        """Method to update design space with opt value."""
        design_space = deepcopy(self.scenario.input_design_space)
        l_variables = design_space['variable']

        for var_name in l_variables:
            var_name_loc = var_name.split('.')[-1]
            full_name_var = self.get_namespace_from_var_name(var_name_loc)
            if full_name_var in self.activated_variables:
                value_x_opt = [self.scenario.formulation.design_space.get_current_value([full_name_var])]
                if self.scenario.dict_desactivated_elem[full_name_var] != {}:
                    # insert a desactivated element
                    for _pos, _val in zip(
                        self.scenario.dict_desactivated_elem[full_name_var]['position'],
                        self.scenario.dict_desactivated_elem[full_name_var]['value'],
                    ):
                        value_x_opt.insert(_pos, _val)

                design_space.loc[design_space['variable'] == var_name_loc, 'value'] = pd.Series(
                    [value_x_opt] * len(design_space)
                )
        self.design_space_out = design_space
        # self.local_data.update({
        #     [key for key in self.get_output_data_names() if 'design_space_out' in key][
        #         0]: design_space})

    def update_post_processing_df(self):
        """Gathers the data for plotting the MDO graphs"""
        dataset = self.scenario.to_dataset()
        dataframe = dataset.copy()
        # quick fix to avoind NaN in the resulting dataframe
        # context : empty fields due to several calls to the same design space lead to NaN in dataframes
        # TODO: post proc this dataframe (or directly retrieve values from database) so that NaN values are replaced by already computed values
        dataframe = dataframe.fillna(-1)
        # dataframe = dataframe.rename(columns=rename_func)

        constraints_names = [
            constraint.name for constraint in self.scenario.formulation.optimization_problem.constraints
        ]
        objective_name = self.scenario.formulation.optimization_problem.objective.name

        def correct_var_name(varname: str) -> str:
            """Removes study name from variable name"""
            return ".".join(varname.split(".")[1:])

        return {
            "objective": np.array(dataframe[dataframe.FUNCTION_GROUP][objective_name].values),
            "variables": {
                correct_var_name(var): np.array(dataframe[dataframe.DESIGN_GROUP][var].values)
                for var in self.scenario.design_space.variable_names
            },
            "constraints": {
                correct_var_name(var): np.array(dataframe[dataframe.FUNCTION_GROUP][var].values)
                for var in constraints_names
            },
        }

    def add_design_space_inputs_to_local_data(self):
        """Add Design space inputs values to the local_data to store it in the dm"""
        problem = self.scenario.formulation.optimization_problem

        x = problem.solution.x_opt if problem.solution is not None else problem.design_space.get_current_value()
        current_idx = 0
        for k, v in problem.design_space._variables.items():
            k_size = v.size
            # WARNING we fill input in local_data that will be deleted by GEMSEO because they are not outputs ...
            # Only solution is to specify design space inputs as outputs of the mdoscenario
            self.io.data.update({k: x[current_idx : current_idx + k_size]})
            current_idx += k_size

    def get_namespace_from_var_name(self, var_name):
        subcoupling = self.scenario.disciplines[0]
        namespace_list = [
            full_name
            for full_name in subcoupling.get_input_data_names()
            if (var_name == full_name.split('.')[-1] or var_name == full_name)
        ]
        if len(namespace_list) == 1:
            return namespace_list[0]
        msg = f'Cannot find the variable {var_name} in the sub-coupling input grammar of the optim scenario {self.name}'
        raise ValueError(msg)
