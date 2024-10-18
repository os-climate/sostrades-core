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
import logging

from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from sostrades_core.execution_engine.sos_mdo_scenario import SoSMDOScenario

class SoSMDOScenarioAdapter(MDOScenarioAdapter):
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
                 name,
                 formulation,
                 objective_name,
                 design_space,
                 maximize_objective,
                 input_names,
                 output_names,
                 logger: logging.Logger,
                 reduced_dm=None,
                 mdo_options={}):
        """
        Constructor
        """
        self.logger = logger
        self.formulation = formulation
        self.objective_name = objective_name
        self.name = name
        self.scenario = SoSMDOScenario(disciplines, self.formulation, self.objective_name, design_space, self.name,
                                       maximize_objective, logger)

        super().__init__(self.scenario,
                         input_names=[],
                         output_names=output_names, name=f'{self.name}_adapter'
        )
        self.scenario.eval_mode = mdo_options.pop('eval_mode')
        self.scenario.eval_jac = mdo_options.pop('eval_jac')
        self.scenario.dict_desactivated_elem =  mdo_options.pop('dict_desactivated_elem')
        self.scenario.input_design_space = mdo_options.pop('input_design_space')
        self.scenario.desactivate_optim_out_storage = mdo_options.pop('desactivate_optim_out_storage')

        self.mdo_options = mdo_options

        self.reduced_dm = reduced_dm
        self.activated_variables = self.scenario.formulation.design_space.variable_names
        self.is_sos_coupling = False

    def _run(self) -> None:
        self._pre_run()
        # with LoggingContext(LOGGING_SETTINGS.logger, level=self.__scenario_log_level):
        self.scenario.execute(**self.mdo_options)
        if self.scenario.eval_mode:
            self._retrieve_top_level_outputs()
        else:
            self._post_run()

        # I think everything is in the post run
        #        self.scenario_outputs_dict =
        # self.scenario_outputs = [discipline.get_output_data()
        #                          for discipline in self.disciplines]
        # for data in outputs:
        #     self.io.data.update(data)
        #
        # self.add_design_space_inputs_to_local_data()
        # # save or not the output of design space for post processings
        # if not self.desactivate_optim_out_storage:
        #     self.update_design_space_out()
        #     post_processing_mdo_data = {}
        #     if not self.eval_mode:
        #         post_processing_mdo_data = self.update_post_processing_df()
        #     self.io.data.update({
        #         [key for key in self.get_output_data_names() if self.POST_PROC_MDO_DATA in key][
        #             0]: post_processing_mdo_data})

    def update_post_processing_df(self):
        """Gathers the data for plotting the MDO graphs"""
        dataset = self.to_dataset()
        dataframe = dataset.copy()
        # quick fix to avoind NaN in the resulting dataframe
        # context : empty fields due to several calls to the same design space lead to NaN in dataframes
        # TODO: post proc this dataframe (or directly retrieve values from database) so that NaN values are replaced by already computed values
        dataframe = dataframe.fillna(-1)
        # dataframe = dataframe.rename(columns=rename_func)

        constraints_names = [constraint.name for constraint in
                             self.formulation.optimization_problem.constraints]
        objective_name = self.formulation.optimization_problem.objective.name

        def correct_var_name(varname: str) -> str:
            """removes study name from variable name"""
            corrected_var_name = ".".join(varname.split(".")[1:])
            return corrected_var_name

        post_processing_mdo_data = {
            "objective": np.array(dataframe[dataframe.FUNCTION_GROUP][objective_name].values),
            "variables": {correct_var_name(var): np.array(dataframe[dataframe.DESIGN_GROUP][var].values) for var in
                          self.design_space.variable_names},
            "constraints": {correct_var_name(var): np.array(dataframe[dataframe.FUNCTION_GROUP][var].values) for var in
                            constraints_names}
        }
        return post_processing_mdo_data
