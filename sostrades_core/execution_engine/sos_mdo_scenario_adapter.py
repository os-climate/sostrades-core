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
from gemseo.scenarios.mdo_scenario import MDOScenario


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
        self.scenario = MDOScenario(disciplines,
                         self.formulation,
                         self.objective_name,
                         design_space,
                         name=self.name,
                                    maximize_objective = maximize_objective)

        super().__init__(self.scenario,
            input_names=[],
            output_names=[],name=f'{self.name}_adapter'
        )
        self.scenario.eval_mode = mdo_options.pop('eval_mode')
        self.scenario.eval_jac = mdo_options.pop('eval_jac')
        self.scenario.dict_desactivated_elem =  mdo_options.pop('dict_desactivated_elem')
        self.scenario.input_design_space = mdo_options.pop('input_design_space')
        self.scenario.desactivate_optim_out_storage = mdo_options.pop('desactivate_optim_out_storage')

        self.scenario.mdo_options = mdo_options

        self.reduced_dm = reduced_dm
        self.activated_variables = self.scenario.formulation.design_space.variable_names
        self.is_sos_coupling = False

