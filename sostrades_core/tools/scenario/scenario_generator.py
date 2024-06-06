'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from copy import deepcopy
from itertools import product

from pandas.core.common import flatten

from sostrades_core.tools.scenario.scenario_manager import ScenarioManager

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

class ScenarioGenerator:
    """
    Class to instantiate all classes of the chosen scenario (Products, Actors...) depending on the entry
    """

    def __init__(self, name=None, name_manager=None):
        """
        Constructor
        """
        self.name = name
        self.name_manager = name_manager
        self.scenario_manager = ScenarioManager(
            self.name_manager)
        self.scenarios_dict = {}
        self.scenarios_parameter = []

    def generate_scenarios(self, inputs_dict):
        """ generate scenarios for each products
        args:
            inputs_dict: dict of scenario parameters
            inputs_parameter: keys in inputs_dict
        """
        nb_scenario = 0
        values_parameter = []
        self.scenarios_parameter = []
        self.scenarios_dict = {}

        for input_name in inputs_dict.keys():
            self.scenarios_parameter.append(input_name)
            values_parameter.append(
                self.generate_combinations(inputs_dict[input_name]))

        for product_value in product(*values_parameter):
            inputs_scenario = {}
            nb_scenario += 1
            scenario_name = 'scenario_' + str(nb_scenario)

            for i in range(len(self.scenarios_parameter)):
                inputs_scenario[self.scenarios_parameter[i]] = list(product_value)[
                    i]
            if inputs_scenario != {}:
                self.scenarios_dict.update(
                    self.configure_scenario(inputs_scenario, scenario_name))

        return self.scenarios_dict

    def generate_combinations(self, input_value):
        if input_value is None:
            return []
        if isinstance(input_value, (float, int, str)):
            return [input_value]
        if isinstance(input_value, list):
            return input_value
        if isinstance(input_value, dict):
            val_in_dict = list(input_value.values())

            # patch to load dict of list using strings
            val_in_dict_copy = deepcopy(val_in_dict)
            val_in_dict = []
            for val in val_in_dict_copy:
                if isinstance(val, str):
                    val_in_dict.append(eval(val))
                else:
                    val_in_dict.append(val)

            combinations = []
            for product_val in product(*val_in_dict):
                combinations.append(list(flatten(list(product_val))))
            return combinations

    def configure_scenario(self, inputs_scenario, scenario_name):
        inputs = deepcopy(inputs_scenario)
        scenario = self.scenario_manager.add_scenario(
            scenario_name)
        if self.scenarios_parameter == []:
            self.scenarios_parameter = scenario.get_scenario_parameters()
        return {scenario_name: inputs}

    def get_scenarios_parameter(self):
        return self.scenarios_parameter
