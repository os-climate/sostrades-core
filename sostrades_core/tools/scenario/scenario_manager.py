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

from sostrades_core.tools.scenario.scenario import Scenario

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""


class ScenarioManager:
    """
    Class to manage all scenarios
    """

    def __init__(self, name):
        """
        Constructor
        """
        self.name = name
        self.list_scenarios_ids = []
        self.list_scenarios = []

    def add_scenario(self, name):
        """
        Add new scenario
        """
        new_scenario = Scenario(name, scenario_manager=self)
        self.list_scenarios.append(new_scenario)
        self.list_scenarios_ids.append(new_scenario.name)
        return new_scenario

    def get_scenario(self, name):
        """
        Get scenario from name
        """
        index = self.list_scenarios_ids.index(name)
        return self.list_scenarios[index]
