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

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""


class Scenario:
    """
    Class to represent scenario object
    """

    def __init__(self, name, scenario_manager=None):
        """
        Constructor for scenario class
        """
        self.name = name
        self.scenario_manager = scenario_manager
        self.parameters = []

    def get_scenario_parameters(self):
        return self.parameters
