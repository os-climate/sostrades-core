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
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8S
"""
import unittest

from sostrades_core.tools.scenario.scenario_generator import ScenarioGenerator


class TestScenarioGenerator(unittest.TestCase):
    """
    Scenario generator test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.scenario_generator = ScenarioGenerator()

        dict_products = {'Airbus': [['CH19_Kero'], ['CH19_Kero', 'CH19_H2']],
                         'Boeing': [['BCH19_Kero'], ['BCH19_H2']]}
        self.dict_parameters = {'envscenarios': ['NPS', '2DS'],
                                'products': dict_products}
        self.result = [{'envscenarios': 'NPS', 'products': ['CH19_Kero', 'BCH19_Kero']},
                       {'envscenarios': 'NPS', 'products': [
                           'CH19_Kero', 'BCH19_H2']},
                       {'envscenarios': 'NPS', 'products': [
                           'CH19_Kero', 'CH19_H2', 'BCH19_Kero']},
                       {'envscenarios': 'NPS', 'products': [
                           'CH19_Kero', 'CH19_H2', 'BCH19_H2']},
                       {'envscenarios': '2DS', 'products': [
                           'CH19_Kero', 'BCH19_Kero']},
                       {'envscenarios': '2DS', 'products': [
                           'CH19_Kero', 'BCH19_H2']},
                       {'envscenarios': '2DS', 'products': [
                           'CH19_Kero', 'CH19_H2', 'BCH19_Kero']},
                       {'envscenarios': '2DS', 'products': ['CH19_Kero', 'CH19_H2', 'BCH19_H2']}]

    def test_01_generate_scenarios(self):
        '''
        default initialisation test
        '''
        self.generated_scenarios = self.scenario_generator.generate_scenarios(
            self.dict_parameters)

        self.assertListEqual(list(self.dict_parameters.keys()),
                             self.scenario_generator.get_scenarios_parameter(), 'Scenario parameters are incorrect')

        self.assertListEqual(list(self.generated_scenarios.keys()), [
                             f'scenario_{i}' for i in range(1, 9)], 'Scenario names are incorrect')

        self.assertListEqual(list(self.generated_scenarios.values(
        )), self.result, 'Generated scenarios are incorrect')


if __name__ == "__main__":
    unittest.main()
