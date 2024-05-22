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
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestMaturity(unittest.TestCase):
    """
    Class to test the maturity of SoSDiscipline/Coupling
    """

    def setUp(self):
        # Define maturities

        self.fake = 'Fake'
        self.dev = 'Research'
        self.official = 'Official'
        self.official_validated = 'Official Validated'

        self.maturity_list = [self.fake, self.dev,
                              self.official, self.official_validated]

        self.repo = 'sostrades_core.sos_processes.test'
        self.name = 'EETests'
        self.exec_eng = ExecutionEngine(self.name)
        self.exec_eng.select_root_process(self.repo,
                                          'test_disc1_disc2_coupling')

        values_dict = {}
        values_dict['EETests.Disc1.a'] = 10.
        values_dict['EETests.Disc1.b'] = 20.
        values_dict['EETests.Disc2.power'] = 2
        values_dict['EETests.Disc2.constant'] = -10.
        values_dict['EETests.x'] = 3.
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

    def test_01_discipline_maturity(self):

        disc1 = self.exec_eng.root_process.proxy_disciplines[0]
        maturity_to_test = disc1.get_maturity()
        ref_maturity = self.fake
        self.assertEqual(maturity_to_test, ref_maturity,
                         "Maturities are not equal, expected {} maturity, got {}".format(ref_maturity, maturity_to_test))

    def test_02_coupling_maturity(self):

        maturity_to_test = self.exec_eng.root_process.get_maturity()
        ref_maturity = dict(zip(self.maturity_list, [2, 0, 0, 0]))
        self.assertDictEqual(maturity_to_test, ref_maturity,
                             "coupling maturities doesn't match")
