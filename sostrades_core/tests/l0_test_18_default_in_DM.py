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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from logging import Handler


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestDefaultInDM(unittest.TestCase):
    """
    Default value in data manager test class
    """

    def setUp(self):

        self.name = 'EETests'
        self.exec_eng = ExecutionEngine(self.name)
        self.repo = 'sostrades_core.sos_processes.test'
        self.exec_eng.select_root_process(self.repo,
                                          'test_disc1_disc2_coupling')

        self.exec_eng2 = ExecutionEngine(self.name)
        self.my_handler = UnitTestHandler()
        self.exec_eng2.logger.addHandler(self.my_handler)
        self.exec_eng2.select_root_process(self.repo,
                                           'test_disc1_disc2_couplingdefault')

    def test_01_default_dm(self):
        """
        Function to test the same coupling using values loaded in the data_dict of the datamanger
        and values entered in DESC_IN

        Check if the function set_dynamic_default_values is working
        """
        values_dict = {}
        a = 10.
        values_dict['EETests.Disc1.a'] = a
        values_dict['EETests.Disc1.b'] = 40.
        values_dict['EETests.Disc2.power'] = 2
        values_dict['EETests.Disc2.constant'] = -10.
        values_dict['EETests.x'] = 3.
        self.exec_eng.load_study_from_input_dict(values_dict)
#         res = self.exec_eng.execute()

        values_dict2 = {}
        values_dict2['EETests.Disc2.power'] = 2
        values_dict2['EETests.Disc2.constant'] = -10.
        values_dict2['EETests.x'] = 3.
        self.exec_eng2.load_study_from_input_dict(values_dict2)

        default_b = self.exec_eng2.dm.get_data('EETests.Disc1.b', 'default')
        value_b = self.exec_eng2.dm.get_value('EETests.Disc1.b')

        # Check if the function set_dynamic_default_values is working
        self.assertEqual(default_b, 4 * a)
        self.assertEqual(value_b, 4 * a)
#         msg_log_error = 'Try to set a default value for the variable c in Disc1 which is not an input of this discipline '
#         msg = f'{msg_log_error} not in {self.my_handler.msg_list}'
#         self.assertTrue(msg_log_error in self.my_handler.msg_list, msg)
#         res2 = self.exec_eng2.execute()

        # Check that res2 equals res1 : Disc1.a was loaded from default value
        # in DESC_IN
#         self.assertEqual(res, res2, "results are not equal")
