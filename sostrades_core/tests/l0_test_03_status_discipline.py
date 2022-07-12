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
import numpy as np
import time
import _thread

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestStatusDiscipline(unittest.TestCase):
    """
    SoSDiscipline status test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dic_status = {
            'CONFIGURE': False,
            'RUNNING': False,
            'DONE': False}

        self.ee = ExecutionEngine('Test')

        ns_dict = {'ns_ac': 'Test'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'sostrades_core.sos_wrapping.test_discs.disc1_status.Disc1status'
        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod_path)

        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()

        values_dict = {}
        values_dict['Test.Disc1.a'] = 10.
        values_dict['Test.Disc1.b'] = 5.
        values_dict['Test.x'] = 2.
        self.ee.dm.set_values_from_dict(values_dict)

        self.process = self.ee.root_process

        if self.process.status in self.dic_status.keys():
            self.dic_status[self.process.status] = True

    def tearDown(self):
        pass

    # def test_01_execute(self):
    #
    #     time.sleep(0.2)
    #     self.ee.execute()
    #
    # def test_02_check_status(self):
    #
    #     t0 = time.time()
    #
    #     _thread.start_new_thread(self.test_01_execute, ())
    #
    #     while(time.time() - t0 < 3):
    #         if self.process._status in self.dic_status.keys():
    #             self.dic_status[self.process._status] = True
    #
    #     self.assertTrue(np.all(list(self.dic_status.values())),
    #                     'Missing status')
