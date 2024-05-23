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
import _thread
import time
import unittest

import numpy as np

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestStatusCoupling(unittest.TestCase):
    """
    SoSCoupling status test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.dic_status = {'CONFIGURE': False,
                           'RUNNING': False,
                           'DONE': False}
        self.exec_eng = ExecutionEngine(self.name)

        ns_dict = {'ns_ac': self.name}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mod_path = f'{self.base_path}.disc1_status.Disc1status'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_path)

        mod_path = f'{self.base_path}.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_path)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.exec_eng.configure()

        self.process = self.exec_eng.root_process

        # First process status check before configure
        if self.process.status in self.dic_status.keys():
            self.dic_status[self.process.status] = True

    def tearDown(self):
        pass

    def test_01_execute(self):

        # modify DM
        values_dict = {}
        values_dict['EETests.Disc1.a'] = 10.
        values_dict['EETests.Disc1.b'] = 5.
        values_dict['EETests.x'] = 2.
        values_dict['EETests.Disc2.constant'] = 4.
        values_dict['EETests.Disc2.power'] = 2

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.execute()

    def test_02_check_status(self):

        t0 = time.time()

        _thread.start_new_thread(self.test_01_execute, ())

        while(time.time() - t0 < 3):
            if self.process.status in self.dic_status.keys():
                self.dic_status[self.process.status] = True

        self.assertTrue(np.all(list(self.dic_status.values())),
                        'Missing status')
