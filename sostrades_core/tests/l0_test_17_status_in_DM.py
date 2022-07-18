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


class TestStatusDM(unittest.TestCase):
    """
    ProxyCoupling with status in DM.disciplines_dict test class
    """

    def setUp(self):

        self.name = 'EETests'
        self.repo = 'sostrades_core.sos_processes.test'
        self.exec_eng = ExecutionEngine(self.name)
        self.exec_eng.select_root_process(self.repo,
                                          'test_disc1_disc2_coupling')

        self.VIRTUAL = 'VIRTUAL'
        self.PENDING = 'PENDING'
        self.CONFIGURE = 'CONFIGURE'
        self.DONE = 'DONE'

    def test_01_pending_status(self):
        to_test = [self.exec_eng.dm.disciplines_dict[x]['status']
                   for x in list(self.exec_eng.dm.disciplines_dict.keys())]
        target = [self.CONFIGURE] * 3

        self.assertListEqual(
            to_test, target, "wrong status in disciplines_dict")

    def test_02_done_status(self):
        # modify DM
        values_dict = {}
        values_dict['EETests.Disc1.a'] = 10.
        values_dict['EETests.Disc1.b'] = 20.
        values_dict['EETests.Disc2.power'] = 2
        values_dict['EETests.Disc2.constant'] = -10.
        values_dict['EETests.x'] = 3.

        self.exec_eng.dm.set_values_from_dict(values_dict)
        res = self.exec_eng.execute()
        print(res)
        to_test = [self.exec_eng.dm.disciplines_dict[x]['status']
                   for x in list(self.exec_eng.dm.disciplines_dict.keys())]
        target = [self.DONE] * 3
        self.assertListEqual(
            to_test, target, "wrong status in disciplines_dict")
