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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class TestNSManager(unittest.TestCase):
    """
    Namespace manager test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'MyCase'
        self.exec_eng = ExecutionEngine(self.name)

    def test_01_nsm_basic(self):
        nsm = self.exec_eng.ns_manager
        test = {}
        ns_key1 = 'ns_ac'
        ns1_value = 'toto.AC'
        ns1 = {ns_key1: ns1_value}
        test.update(ns1)
        nsm.add_ns_def(ns1)
        ns_key2 = 'ns_bc'
        ns2_value = 'toto.bc'
        ns2 = {ns_key2: 'toto.bc'}
        test.update(ns2)
        nsm.add_ns_def(ns2)
        self.assertEqual(nsm.shared_ns_dict[ns_key1].get_value(), ns1_value)
        self.assertEqual(nsm.shared_ns_dict[ns_key2].get_value(), ns2_value)

        # ns already exists with same value
        nsm.add_ns_def(ns1)
        self.assertEqual(nsm.shared_ns_dict[ns_key1].get_value(), ns1_value)
        # ns already exists but different value
        ns1_val2 = {ns_key1: ns2_value}
        nsm.add_ns_def(ns1_val2)
        self.assertEqual(nsm.shared_ns_dict[ns_key1].get_value(), ns2_value)
        # reset and redo
        nsm.reset_current_disc_ns()
        ns2_val1 = {ns_key2: ns1_value}
        nsm.add_ns_def(ns2_val1)
        self.assertEqual(nsm.shared_ns_dict[ns_key2].get_value(), ns1_value)

    def test_02_nsm_check_ns_dict(self):
        nsm = self.exec_eng.ns_manager
        nsm.set_current_disc_ns('T.E')
        ns1 = {'ns_ac': 'AC'}
        nsm.add_ns_def(ns1)
        disc = SoSDiscipline('toto', self.exec_eng)
        nsm.create_disc_ns_info(disc)

        self.assertEqual(nsm.shared_ns_dict['ns_ac'].get_value(), 'AC')
        ns_dict = nsm.get_disc_ns_info(disc)

        self.assertEqual(ns_dict['local_ns'].get_value(), 'T.E.toto')
        self.assertListEqual(list(ns_dict.keys()), ['local_ns', 'others_ns'])

        self.assertEqual(ns_dict['others_ns']['ns_ac'].get_value(), 'AC')

    def test_03_nsm_current_ns_reset(self):
        nsm = self.exec_eng.ns_manager
        nsm.reset_current_disc_ns()
        self.assertEqual(nsm.current_disc_ns, None)

    def test_04_nsm_change_disc_ns(self):
        nsm = self.exec_eng.ns_manager
        nsm.set_current_disc_ns('T.E')
        nsm.change_disc_ns('..')
        self.assertEqual(nsm.current_disc_ns, 'T')
        nsm.change_disc_ns('..')
        self.assertEqual(nsm.current_disc_ns, None)
        nsm.change_disc_ns('SA')
        self.assertEqual(nsm.current_disc_ns, 'SA')
        nsm.change_disc_ns('toto')
        self.assertEqual(nsm.current_disc_ns, 'SA.toto')
