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
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestSetupSoSDiscipline(unittest.TestCase):
    """
    Class to test dynamic inputs/outputs adding in setup_sos_discipline method
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.ns_test = 'Test'
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1_setup_sos_discipline.Disc1'
        self.mod2_path = f'{base_path}.disc1_setup_sos_discipline.Disc1ProxyCheck'

    def check_proxy_and_dm_assigned(self, proxy, expected=True):
        proxy_assigned = proxy.mdo_discipline_wrapp.wrapper.proxy is proxy
        dm_assigned = proxy.mdo_discipline_wrapp.wrapper.dm is self.ee.dm
        self.assertTrue(proxy_assigned == expected)
        self.assertTrue(dm_assigned == expected)

    def test_01_setup_sos_discipline(self):
        '''
        check discipline execution with dynamic inputs/outputs
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), [])
        self.assertTrue(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').empty)
        self.assertListEqual(
            self.ee.dm.get_all_namespaces_from_var_name('dyn_input_1'), [])

        AC_list = ['AC1', 'AC2']
        values_dict['Test.AC_list'] = AC_list

        # dynamic inputs/outputs are created during configure step,
        # based on AC_list value
        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertListEqual(
            self.ee.dm.get_all_namespaces_from_var_name('dyn_input_1'), ['Test.Disc1.AC1.dyn_input_1', 'Test.Disc1.AC2.dyn_input_1'])

        default_df = pd.DataFrame(
            [['AC1', 1.0], ['AC2', 1.0]], columns=['AC_name', 'value'])
        self.assertDictEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), default_df.to_dict())

        AC_list = ['AC1', 'AC3']
        dyn_input_2 = pd.DataFrame(
            [['AC1', 2.0], ['AC3', 3.0]], columns=['AC_name', 'value'])
        values_dict['Test.AC_list'] = AC_list
        values_dict['Test.Disc1.dyn_input_2'] = dyn_input_2

        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertIn('Test.Disc1.AC3.dyn_input_1',
                      list(self.ee.dm.data_id_map.keys()))
        self.assertNotIn('Test.Disc1.AC2.dyn_input_1',
                         list(self.ee.dm.data_id_map.keys()))
        self.assertEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), dyn_input_2.to_dict())

        AC_list = ['AC1', 'AC2']
        values_dict['Test.AC_list'] = AC_list
        values_dict[f'Test.Disc1.AC1.dyn_input_1'] = 2
        values_dict[f'Test.Disc1.AC2.dyn_input_1'] = 4
        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertIn('Test.Disc1.AC2.dyn_input_1',
                      list(self.ee.dm.data_id_map.keys()))
        self.assertNotIn('Test.Disc1.AC3.dyn_input_1',
                         list(self.ee.dm.data_id_map.keys()))
        self.assertDictEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), default_df.to_dict())

        self.ee.execute()
        for ac in AC_list:
            self.assertEqual(self.ee.dm.get_value(
                f'Test.Disc1.{ac}.dyn_output'), values_dict[f'Test.Disc1.{ac}.dyn_input_1']**2)

    def test_02_setup_sos_disciplines_signature_change(self):
        '''
        check discipline execution with dynamic inputs/outputs, with implementational checks assuring that the dynamic
        association of proxy and dm objects during configuration and de-association during run is OK
        '''

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        # this discipline is exactly as Disc1 but raises an exception if argument proxy or dm of wrapper are not None.
        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod2_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        # during configuration the proxy should be assigned
        proxy_disc_1 = self.ee.dm.get_disciplines_with_name('Test.Disc1')[0]
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=True)

        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), [])
        self.assertTrue(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').empty)
        self.assertListEqual(
            self.ee.dm.get_all_namespaces_from_var_name('dyn_input_1'), [])

        AC_list = ['AC1', 'AC2']
        values_dict['Test.AC_list'] = AC_list

        # dynamic inputs/outputs are created during configure step,
        # based on AC_list value
        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertListEqual(
            self.ee.dm.get_all_namespaces_from_var_name('dyn_input_1'), ['Test.Disc1.AC1.dyn_input_1', 'Test.Disc1.AC2.dyn_input_1'])

        default_df = pd.DataFrame(
            [['AC1', 1.0], ['AC2', 1.0]], columns=['AC_name', 'value'])
        self.assertDictEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), default_df.to_dict())

        AC_list = ['AC1', 'AC3']
        dyn_input_2 = pd.DataFrame(
            [['AC1', 2.0], ['AC3', 3.0]], columns=['AC_name', 'value'])
        values_dict['Test.AC_list'] = AC_list
        values_dict['Test.Disc1.dyn_input_2'] = dyn_input_2

        # during configuration the proxy should be assigned
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=True)

        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertIn('Test.Disc1.AC3.dyn_input_1',
                      list(self.ee.dm.data_id_map.keys()))
        self.assertNotIn('Test.Disc1.AC2.dyn_input_1',
                         list(self.ee.dm.data_id_map.keys()))
        self.assertEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), dyn_input_2.to_dict())

        AC_list = ['AC1', 'AC2']
        values_dict['Test.AC_list'] = AC_list
        values_dict[f'Test.Disc1.AC1.dyn_input_1'] = 2
        values_dict[f'Test.Disc1.AC2.dyn_input_1'] = 4
        self.ee.load_study_from_input_dict(values_dict)

        self.assertListEqual(self.ee.dm.get_value('Test.AC_list'), AC_list)
        self.assertIn('Test.Disc1.AC2.dyn_input_1',
                      list(self.ee.dm.data_id_map.keys()))
        self.assertNotIn('Test.Disc1.AC3.dyn_input_1',
                         list(self.ee.dm.data_id_map.keys()))
        self.assertDictEqual(self.ee.dm.get_value(
            'Test.Disc1.dyn_input_2').to_dict(), default_df.to_dict())

        # during configuration the proxy should be assigned
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=True)

        # during execution the lack of crash assures that the proxy and dm are not assigned
        self.ee.execute()

        # right after execution the proxy and dm should remain unassigned until further configuration
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=False)
        for ac in AC_list:
            self.assertEqual(self.ee.dm.get_value(
                f'Test.Disc1.{ac}.dyn_output'), values_dict[f'Test.Disc1.{ac}.dyn_input_1']**2)

        # upon reconfiguration and re-execution we should have same behaviour
        values_dict[f'Test.Disc1.AC2.dyn_input_1'] = 10
        self.ee.load_study_from_input_dict(values_dict)
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=True)

        self.ee.execute()
        self.check_proxy_and_dm_assigned(proxy_disc_1, expected=False)
