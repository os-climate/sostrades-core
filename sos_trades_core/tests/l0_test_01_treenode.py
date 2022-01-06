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
Treenode test suite
'''
import unittest

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TestTreenode(unittest.TestCase):
    """
    Treenode test class
    """

    def setUp(self):
        self.repo = 'sos_trades_core.sos_processes.test'

    def tearDown(self):
        pass

    def test_01_ordered_disc_and_recursive_tree(self):
        '''
        Test if the order of the disc in the tree is the ordered defined by the MDAChain
        '''
        print('\nin multilevel SoSCoupling unit test03 \n')

        namespace = 'study'
        ee = ExecutionEngine(namespace)
        ns_dict = {'ns_ac': namespace}
        ee.ns_manager.add_ns_def(ns_dict)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        print('*********** CONFIGURE 1 ****************')
        ee.configure()
        treeview = ee.display_treeview_nodes()

        soscoupling = ee.root_process

        sub_disc_sos1_names = [
            disc.sos_name for disc in soscoupling.sos_disciplines]
        gems_order_sub_disc_names = [
            disc.sos_name for disc in soscoupling.ordered_disc_list]

        print('predefined sub_disc order', sub_disc_sos1_names)
        print('gems order', gems_order_sub_disc_names)
        first_sub_disc = treeview.split('study')[-1].split('|_')[1].strip()
        second_sub_disc = treeview.split('study')[-1].split('|_')[2].strip()

        self.assertListEqual(gems_order_sub_disc_names, [
                             first_sub_disc, second_sub_disc],
                             'The order of sub disciplines in a coupling is different from the MDAChain execution order')

    def test_02_treeview_to_dict(self):

        namespace = 'study'
        ee = ExecutionEngine(namespace)
        ns_dict = {'ns_ac': namespace}
        ee.ns_manager.add_ns_def(ns_dict)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        print('*********** CONFIGURE 1 ****************')
        ee.configure()
        treeview = ee.display_treeview_nodes()
        tw_object = ee.get_treeview()
        tw_dict = tw_object.to_dict()
        self.assertEqual(tw_dict['name'], namespace)
        self.assertEqual(tw_dict['status'], 'CONFIGURE')
        self.assertEqual(tw_dict['full_namespace'], namespace)
        disc_list = ['Disc1', 'Disc2']
        for child in tw_dict['children']:
            self.assertIn(child['name'], disc_list)
            self.assertIn(child['full_namespace'], [
                          f'{namespace}.{disc}' for disc in disc_list])
            self.assertEqual(child['children'], [])
