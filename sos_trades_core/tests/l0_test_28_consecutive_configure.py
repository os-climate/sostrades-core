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


class TestConsecutiveConfigure(unittest.TestCase):
    """
    Class to test consecutive configure step
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Coupling'
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.repo = 'sos_trades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'

    def test_01_consecutive_configure(self):

        ns_dict = {'ns_ac': self.namespace}

        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc1.Disc1'
        builder_list = self.factory.get_builder_from_module('Disc1', mod_list)

        scatter_builder = self.exec_eng.factory.create_scatter_builder(
            'scatter', 'name_list', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_builder)
        # scatter without instance

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        self.assertListEqual(
            ['scatter'], [disc.name for disc in sub_disciplines])

        # scatter instances : ['name_1', 'name_2']

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        print('current', self.exec_eng.factory.current_discipline.sos_disciplines)
        self.assertListEqual(
            ['scatter', 'name_1', 'name_2'], [disc.name for disc in sub_disciplines])

        # scatter instances : ['name_1', 'name_2']

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        self.assertListEqual(
            ['scatter', 'name_1', 'name_2'], [disc.name for disc in sub_disciplines])

        # scatter instances : ['name_1', 'name_3']

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_3']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        self.assertNotIn('name_2', [disc.name for disc in sub_disciplines])
        self.assertIn('name_3', [disc.name for disc in sub_disciplines])

        # scatter instances : ['name_1', 'name_3', 'name_4']

        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_3', 'name_4']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        self.assertIn('name_4', [disc.name for disc in sub_disciplines])

        # scatter without instance

        dict_values = {self.study_name + '.name_list': []}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        sub_disciplines = self.exec_eng.factory.sos_disciplines
        self.assertListEqual(
            ['scatter'], [disc.name for disc in sub_disciplines])

    def test_02_consecutive_configure_with_gather(self):

        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_barrierr'}

        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # instantiate factory # get instantiator from Discipline class
        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', cls_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.study_name +
                    '.Disc2.name_1.constant'] = constant1
        private_val[self.study_name + '.Disc2.name_1.power'] = power1
        private_val[self.study_name +
                    '.Disc2.name_2.constant'] = constant2
        private_val[self.study_name + '.Disc2.name_2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.name_1.x'] = x1
        private_val[self.study_name + '.name_2.x'] = x2
        private_val[self.study_name + '.Disc1.name_1.a'] = a1
        private_val[self.study_name + '.Disc1.name_2.a'] = a2
        private_val[self.study_name + '.Disc1.name_1.b'] = b1
        private_val[self.study_name + '.Disc1.name_2.b'] = b2

        private_val[self.study_name +
                    '.name_list'] = ['name_1', 'name_3']
        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.display_treeview_nodes()

        private_val2 = {}
        private_val2[self.study_name +
                     '.Disc2.name_3.constant'] = constant2
        private_val2[self.study_name + '.Disc2.name_3.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val2[self.study_name + '.name_3.x'] = x1
        private_val2[self.study_name + '.Disc1.name_3.a'] = a1
        private_val2[self.study_name + '.Disc1.name_3.b'] = b1
        self.exec_eng.load_study_from_input_dict(private_val2)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y3 = self.exec_eng.dm.get_value(self.study_name + '.name_3.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y3, a1 * x1 + b1)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z3 = self.exec_eng.dm.get_value(self.study_name + '.name_3.z')
        self.assertEqual(z1, constant1 + y1**power1)
        self.assertEqual(z3, constant2 + y3**power2)

        z_dict = self.exec_eng.dm.get_value(
            self.study_name + '.z_dict')

        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_1': z1, 'name_3': z3})

        y_dict = self.exec_eng.dm.get_value(
            self.study_name + '.y_dict')

        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_1': y1, 'name_3': y3})
