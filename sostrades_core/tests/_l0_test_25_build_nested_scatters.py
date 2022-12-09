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
from sostrades_core.execution_engine.proxy_discipline_gather import ProxyDisciplineGather


class TestBuildScatter(unittest.TestCase):
    """
    Scatter build test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Coupling'
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sostrades_core.sos_processes.test'
        self.sub_proc = 'test_disc1_disc2_coupling'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory

    def test_01_build_coupling_of_scatter(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        # get instantiator from local Process
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)

        self.exec_eng.configure()

        self.assertListEqual([s.disc for s in scatter_list], self.exec_eng.root_process.proxy_disciplines,
                             f'\nlist of scatter disciplines to build {[d.sos_name for d in scatter_list]} is different ' +
                             f'than list of built ones {[d.sos_name for d in self.exec_eng.root_process.proxy_disciplines]}')

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
        self.exec_eng.load_study_from_input_dict(private_val)

        # self.assertListEqual(list(self.exec_eng.dm.disciplines_id_map.keys()),
        #                      ['MyCase', 'MyCase.Disc2', 'MyCase.Disc1', 'MyCase.Disc2.name_1',
        #                          'MyCase.Disc2.name_2', 'MyCase.Disc1.name_1', 'MyCase.Disc1.name_2']
        #                      )

        self.assertListEqual(list(self.exec_eng.dm.disciplines_id_map.keys()),
                             ['MyCase', 'MyCase.Disc1', 'MyCase.Disc2', 'MyCase.Disc1.name_1', 'MyCase.Disc1.name_2',
                              'MyCase.Disc2.name_1', 'MyCase.Disc2.name_2']
                             )
        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

    def test_02_buil_scatter_of_coupling(self):

        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # instantiate factory # get instantiator from Discipline class
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        # builder_list is a list of builders from self.sub_proc
        scatter = self.exec_eng.factory.create_scatter_builder('scatter',
                                                               'name_list', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter)

        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.assertListEqual(list(self.exec_eng.dm.disciplines_id_map.keys()),
                             ['MyCase', 'MyCase.scatter', 'MyCase.scatter.name_1.Disc1',
                              'MyCase.scatter.name_1.Disc2', 'MyCase.scatter.name_2.Disc1',
                              'MyCase.scatter.name_2.Disc2'])

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.study_name +
                    '.scatter.name_1.Disc2.constant'] = constant1
        private_val[self.study_name + '.scatter.name_1.Disc2.power'] = power1
        private_val[self.study_name +
                    '.scatter.name_2.Disc2.constant'] = constant2
        private_val[self.study_name + '.scatter.name_2.Disc2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.study_name + '.scatter.name_1.x'] = x1
        private_val[self.study_name + '.scatter.name_2.x'] = x2
        private_val[self.study_name + '.scatter.name_1.Disc1.a'] = a1
        private_val[self.study_name + '.scatter.name_2.Disc1.a'] = a2
        private_val[self.study_name + '.scatter.name_1.Disc1.b'] = b1
        private_val[self.study_name + '.scatter.name_2.Disc1.b'] = b2
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.scatter.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.scatter.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.scatter.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.scatter.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

    def test_03_build_scatter_of_scatter_of_coupling(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        mydict = {'input_name': 'scenario_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'scenario_name',
                  'scatter_ns': 'ns_scenario'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('scenario_list', mydict)

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)

        scatter = self.factory.create_scatter_builder(
            'scatter_ac', 'name_list', builder_list)
        scatter_sc = self.factory.create_scatter_builder(
            'scatter_sc', 'scenario_list', scatter)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_sc)
        self.exec_eng.configure()

        scenario_list = ['scenario_1', 'scenario_2']
        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.scenario_list': scenario_list}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.assertListEqual(['MyCase',
                              'MyCase.scatter_sc',
                              'MyCase.scatter_sc.scenario_1',
                              'MyCase.scatter_sc.scenario_2',
                              'MyCase.scatter_sc.scenario_1.name_1.Disc1',
                              'MyCase.scatter_sc.scenario_1.name_1.Disc2',
                              'MyCase.scatter_sc.scenario_1.name_2.Disc1',
                              'MyCase.scatter_sc.scenario_1.name_2.Disc2',
                              'MyCase.scatter_sc.scenario_2.name_1.Disc1',
                              'MyCase.scatter_sc.scenario_2.name_1.Disc2',
                              'MyCase.scatter_sc.scenario_2.name_2.Disc1',
                              'MyCase.scatter_sc.scenario_2.name_2.Disc2'],
                             list(self.exec_eng.dm.disciplines_id_map.keys()))

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        for scenario in scenario_list:
            private_val[self.study_name + '.scatter_sc.' + scenario +
                        '.name_1.Disc2.constant'] = constant1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_1.Disc2.power'] = power1
            private_val[self.study_name + '.scatter_sc.' + scenario +
                        '.name_2.Disc2.constant'] = constant2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_2.Disc2.power'] = power2

            x1 = 2
            a1 = 3
            b1 = 4
            x2 = 4
            a2 = 6
            b2 = 2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_1.x'] = x1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_2.x'] = x2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_1.Disc1.a'] = a1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_2.Disc1.a'] = a2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_1.Disc1.b'] = b1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_2.Disc1.b'] = b2
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        scenario = 'scenario_1'
        y1 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

    def test_04_build_scatter_of_coupling_of_scatter(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        mydict = {'input_name': 'scenario_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'scenario_name',
                  'scatter_ns': 'ns_scenario'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('scenario_list', mydict)

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)

        scatter_list = []
        for builder in builder_list:
            scatter = self.factory.create_scatter_builder(
                builder.sos_name, 'name_list', builder)
            scatter_list.append(scatter)

        scatter_sc = self.exec_eng.factory.create_scatter_builder(
            'scatter_sc', 'scenario_list', scatter_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_sc)
        self.exec_eng.configure()

        scenario_list = ['scenario_1', 'scenario_2']
        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.scenario_list': scenario_list}

        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values = {self.study_name +
                       '.name_list': ['name_1', 'name_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()
        self.assertCountEqual(['MyCase', 'MyCase.scatter_sc', 'MyCase.scatter_sc.scenario_1.Disc2',
                               'MyCase.scatter_sc.scenario_1.Disc1', 'MyCase.scatter_sc.scenario_2.Disc2',
                               'MyCase.scatter_sc.scenario_2.Disc1', 'MyCase.scatter_sc.scenario_1.Disc2.name_1',
                               'MyCase.scatter_sc.scenario_1.Disc2.name_2', 'MyCase.scatter_sc.scenario_2.Disc2.name_1',
                               'MyCase.scatter_sc.scenario_2.Disc2.name_2', 'MyCase.scatter_sc.scenario_1.Disc1.name_1',
                               'MyCase.scatter_sc.scenario_1.Disc1.name_2', 'MyCase.scatter_sc.scenario_2.Disc1.name_1',
                               'MyCase.scatter_sc.scenario_2.Disc1.name_2'],
                              list(self.exec_eng.dm.disciplines_id_map.keys()))

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        for scenario in scenario_list:
            private_val[self.study_name + '.scatter_sc.' + scenario +
                        '.Disc2.name_1.constant'] = constant1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc2.name_1.power'] = power1
            private_val[self.study_name + '.scatter_sc.' + scenario +
                        '.Disc2.name_2.constant'] = constant2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc2.name_2.power'] = power2

            x1 = 2
            a1 = 3
            b1 = 4
            x2 = 4
            a2 = 6
            b2 = 2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_1.x'] = x1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.name_2.x'] = x2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc1.name_1.a'] = a1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc1.name_2.a'] = a2
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc1.name_1.b'] = b1
            private_val[self.study_name + '.scatter_sc.' +
                        scenario + '.Disc1.name_2.b'] = b2

        for ns in self.exec_eng.ns_manager.ns_list:
            print(ns.name, ns.value)
        for key in self.exec_eng.dm.data_id_map:
            print(key)
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        scenario = 'scenario_1'
        y1 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(
            self.study_name + '.scatter_sc.' + scenario + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

    def test_05_build_coupling_of_scatter_with_auto_gather(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns_in': 'ns_barrierr',
                  'gather_ns_out': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        # instantiate factory # get instantiator from Discipline class
        cls_list = self.factory.get_builder_from_process(repo=self.repo,
                                                         mod_id=self.sub_proc)  # get instantiator from Process
        scatter_builder_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', cls_list, autogather=True)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            scatter_builder_list)
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
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

        z_dict = self.exec_eng.dm.get_value(
            self.study_name + '.z_dict')
        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_1': z1, 'name_2': z2})

        y_dict = self.exec_eng.dm.get_value(
            self.study_name + '.y_dict')
        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_1': y1, 'name_2': y2})

        # test data_in/data_out referencing in dm
        disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2')[0]
        disc1_name1 = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1.name_1')[0]

        # Check that two disciplines are called Mycase.Disc1 the scatter and
        # its associated gather
        self.assertEqual(len(disc1), 2)
        for disc in disc1:
            if isinstance(disc, ProxyDisciplineGather):
                disc1_gather = disc

        self.assertEqual(disc1_gather.get_var_full_name('name_1.y', disc1_gather.get_data_in()),
                         disc1_name1.get_var_full_name('y', disc1_name1.get_data_out()))
        self.assertEqual(self.exec_eng.dm.get_value('MyCase.name_1.y'), 10)
        # scatter output y is referenced in dm
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.y'), disc1_name1.get_data_out()['y'])
        # gather input name_1.y is not referenced in dm
        self.assertNotEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.y'), disc1_gather.get_data_in()['name_1.y'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), disc1_gather.get_data_in()['name_list']['value'])
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_list'), disc2.get_data_in()['name_list'])

        # check user_level of gather inputs
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.y', 'user_level'), 1)
        disc2_name_1 = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc2.name_1')[0]
        self.assertEqual(disc2_name_1.get_data_in()['y']['user_level'], 1)
        disc1_name_1 = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Disc1.name_1')[0]
        self.assertEqual(disc1_name_1.get_data_out()['y']['user_level'], 1)
        gather_disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[
            1]
        self.assertEqual(gather_disc1.get_data_in()['name_1.y']['user_level'], 3)

    def test_06_build_scatter_of_scatter_of_coupling_of_scatter(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        mydict = {'input_name': 'scenario_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'scenario_name',
                  'scatter_ns': 'ns_scenario'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('scenario_list', mydict)

        mydict = {'input_name': 'toplevel_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'toplevel_name',
                  'scatter_ns': 'ns_toplevel'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('toplevel_list', mydict)

        mydict = {'input_name': 'secondlevel_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'secondlevel_name',
                  'scatter_ns': 'ns_secondlevel'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('secondlevel_list', mydict)

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)

        scatter_list = []
        for builder in builder_list:
            scatter = self.factory.create_scatter_builder(
                builder.sos_name, 'name_list', builder)
            scatter_list.append(scatter)

        scatter_sc = self.factory.create_scatter_builder(
            'scatter_sc', 'scenario_list', scatter_list)

        scatter_second_level = self.factory.create_scatter_builder(
            'scatter_second_level', 'secondlevel_list', scatter_sc)
        scatter_top_level = self.exec_eng.factory.create_scatter_builder(
            'scatter_sc', 'toplevel_list', scatter_second_level)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            scatter_top_level)
        self.exec_eng.configure()

        toplevel_list = ['toto_1', 'toto_2', 'toto_3']
        dict_values = {self.study_name +
                       '.toplevel_list': toplevel_list}
        self.exec_eng.load_study_from_input_dict(dict_values)

        secondlevel_list = ['tata_1']
        dict_values = {self.study_name +
                       '.secondlevel_list': secondlevel_list}
        self.exec_eng.load_study_from_input_dict(dict_values)

        scenario_list = ['scenario_1', 'scenario_2']
        # User fill in the fields in the GUI
        dict_values = {self.study_name +
                       '.scenario_list': scenario_list}
        self.exec_eng.load_study_from_input_dict(dict_values)

        name_list = ['name_1', 'name_2']
        dict_values = {self.study_name +
                       '.name_list': name_list}
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        list_disciplines = list(self.exec_eng.dm.disciplines_id_map.keys())
        list_disciplines_wo_scatter = [disc for disc in list_disciplines if disc.endswith(
            'Disc1') or disc.endswith('Disc2')]

        self.assertEqual(len(toplevel_list) * len(secondlevel_list) * len(name_list)
                         * len(scenario_list), len(list_disciplines_wo_scatter))

    def test_07_build_coupling_of_scatter_with_auto_gather_from_process(self):
        # load process in GUI
        builders = self.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
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
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

        z_dict = self.exec_eng.dm.get_value(
            self.study_name + '.z_dict')
        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_1': z1, 'name_2': z2})

        y_dict = self.exec_eng.dm.get_value(
            self.study_name + '.y_dict')
        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_1': y1, 'name_2': y2})

    def test_08_build_scatter_of_scatter_of_coupling_of_scatter_with_load_study_from_input_dict(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')

        mydict = {'input_name': 'scenario_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'scenario_name',
                  'scatter_ns': 'ns_scenario'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('scenario_list', mydict)

        mydict = {'input_name': 'toplevel_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'toplevel_name',
                  'scatter_ns': 'ns_toplevel'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('toplevel_list', mydict)

        mydict = {'input_name': 'secondlevel_list',

                  'input_ns': 'ns_barrierr',
                  'output_name': 'secondlevel_name',
                  'scatter_ns': 'ns_secondlevel'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('secondlevel_list', mydict)
        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)

        scatter_list = []
        for builder in builder_list:
            scatter = self.factory.create_scatter_builder(
                builder.sos_name, 'name_list', builder)
            scatter_list.append(scatter)

        scatter_sc = self.factory.create_scatter_builder(
            'scatter_sc', 'scenario_list', scatter_list)

        scatter_second_level = self.factory.create_scatter_builder(
            'scatter_second_level', 'secondlevel_list', scatter_sc)
        scatter_top_level = self.exec_eng.factory.create_scatter_builder(
            'scatter_sc', 'toplevel_list', scatter_second_level)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            scatter_top_level)

        self.exec_eng.configure()

        toplevel_list = ['toto_1', 'toto_2', 'toto_3']
        dict_values = {self.study_name +
                       '.toplevel_list': toplevel_list}

        secondlevel_list = ['tata_1']
        dict_values.update({self.study_name +
                            '.secondlevel_list': secondlevel_list})

        scenario_list = ['scenario_1', 'scenario_2']
        # User fill in the fields in the GUI
        dict_values.update({self.study_name +
                            '.scenario_list': scenario_list})

        name_list = ['name_1', 'name_2']
        dict_values.update({self.study_name +
                            '.name_list': name_list})

        self.exec_eng.display_treeview_nodes()

        self.exec_eng.load_study_from_input_dict(
            dict_values)

        self.exec_eng.display_treeview_nodes()
        list_disciplines = list(self.exec_eng.dm.disciplines_id_map.keys())
        list_disciplines_wo_scatter = [disc for disc in list_disciplines if disc.endswith(
            'Disc1') or disc.endswith('Disc2')]

        self.assertEqual(len(toplevel_list) * len(secondlevel_list) * len(name_list)
                         * len(scenario_list), len(list_disciplines_wo_scatter))

    def test_09_scatter_with_local_input(self):
        # load process in GUI
        mydict = {'input_name': 'name_list',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.scattermap_manager.add_build_map('name_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        # get instantiator from local Process
        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id=self.sub_proc)
        scatter_list = self.exec_eng.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter_list)
        self.exec_eng.configure()

        self.assertListEqual(list(self.exec_eng.dm.get_all_namespaces_from_var_name(
            'name_list')), ['MyCase.Disc1.name_list', 'MyCase.Disc2.name_list'])
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc1.name_list', 'visibility'), 'Local')
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc2.name_list', 'visibility'), 'Local')

        dict_values = {'MyCase.Disc2.name_list': ['name_1', 'name_2'],
                       'MyCase.Disc1.name_list': ['name_1', 'name_2']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3
        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2

        private_val = {}
        private_val[self.study_name +
                    '.Disc2.name_1.constant'] = constant1
        private_val[self.study_name + '.Disc2.name_1.power'] = power1
        private_val[self.study_name +
                    '.Disc2.name_2.constant'] = constant2
        private_val[self.study_name + '.Disc2.name_2.power'] = power2
        private_val[self.study_name + '.name_1.x'] = x1
        private_val[self.study_name + '.name_2.x'] = x2
        private_val[self.study_name + '.Disc1.name_1.a'] = a1
        private_val[self.study_name + '.Disc1.name_2.a'] = a2
        private_val[self.study_name + '.Disc1.name_1.b'] = b1
        private_val[self.study_name + '.Disc1.name_2.b'] = b2
        self.exec_eng.load_study_from_input_dict(private_val)

        self.assertListEqual(list(self.exec_eng.dm.disciplines_id_map.keys()),
                             ['MyCase', 'MyCase.Disc1', 'MyCase.Disc2', 'MyCase.Disc1.name_1',
                              'MyCase.Disc1.name_2', 'MyCase.Disc2.name_1', 'MyCase.Disc2.name_2']
                             )
        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.study_name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.study_name + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)
