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
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder
from copy import deepcopy
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path

from numpy import array

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sos_trades_core.sos_processes.test.test_scatter_disc1_disc3_from_proc.usecase1 import Study


class TestNamespaceHandling(unittest.TestCase):
    """
    Namespace handling test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.name = 'EE'
        self.root_dir = gettempdir()

    def tearDown(self):
        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_simple_samediscipline_process(self):

        exec_eng = ExecutionEngine(self.name)

        ns_7 = exec_eng.ns_manager.add_ns('ns_protected', f'{self.name}.Disc7')
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)
        disc7_builder.associate_namespaces(ns_7)

        ns_72 = exec_eng.ns_manager.add_ns(
            'ns_protected', f'{self.name}.Disc72')
        disc7_builder2 = exec_eng.factory.get_builder_from_module(
            'Disc72', mod_list)
        disc7_builder2.associate_namespaces(ns_72)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc7_builder, disc7_builder2])
        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.Disc7.h'] = array([8., 9.])
        values_dict['EE.Disc72.h'] = array([82., 92.])
        exec_eng.load_study_from_input_dict(values_dict)

        self.assertListEqual(exec_eng.root_process.sos_disciplines[0].associated_namespaces, [
                             'ns_protected__EE.Disc7'])
        self.assertListEqual(exec_eng.root_process.sos_disciplines[1].associated_namespaces, [
                             'ns_protected__EE.Disc72'])

        self.assertTrue(exec_eng.dm.get_data('EE.Disc7.h', 'ns_reference'),
                        exec_eng.ns_manager.all_ns_dict[exec_eng.root_process.sos_disciplines[0].associated_namespaces[0]])
        exec_eng.execute()

    def test_02_same_sub_process_inside_a_process(self):

        exec_eng = ExecutionEngine(self.name)

        disc_dir = 'sos_trades_core.sos_wrapping.test_discs.'
        mods_dict = {'proc1.Disc2': disc_dir + 'disc2.Disc2',
                     'proc1.Disc1': disc_dir + 'disc1.Disc1'}
        proc_builder = BaseProcessBuilder(exec_eng)
        builder_list = proc_builder.create_builder_list(
            mods_dict, ns_dict={'ns_ac': f'{exec_eng.study_name}.proc1'}, associate_namespace=True)

        mods_dict = {'proc2.Disc2': disc_dir + 'disc2.Disc2',
                     'proc2.Disc1': disc_dir + 'disc1.Disc1'}
        builder_list2 = proc_builder.create_builder_list(
            mods_dict, ns_dict={'ns_ac': f'{exec_eng.study_name}.proc2'}, associate_namespace=True)

        builder_list.extend(builder_list2)
        exec_eng.factory.set_builders_to_coupling_builder(builder_list
                                                          )
        exec_eng.configure()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3
        private_val = {}
        private_val[self.name +
                    '.proc1.Disc2.constant'] = constant1
        private_val[self.name + '.proc1.Disc2.power'] = power1
        private_val[self.name +
                    '.proc2.Disc2.constant'] = constant2
        private_val[self.name + '.proc2.Disc2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.name + '.proc1.x'] = x1
        private_val[self.name + '.proc2.x'] = x2
        private_val[self.name + '.proc1.Disc1.a'] = a1
        private_val[self.name + '.proc2.Disc1.a'] = a2
        private_val[self.name + '.proc1.Disc1.b'] = b1
        private_val[self.name + '.proc2.Disc1.b'] = b2
        exec_eng.load_study_from_input_dict(private_val)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

    def _test_03_same_sub_process_inside_a_process(self):

        exec_eng = ExecutionEngine(self.name)

        builder_list1 = exec_eng.factory.get_builder_from_process(
            'sos_trades_core.sos_processes.test', 'test_scatter_disc1_disc3_from_proc')
        builder_list2 = deepcopy(builder_list1)
        ns_list_standard = deepcopy(exec_eng.ns_manager.ns_list)
        ns_scatter1 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
            'Scatter1', after_name=exec_eng.study_name)
        for builder in builder_list1:
            builder.set_disc_name(f'Scatter1.{builder.sos_name}')
            builder.associate_namespaces(ns_scatter1)
        ns_scatter2 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
            'Scatter2', after_name=exec_eng.study_name, namespace_list=ns_list_standard)
        for builder in builder_list2:
            builder.set_disc_name(f'Scatter2.{builder.sos_name}')
            builder.associate_namespaces(ns_scatter2)
        builder_list1.extend(builder_list2)
        exec_eng.factory.set_builders_to_coupling_builder(builder_list1)
        exec_eng.configure()

        study = Study(execution_engine=exec_eng)
        study.study_name = self.name
        private_val_list = study.setup_usecase()
        private_val = {}
        for dic in private_val_list:
            private_val.update(dic)
        exec_eng.load_study_from_input_dict(private_val)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()


if '__main__' == __name__:
    cls = TestNamespaceHandling()
    cls.setUp()
    cls._test_03_same_sub_process_inside_a_process()
