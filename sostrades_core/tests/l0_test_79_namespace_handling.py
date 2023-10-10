'''
Copyright 2022 Airbus SAS
Modifications on 2023/07/25-2023/11/02 Copyright 2023 Capgemini

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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from copy import deepcopy

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

import numpy as np
from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver.usecase_scatter import Study


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
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)
        disc7_builder.associate_namespaces(ns_7)

        # for associated namespaces, to add a value to the existing namespace, remove namespace cleaning
        ns_72 = exec_eng.ns_manager.add_ns(
            'ns_protected', f'{self.name}.Disc72', clean_existing=False)
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

        self.assertListEqual(exec_eng.root_process.proxy_disciplines[0].associated_namespaces, [
            'ns_protected__EE.Disc7'])
        self.assertListEqual(exec_eng.root_process.proxy_disciplines[1].associated_namespaces, [
            'ns_protected__EE.Disc72'])

        self.assertTrue(exec_eng.dm.get_data('EE.Disc7.h', 'ns_reference'),
                        exec_eng.ns_manager.all_ns_dict[
                            exec_eng.root_process.proxy_disciplines[0].associated_namespaces[0]])
        exec_eng.execute()

    def test_02_same_sub_process_inside_a_process(self):

        exec_eng = ExecutionEngine(self.name)

        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
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

    def test_03_two_same_scatter_inside_a_process(self):

        exec_eng = ExecutionEngine(self.name)
        repo_name = 'sostrades_core.sos_processes.test.tests_driver_eval.multi'
        builder_list1 = exec_eng.factory.get_builder_from_process(
            repo_name, 'test_multi_driver')
        builder_list2 = exec_eng.factory.get_builder_from_process(
            repo_name, 'test_multi_driver')
        ns_list_standard = deepcopy(exec_eng.ns_manager.ns_list)
        # ns_scatter1 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
        #     'Scatter1', after_name=exec_eng.study_name)
        for builder in builder_list1:
            builder.set_disc_name(f'Scatter1.{builder.sos_name}')
            # builder.associate_namespaces(ns_scatter1)
        # ns_scatter2 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
        #     'Scatter2', after_name=exec_eng.study_name, namespace_list=ns_list_standard)
        for builder in builder_list2:
            builder.set_disc_name(f'Scatter2.{builder.sos_name}')
            # builder.associate_namespaces(ns_scatter2)
        builder_list1.extend(builder_list2)
        exec_eng.factory.set_builders_to_coupling_builder(builder_list1)
        exec_eng.configure()

        study = Study(execution_engine=exec_eng)
        study.study_name = self.name
        private_val_list = study.setup_usecase()
        private_val = {}
        for dic in private_val_list:
            private_val.update(dic)

        private_val_scatter1 = {name.replace(
            f'{self.name}.', f'{self.name}.Scatter1.'): value for name, value in private_val.items()}
        private_val_scatter2 = {name.replace(
            f'{self.name}.', f'{self.name}.Scatter2.'): value for name, value in private_val.items()}
        private_val_scatter1.update(private_val_scatter2)
        exec_eng.load_study_from_input_dict(private_val_scatter1)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

    def _test_04_two_archibuilder_with_scatter_inside_a_process(self):
        # Archi builder is not yet migrated
        exec_eng = ExecutionEngine(self.name)

        builder_1 = exec_eng.factory.get_builder_from_process(
            'sos_trades_core.sos_processes.test', 'test_architecture')
        builder_2 = exec_eng.factory.get_builder_from_process(
            'sos_trades_core.sos_processes.test', 'test_architecture')
        ns_list_standard = deepcopy(exec_eng.ns_manager.ns_list)
        ns_archi1 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
            'Archi1', after_name=exec_eng.study_name)

        builder_1.set_disc_name(f'Archi1.{builder_1.sos_name}')
        builder_1.associate_namespaces(ns_archi1)

        ns_archi2 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
            'Archi2', after_name=exec_eng.study_name, namespace_list=ns_list_standard)

        builder_2.set_disc_name(f'Archi2.{builder_2.sos_name}')
        builder_2.associate_namespaces(ns_archi2)

        exec_eng.factory.set_builders_to_coupling_builder(
            [builder_1, builder_2])
        exec_eng.configure()

        study = Study(execution_engine=exec_eng)
        study.study_name = self.name
        private_val_list = study.setup_usecase()
        private_val = {}
        for dic in private_val_list:
            private_val.update(dic)

        private_val_archi1 = {name.replace(
            f'{self.name}.', f'{self.name}.Archi1.'): value for name, value in private_val.items()}
        private_val_archi2 = {name.replace(
            f'{self.name}.', f'{self.name}.Archi2.'): value for name, value in private_val.items()}
        private_val_archi1.update(private_val_archi2)
        exec_eng.load_study_from_input_dict(private_val_archi1)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

    def test_05_update_associated_ns_with_extra_ns(self):

        exec_eng = ExecutionEngine(self.name)

        ns_7 = exec_eng.ns_manager.add_ns('ns_protected', f'{self.name}.Disc7')
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)
        disc7_builder.associate_namespaces(ns_7)

        # for associated namespaces, to add a value to the existing namespace, remove namespace cleaning
        ns_72 = exec_eng.ns_manager.add_ns('ns_protected', f'{self.name}.Disc72', clean_existing=False)
        disc7_builder2 = exec_eng.factory.get_builder_from_module(
            'Disc72', mod_list)
        disc7_builder2.associate_namespaces(ns_72)

        builder_list = [disc7_builder, disc7_builder2]

        self.assertListEqual(disc7_builder.associated_namespaces, ['ns_protected__EE.Disc7'])
        self.assertListEqual(disc7_builder2.associated_namespaces, ['ns_protected__EE.Disc72'])
        self.assertEqual(exec_eng.ns_manager.all_ns_dict[disc7_builder.associated_namespaces[0]].value,
                         f'{self.name}.Disc7')
        # update namespaces which are associated in builder_list
        extra_name = 'extra_name'
        for builder in builder_list:
            builder.update_associated_namespaces_with_extra_name(extra_name, after_name=self.name)
        self.assertListEqual(disc7_builder.associated_namespaces, [f'ns_protected__EE.{extra_name}.Disc7'])
        self.assertListEqual(disc7_builder2.associated_namespaces, [f'ns_protected__EE.{extra_name}.Disc72'])
        self.assertEqual(exec_eng.ns_manager.all_ns_dict[disc7_builder.associated_namespaces[0]].value,
                         f'{self.name}.{extra_name}.Disc7')
        self.assertFalse('ns_protected__EE.Disc7' in exec_eng.ns_manager.all_ns_dict)
        exec_eng.factory.set_builders_to_coupling_builder(builder_list
                                                          )
        exec_eng.configure()

        # # additional test to verify that values_in are used
        # values_dict = {}
        # values_dict['EE.Disc7.h'] = array([8., 9.])
        # values_dict['EE.Disc72.h'] = array([82., 92.])
        # exec_eng.load_study_from_input_dict(values_dict)
        #
        # self.assertListEqual(exec_eng.root_process.proxy_disciplines[0].associated_namespaces, [
        #     'ns_protected__EE.Disc7'])
        # self.assertListEqual(exec_eng.root_process.proxy_disciplines[1].associated_namespaces, [
        #     'ns_protected__EE.Disc72'])
        #
        # self.assertTrue(exec_eng.dm.get_data('EE.Disc7.h', 'ns_reference'),
        #                 exec_eng.ns_manager.all_ns_dict[
        #                     exec_eng.root_process.proxy_disciplines[0].associated_namespaces[0]])
        # exec_eng.execute()


if '__main__' == __name__:
    cls = TestNamespaceHandling()
    cls.setUp()
    cls.test_03_two_same_scatter_inside_a_process()
