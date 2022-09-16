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
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

import numpy as np
from numpy import array

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sos_trades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager


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


if '__main__' == __name__:
    cls = TestNamespaceHandling()
    cls.setUp()
    cls.test_01_simple_samediscipline_process()
