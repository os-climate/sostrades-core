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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from copy import deepcopy

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path

from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver.usecase_scatter import Study


class TestSameVarnameHandling(unittest.TestCase):
    """
    Same var name handling test class
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

    def test_01_same_var_name_inst_desc_in(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_x1', f'{self.name}.X1_ns')
        exec_eng.ns_manager.add_ns('ns_x2', f'{self.name}.X2_ns')
        exec_eng.ns_manager.add_ns('ns_x', f'{self.name}.Disc1')
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1_same_vars.Disc1'
        disc1_builder = exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc1_builder)
        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        x1 = 2.0
        x2 = 4.0
        true_x = 3.0
        a = 1.0
        values_dict[f'{self.name}.X1_ns.x'] = x1
        values_dict[f'{self.name}.X2_ns.x'] = x2
        values_dict[f'{self.name}.Disc1.x'] = true_x
        values_dict[f'{self.name}.Disc1.a'] = a
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        y = exec_eng.dm.get_value(f'{self.name}.Disc1.y')
        true_x_out = exec_eng.dm.get_value(f'{self.name}.Disc1.true_x')
        test_a = exec_eng.dm.get_value(f'{self.name}.Disc1.test_a')
        self.assertEqual(y, x1 + x2)
        self.assertEqual(true_x, true_x_out)
        self.assertTrue(test_a)


if '__main__' == __name__:
    cls = TestSameVarnameHandling()
    cls.setUp()
    cls.test_01_same_var_name_inst_desc_in()
