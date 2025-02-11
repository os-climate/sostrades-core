'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2024/06/10 Copyright 2023 Capgemini

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
import unittest
from pathlib import Path
from tempfile import gettempdir

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.folder_operations import rmtree_safe


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
            if Path(dir_to_del).is_dir():
                rmtree_safe(dir_to_del)

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

    def _test_02_mda_io_with_namespaced_tuples(self):
        """
        Test the proper construction of coupling IO in the presence of weak couplings that come from namespaces which
        have different name but same value.
        """
        exec_eng = ExecutionEngine(self.name)
        # the two disciplines built below are weakly coupled through variable y declared in different-name namespaces
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        disc1_builder = exec_eng.factory.get_builder_from_module("Disc1", disc_dir + 'disc1.Disc1')
        disc2_builder = exec_eng.factory.get_builder_from_module("Disc2", disc_dir + 'disc2b.Disc2')
        # set two namespaces to same value as to make the weak coupling effective
        ns_dict = {'ns_ac': exec_eng.study_name, "ns_other": exec_eng.study_name}
        exec_eng.ns_manager.add_ns_def(ns_dict)
        exec_eng.factory.set_builders_to_coupling_builder([disc1_builder, disc2_builder])
        private_values = {
            self.name + '.x': 10.,
            self.name + '.Disc1.a': 5.,
            self.name + '.Disc1.b': 25431.,
            self.name + '.y': 4.,
            self.name + '.Disc2.constant': 3.1416,
            self.name + '.Disc2.power': 2}
        exec_eng.configure()
        exec_eng.load_study_from_input_dict(private_values)
        exec_eng.prepare_execution()

        # the weak coupling should be an output and not an input of the process on both GEMSEO and Proxy sides
        self.assertIn(f"{self.name}.y", exec_eng.root_process.discipline_wrapp.discipline.get_output_data_names())
        self.assertNotIn(f"{self.name}.y", exec_eng.root_process.discipline_wrapp.discipline.get_input_data_names())
        self.assertIn(f"{self.name}.y", exec_eng.root_process.get_output_data_names())
        self.assertNotIn(f"{self.name}.y", exec_eng.root_process.get_input_data_names())


if '__main__' == __name__:
    cls = TestSameVarnameHandling()
    cls.setUp()
    cls.test_01_same_var_name_inst_desc_in()
