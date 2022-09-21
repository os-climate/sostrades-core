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

from sos_trades_core.sos_wrapping.test_discs.disc1 import Disc1
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
import pandas as pd


class TestSoSDiscipline(unittest.TestCase):
    """
    SoSDiscipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.ns_test = 'Test'
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'
        self.mod8_path = f'{base_path}.disc8.Disc8'
        self.mod11_path = f'{base_path}.disc11.Disc11'

    def test_01_instantiate_sosdiscipline(self):
        '''
        default initialisation test
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        sosdisc_instance = Disc1(self.name, self.ee)
        self.assertIsInstance(sosdisc_instance, SoSDiscipline,
                              "'{}' is not a SoSDiscipline".format(sosdisc_instance))

    def test_02_check_io_data(self):
        '''
        check selection of coupling variables
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        sosdisc_instance = Disc1(self.name, self.ee)
        data_names_in = sosdisc_instance.get_input_data_names()
        data_names_out = sosdisc_instance.get_output_data_names()
        self.assertEqual(data_names_in, data_names_in)
        self.assertEqual(data_names_out, data_names_out)

    def test_03_load_input_data_values(self):
        '''
        check loading of private data
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()
        x_var = self.ns_test + '.x'
        priv_in_values = {x_var: 99.}
        priv_in_values_data_in = {'x': 99.}
        self.ee.load_study_from_input_dict(priv_in_values)
        self.ee.root_process.update_from_dm()
        data_in = self.ee.root_process.sos_disciplines[0].get_data_in()
        for key, val in priv_in_values_data_in.items():
            self.assertTrue(SoSDiscipline.VALUE in data_in[key])
            self.assertEqual(val, data_in[key][SoSDiscipline.VALUE])

    def test_04_execution_success(self):
        '''
        check discipline execution
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict.keys():
            self.assertEqual(self.ee.dm.get_discipline(
                disc_id).status, 'DONE')

        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict.keys():
            self.assertEqual(self.ee.dm.get_discipline(
                disc_id).status, 'DONE')

        # get post-processing of disc1
        disc1 = self.ee.dm.get_disciplines_with_name('Test.Disc1')[0]
        filter = disc1.get_chart_filter_list()
        graph_list = disc1.get_post_processing_list(filter)
        # graph_list[0].to_plotly().show()

        y = self.ee.dm.get_value(self.name + '.y')

        self.assertEqual(y, a * x + b)

    def test_05_execution_failure(self):
        '''
        check discipline execution failure when no coupling values
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()

        values_dict = {self.ns_test + '.Disc1.a': 1.,
                       self.ns_test + '.Disc1.b': 3.}

        self.ee.load_study_from_input_dict(values_dict)

        # exception raised by check_inputs function: Test.x is not set
        self.assertRaises(ValueError, self.ee.execute)

    def test_06_namespace_appliance(self):
        '''
        check namespace appliance
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()

        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.ns_test + '.x': x,
                       self.ns_test + '.Disc1.a': a,
                       self.ns_test + '.Disc1.b': b}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        self.assertEqual(self.ee.dm.get_value(
            self.ns_test + '.x'), values_dict[self.ns_test + '.x'])

    def test_07_get_sos_io_asdict(self):
        '''
        check discipline namespace update
        '''
        ee = ExecutionEngine('Test')

        ns_dict = {'ns_ac': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        ee.factory.set_builders_to_coupling_builder(disc1_builder)

        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc1.a'] = 10.
        values_dict[self.ns_test + '.Disc1.b'] = 20.
        values_dict[self.ns_test + '.x'] = 10.

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc1 = ee.root_process.sos_disciplines[0]
        inp_dict = disc1.get_sosdisc_inputs(
            ['a', 'b'], in_dict=True)
        ref_inp = {'a': 10.0, 'b': 20.0}
        self.assertDictEqual(ref_inp, inp_dict, 'error in input dict')
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc1.get_sosdisc_outputs(
            ['indicator', 'y'], in_dict=True)
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_08_get_sos_io_no_inputs(self):
        '''
        check discipline namespace update
        '''
        ee = ExecutionEngine('Test')
        ns_dict = {'ns_ac': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        ee.factory.set_builders_to_coupling_builder(disc1_builder)

        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc1.a'] = 10.
        values_dict[self.ns_test + '.Disc1.b'] = 20.
        values_dict[self.ns_test + '.x'] = 10.

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc1 = ee.root_process.sos_disciplines[0]
        inp_dict = disc1.get_sosdisc_inputs()
        ref_inp = {
            'x': 10.0,
            'a': 10.0,
            'b': 20.0}

        for key in ref_inp:
            self.assertEqual(ref_inp[key], inp_dict[key],
                             'error in input dict')
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc1.get_sosdisc_outputs()
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_09_check_factory_with_1_added_disc(self):
        '''
        check if the root of the factory is the discipline if only 1 disc is added
        and a coupling if two are added
        '''
        ns_dict = {'ns_ac': self.ns_test}
        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.configure()
        priv_in_values = {self.ns_test + '.x': 99.,
                          self.ns_test + '.Disc1.a': 1.,
                          self.ns_test + '.Disc1.b': 3.}

        self.ee.load_study_from_input_dict(priv_in_values)

        self.ee.execute()

        self.assertIsInstance(self.ee.root_process, SoSDiscipline,
                              'The root of the factory must be a SoSDiscipline because only one disc has been added')

        # Now we try with two disciplines
        ee2 = ExecutionEngine('Test2')

        ns_dict = {'ns_ac': 'Test2'}
        ee2.ns_manager.add_ns_def(ns_dict)

        disc1_builder = ee2.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc2_builder = ee2.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        ee2.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])
        ee2.configure()

        priv_in_values = {'Test2.x': 99.,
                          'Test2.Disc1.a': 1.,
                          'Test2.Disc1.b': 3.,
                          'Test2.y': 10.,
                          'Test2.Disc2.constant': 4.,
                          'Test2.Disc2.power': 2}
        ee2.load_study_from_input_dict(priv_in_values)

        ee2.execute()

        self.assertIsInstance(ee2.root_process, SoSCoupling,
                              'The root of the factory must be a SoSDiscipline because only one disc has been added')

    def test_10_check_overwrite_of_default_values(self):
        '''
        check defaults for public
        '''
        ee = ExecutionEngine('Test')
        ns_dict = {'ns_protected': self.ns_test}
        ee.ns_manager.add_ns_def(ns_dict)

        disc8_builder = ee.factory.get_builder_from_module(
            'Disc8', self.mod8_path)
        ee.factory.set_builders_to_coupling_builder(disc8_builder)

        ee.configure()

        values_dict = {}
        values_dict[self.ns_test + '.Disc8.a'] = 10.
        # default value for 'b' is 2
        values_dict[self.ns_test + '.Disc8.b'] = 20.
        values_dict[self.ns_test + '.x'] = 10.

        ee.load_study_from_input_dict(values_dict)

        # get inputs and compare to reference
        disc8 = ee.root_process.sos_disciplines[0]

        inp_dict = disc8.get_sosdisc_inputs(
            in_dict=True)
        ref_inp = {
            'x': 10.0,
            'a': 10.0,
            'b': 20.0}

        for key in ref_inp:
            self.assertEqual(ref_inp[key], inp_dict[key],
                             'error in input dict')
        ee.execute()

        # get outputs and compare to reference
        out_dict = disc8.get_sosdisc_outputs()
        ref_out = {'indicator': 200.0, 'y': 120.0}
        self.assertDictEqual(ref_out, out_dict, 'error in input dict')

    def test_11_check_simpy_formula(self):
        '''
        check simpy formula usage
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        x = 3.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': '3*Test.x',
                       self.name + '.Disc1.b': '2*Test.Disc1.a'}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict.keys():
            self.assertEqual(self.ee.dm.get_discipline(
                disc_id).status, 'DONE')

        self.ee.execute()

        # check status DONE after execution
        for disc_id in self.ee.dm.disciplines_dict.keys():
            self.assertEqual(self.ee.dm.get_discipline(
                disc_id).status, 'DONE')

        # get post-processing of disc1
        disc1 = self.ee.dm.get_disciplines_with_name('Test.Disc1')[0]
        filter = disc1.get_chart_filter_list()
        graph_list = disc1.get_post_processing_list(filter)
        # graph_list[0].to_plotly().show()

        y = self.ee.dm.get_value(self.name + '.y')

        self.assertEqual(y, 45)

    def test_12_check_simpy_formula_with_df(self):
        '''
        check simpy formula usage
        '''
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc11_builder = self.ee.factory.get_builder_from_module(
            'Disc11', self.mod11_path)
        self.ee.factory.set_builders_to_coupling_builder(disc11_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        x = 3.0
        test_df = pd.DataFrame()
        test_df['a'] = ['3*Test.x']
        test_df['b'] = ['2*Test.Disc11.test_df.a']
        c_dict = {}
        c_dict['c'] = 'Test.Disc11.test_df.a + Test.Disc11.test_df.b'
        test_string = '3+1'
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc11.test_df': test_df,
                       self.name + '.Disc11.c_dict': c_dict,
                       self.name + '.Disc11.test_string': test_string, }

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()
        y = self.ee.dm.get_value(self.name + '.y')
        out_string = self.ee.dm.get_value(self.name + '.Disc11.out_string')

        self.assertEqual(y, 72)
        self.assertEqual(test_string, '3+1')
