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
import sys
from copy import deepcopy

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


def print_test_name():
    # prints the name of current method
    print('\n' + "-" * 10)
    print(sys._getframe(1).f_code.co_name)
    print("-" * 10 + '\n')


class TestCache(unittest.TestCase):
    """
    SoSDiscipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine(self.name)
        self.repo = 'sos_trades_core.sos_processes.test'
        self.ns_test = 'Test'
        self.factory = self.ee.factory
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'

        self.disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.desc_in = deepcopy(self.disc1_builder.cls.DESC_IN)

        self.disc2_builder = self.ee.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        self.desc_in2 = deepcopy(self.disc2_builder.cls.DESC_IN)

        # avoid truncature in error messages for assertDictEqual
        self.maxDiff = None

    def tearDown(self):
        # TODO : make a discipline instead of manipulating DESC_INs
        self.disc1_builder.cls.DESC_IN = self.desc_in

        self.disc2_builder.cls.DESC_IN = self.desc_in2

    def test_1_test_cache_discipline_without_input_change(self):
        '''
        check discipline namespace update
        '''
        self.ee.select_root_process(self.repo, 'test_disc1')

        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # first execute
        res_1 = self.ee.execute()
        # get number of calls after first call
        n_call_1 = self.ee.root_process.n_calls

        # second execute without change of parameters
        res_2 = self.ee.execute()

        # get number of calls after second call
        n_call_2 = self.ee.root_process.n_calls

        self.assertEqual(n_call_2, n_call_1)

    def test_2_test_cache_discipline_with_input_change(self):
        '''
        check discipline namespace update
        '''
        print_test_name()
        self.ee.select_root_process(self.repo, 'test_disc1')
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # first execute
        self.ee.execute()
        # get number of calls after first call
        n_call_1 = self.ee.root_process.n_calls

        # second execute with change of a private parameter
        values_dict[f'{self.name}.Disc1.a'] = 1.
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # get number of calls after second call
        n_call_2 = self.ee.root_process.n_calls
        self.assertEqual(n_call_2, n_call_1 + 1)

        # third execute with change of a protected parameter
        values_dict[f'{self.name}.x'] = 1.
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # get number of calls after third call
        n_call_2 = self.ee.root_process.n_calls
        self.assertEqual(n_call_2, n_call_1 + 2)

    def test_3_test_cache_coupling_without_input_change(self):
        '''
        check discipline namespace update
        '''
        print_test_name()
        self.ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.

        # first execute
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()

        # second execute
        self.ee.execute()
        # check nb of calls of sos_coupling
        self.assertEqual(self.ee.root_process.n_calls, 1)
        # check nb of calls of subdisciplines
        for disc in self.ee.root_process.disciplines:
            self.assertEqual(disc.n_calls, 1)

    def test_4_test_cache_coupling_with_input_change(self):
        '''
        check discipline namespace update
        '''
        print_test_name()
        self.ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')

        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execute
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.a'] = 1.
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on second discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc2.power'] = 1.
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 0
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # third execute with modif on a protected variable
        values_dict[f'{self.name}.x'] = 2.
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

    def test_5_cache_coupling_wo_change_of_strings(self):
        ''' test with input type not converted by SoSTrades
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'string_list'}  # add new string variable
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        self.ee.display_treeview_nodes()

        for key in self.ee.dm.data_id_map:
            print("   key", key)
        a = 1.0
        b = 3.0
        x = 99.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b,
                       self.name + '.Disc1.an_input_1': 'value_1',
                       self.name + '.Disc1.an_input_2': ['value_2', 'value_3'],
                       self.name + '.Disc2.constant': 1.5,
                       self.name + '.Disc2.power': 2}

        # set input data
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execution
        self.ee.execute()
        # ref update
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1

        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execution w/o change
        self.ee.execute()
        # ref update
        n_calls_sosc += 0
        n_calls_disc1 += 0
        n_calls_disc2 += 0
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

    def test_6_test_cache_coupling_with_string_change(self):
        '''
        check discipline namespace update with string change
        and check metadata known values
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'string_list'}  # add new string variable
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        disc2_builder.cls.DESC_IN['an_input_3'] = {
            'type': 'dict'}  # add new string variable
        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['value_2', 'value_3']
        values_dict[f'{self.name}.Disc2.an_input_3'] = {
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=', 'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execute
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_new'
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # third execute with modif on privates on second discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc2.an_input_3'] = {
            'value1': 'new', 'value2': '+++'}
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 0
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # fourth execute with second modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_new2'
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_1')[
            SoSDiscipline.TYPE_METADATA]
        metadata_ref = [
            {'known_values': {'value_1': 1, 'value_new': 2, 'value_new2': 3}}]
        self.assertListEqual(metadata, metadata_ref)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)
        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_1')[
            SoSDiscipline.TYPE_METADATA]
        metadata_ref = [
            {'known_values': {'value_1': 1, 'value_new': 2, 'value_new2': 3}}]
        self.assertListEqual(metadata, metadata_ref)

    def test_7_test_cache_coupling_with_string_of_dict_change(self):
        '''
        check discipline namespace update with string change
        and check metadata known values
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'dict'}  # add new string variable
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        values_dict[f'{self.name}.Disc1.an_input_2'] = {
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=', 'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execute
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = {
            'value1': 'valuenew1', 'value2': 'value_new2'}
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = {
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=', 'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)
        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_2')[
            SoSDiscipline.TYPE_METADATA]

        metadata_ref_known_values1 = {
            'STEPS_bzefivbzei))(((__)----+!!!:;=': 1, 'valuenew1': 2}
        metadata_ref_known_values2 = {
            'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva': 1, 'value_new2': 2}

        self.assertDictEqual(
            metadata[0]['known_values'], metadata_ref_known_values1)
        self.assertDictEqual(
            metadata[1]['known_values'], metadata_ref_known_values2)

    def test_8_test_cache_coupling_with_string_list_change(self):
        '''
        check discipline namespace update with string change 
        and check metadata known values 
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'string_list'}  # add new string variable
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC2']
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execute
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC2', 'AC_new']
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        metadata_ref_known_values1 = {'known_values': {'AC1': 1}}
        metadata_ref_known_values2 = {'known_values': {'AC2': 1}}
        metadata_ref_known_values3 = {'known_values': {'AC_new': 1}}

        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_2')[
            SoSDiscipline.TYPE_METADATA]
        self.assertDictEqual(
            metadata[0], metadata_ref_known_values1)
        self.assertDictEqual(
            metadata[1], metadata_ref_known_values2)
        self.assertDictEqual(
            metadata[2], metadata_ref_known_values3)
        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC3']
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC2']
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1

        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_2')[
            SoSDiscipline.TYPE_METADATA]

        metadata_ref_known_values1 = {'known_values': {'AC1': 1}}
        metadata_ref_known_values2 = {'known_values': {'AC2': 1, 'AC3': 2}}

        self.assertDictEqual(
            metadata[0], metadata_ref_known_values1)
        self.assertDictEqual(
            metadata[1], metadata_ref_known_values2)

        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # last execute without changes
        self.ee.execute()
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

    def test_9_test_cache_coupling_with_string_list_of_dict_change(self):
        '''
        check discipline namespace update with string change 
        and check metadata known values 
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'dict'}  # add new string variable
        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)
        self.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.ee.configure()
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        values_dict[f'{self.name}.Disc1.an_input_2'] = {'scenario1': [
            'AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 'string2']}
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.dm.set_values_from_dict(values_dict)

        # get disciplines objects
        disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
        disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
        sos_coupl = self.ee.root_process
        n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0

        # first execute
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = {'scenario1': [
            'ACnew', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 'string2']}
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1
        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = {'scenario1': [
            'AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 'string2']}
        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        #n_calls_disc2 += 1

        metadata = self.ee.dm.get_data(f'{self.name}.Disc1.an_input_2')[
            SoSDiscipline.TYPE_METADATA]
        print(metadata)
        metadata_ref_known_values1 = {0: {'known_values': {
            'AC1': 1, 'ACnew': 2}}, 1: {'known_values': {'AC2': 1}}}
        metadata_ref_known_values2 = {
            0: {'known_values': {'AC3': 1}}, 1: {'known_values': {'AC4': 1}}}
        metadata_ref_known_values3 = {0: {'known_values': {
            'string': 1}}, 2: {'known_values': {'string2': 1}}}
        self.maxDiff = None
        self.assertDictEqual(
            metadata[0]['known_values'], metadata_ref_known_values1)
        self.assertDictEqual(
            metadata[1]['known_values'], metadata_ref_known_values2)
        self.assertDictEqual(
            metadata[2]['known_values'], metadata_ref_known_values3)

        # check
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)

        # last execute without changes
        self.ee.execute()
        self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(disc1.n_calls, n_calls_disc1)
        self.assertEqual(disc2.n_calls, n_calls_disc2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
