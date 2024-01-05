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

import unittest
import sys
from copy import deepcopy
from os import remove

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from gemseo.core.discipline import MDODiscipline


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
        self.repo = 'sostrades_core.sos_processes.test'
        self.ns_test = 'Test'
        self.factory = self.ee.factory
        base_path = 'sostrades_core.sos_wrapping.test_discs'
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

        try:
            remove('.\cache.h5')
        except OSError:
            pass

    def test_1_test_cache_discipline_without_input_change(self):
        '''
        Checks cache setting and number of calls.
        '''
        # WITHOUT CACHE

        self.ee.select_root_process(self.repo, 'test_disc1')

        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.x'] = 3.
        self.ee.load_study_from_input_dict(values_dict)

        # first execute
        res_1 = self.ee.execute()

        # check cache is None
        self.assertEqual(self.ee.dm.get_value('SoSDisc.cache_type'), MDODiscipline.CacheType.NONE)
        self.assertEqual(self.ee.dm.get_value(
            'SoSDisc.Disc1.cache_type'), MDODiscipline.CacheType.NONE)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.cache, None)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.mdo_chain.cache, None)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[0].cache, None)
        # get number of calls after first call
        n_call_root_1 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_1 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[
            0].n_calls

        # second execute without change of parameters
        res_2 = self.ee.execute()

        # get number of calls after second call
        n_call_root_2 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_2 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[
            0].n_calls

        self.assertEqual(n_call_root_2, n_call_root_1 + 1)
        self.assertEqual(n_call_2, n_call_1 + 1)

        # ACTIVATE SIMPLE CACHE ROOT PROCESS

        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'
        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

        # first execute
        res_1 = self.ee.execute()

        self.assertEqual(self.ee.dm.get_value(
            'SoSDisc.Disc1.cache_type'), 'SimpleCache')
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.cache.__class__.__name__, 'SimpleCache')
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.mdo_chain.cache.__class__.__name__, 'SimpleCache')
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[0].cache.__class__.__name__,
            'SimpleCache')
        # get number of calls after first call
        n_call_root_1 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_1 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[
            0].n_calls

        # second execute without change of parameters
        res_2 = self.ee.execute()

        # get number of calls after second call
        n_call_root_2 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_2 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls

        self.assertEqual(n_call_root_2, n_call_root_1)
        self.assertEqual(n_call_2, n_call_1)

        # DESACTIVATE CACHE

        values_dict[f'{self.name}.cache_type'] = MDODiscipline.CacheType.NONE
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.prepare_execution()

        # check cache is None
        self.assertEqual(self.ee.dm.get_value('SoSDisc.cache_type'), MDODiscipline.CacheType.NONE)
        self.assertEqual(self.ee.dm.get_value(
            'SoSDisc.Disc1.cache_type'), MDODiscipline.CacheType.NONE)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.cache, None)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.mdo_chain.cache, None)
        self.assertEqual(
            self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.cache, None)

        # ACTIVATE CACHE FOR DISC1 ONLY

        values_dict[f'{self.name}.Disc1.cache_type'] = 'SimpleCache'
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.prepare_execution()

        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.cache, None)
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.mdo_chain.cache, None)
        self.assertEqual(
            self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.cache.__class__.__name__,
            'SimpleCache')

        # first execute
        res_1 = self.ee.execute()
        # get number of calls after first call
        n_call_root_1 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_1 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls

        # second execute without change of parameters
        res_2 = self.ee.execute()

        # get number of calls after second call
        n_call_root_2 = self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls
        n_call_2 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls

        # 2 calls for root process, 1 call for Disc1
        self.assertEqual(n_call_root_2, n_call_root_1 + 1)
        self.assertEqual(n_call_2, n_call_1)

    def test_2_test_cache_discipline_with_input_change(self):
        '''
        Checks number of calls is coherent.
        '''
        print_test_name()
        self.ee.select_root_process(self.repo, 'test_disc1')
        # set input data
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.x'] = 3.
        values_dict[f'{self.name}.Disc1.cache_type'] = 'SimpleCache'
        self.ee.load_study_from_input_dict(values_dict)

        # first execute
        self.ee.execute()
        # get number of calls after first call
        n_call_1 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls

        # second execute with change of a private parameter
        values_dict[f'{self.name}.Disc1.a'] = 1.
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # get number of calls after second call
        n_call_2 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls
        self.assertEqual(n_call_2, n_call_1 + 1)

        # third execute with change of a protected parameter
        values_dict[f'{self.name}.x'] = 1.
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # get number of calls after third call
        n_call_2 = self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.n_calls
        self.assertEqual(n_call_2, n_call_1 + 2)

    def test_3_test_cache_coupling_without_input_change(self):
        '''
        Checks number of calls is coherent.
        '''
        print_test_name()
        self.ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        values_dict = {}
        values_dict[f'{self.name}.Disc1.a'] = 10.
        values_dict[f'{self.name}.Disc1.b'] = 20.
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        values_dict[f'{self.name}.Disc1.cache_type'] = 'SimpleCache'
        values_dict[f'{self.name}.Disc2.cache_type'] = 'SimpleCache'
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'

        # first execute
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # second execute
        self.ee.execute()
        # check nb of calls of sos_coupling
        self.assertEqual(
            self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls, 1)
        # check nb of calls of subdisciplines
        for disc in self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines:
            self.assertEqual(disc.n_calls, 1)

    def test_4_test_cache_coupling_with_input_change(self):
        '''
        Checks number of calls is coherent.
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
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'
        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.a'] = 1.
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on second discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc2.power'] = 1
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 0
        n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # third execute with modif on a protected variable
        values_dict[f'{self.name}.x'] = 2.
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    def test_5_cache_coupling_wo_change_of_strings(self):
        '''
        Checks number of calls is coherent.
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}}  # add new string variable
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
                       self.name + '.Disc2.power': 2,
                       self.name + '.cache_type': 'SimpleCache',
                       f'{self.name}.propagate_cache_to_children': True}

        # set input data
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execution w/o change
        self.ee.execute()
        # ref update
        n_calls_sosc += 0
        n_calls_disc1 += 0
        n_calls_disc2 += 0
        # check
        # self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    def test_6_test_cache_coupling_with_string_change(self):
        '''
        Checks number of calls is coherent.
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}, }  # add new string variable
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
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
            'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'

        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_new'
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # third execute with modif on privates on second discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc2.an_input_3'] = {
            'value1': 'new', 'value2': '+++'}
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 0
        n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # fourth execute with second modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_new2'
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_1'] = 'value_1'
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    def test_7_test_cache_coupling_with_string_of_dict_change(self):
        '''
        Checks number of calls is coherent.
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
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
            'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        values_dict[f'{self.name}.Disc2.power'] = 2
        values_dict[f'{self.name}.Disc2.constant'] = -10.
        values_dict[f'{self.name}.x'] = 3.
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'

        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = {
            'value1': 'valuenew1', 'value2': 'value_new2'}
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = {
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
            'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    def test_8_test_cache_coupling_with_string_list_change(self):
        '''
        '''
        ns_dict = {'ns_ac': self.name}

        self.ee.ns_manager.add_ns_def(ns_dict)

        disc1_builder = self.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        disc1_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc1_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}, }  # add new string variable
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
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'

        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC2', 'AC_new']
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC3']
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        values_dict[f'{self.name}.Disc1.an_input_2'] = ['AC1', 'AC2']
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1

        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # last execute without changes
        self.ee.execute()
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    def test_9_test_cache_coupling_with_string_list_of_dict_change(self):
        '''
        Checks number of calls is coherent.
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
        values_dict[f'{self.name}.cache_type'] = 'SimpleCache'

        values_dict[f'{self.name}.propagate_cache_to_children'] = True
        self.ee.load_study_from_input_dict(values_dict)

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
        self.assertEqual(
            sos_coupl.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # second execute with modif on privates on first discipline
        # so that all disciplines are executed twice
        values_dict[f'{self.name}.Disc1.an_input_2'] = {'scenario1': [
            'ACnew', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 'string2']}
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1
        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # fifth execute with second modif on privates on first discipline
        # but the same as the first execute : all disciplines must be
        # reexecuted
        values_dict[f'{self.name}.Disc1.an_input_2'] = {'scenario1': [
            'AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 'string2']}
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()
        # ref
        n_calls_sosc += 1
        n_calls_disc1 += 1
        # n_calls_disc2 += 1

        # check
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

        # last execute without changes
        self.ee.execute()
        #         self.assertEqual(sos_coupl.n_calls, n_calls_sosc)
        self.assertEqual(
            disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
        self.assertEqual(
            disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)

    # def _test_10_cache_on_sellar_optim_gemseo_scenario(self):
    #     '''
    #     Test commented because it builds the process with MDODiscipline objects without using Execution Engine,
    #     so that can't call SoSDiscipline method (_convert_new_type_into_array in MDA)
    #     '''
    #
    #     disciplines = [Sellar1(residual_form=False),
    #                    Sellar2(residual_form=False),
    #                    SellarSystem()]
    #
    #     design_space = SellarDesignSpace()
    #
    #     scenario = MDOScenario(disciplines,
    #                            formulation="MDF",  # "DisciplinaryOpt",
    #                            objective_name='obj',
    #                            design_space=design_space,
    #                            tolerance=1e-8,
    #                            inner_mda_name='MDAGaussSeidel')  # 'MDAJacobi'
    #     scenario.set_differentiation_method("user")  # user
    #
    #     # add constraints
    #     scenario.add_constraint("c_1", "ineq")
    #     scenario.add_constraint("c_2", "ineq")
    #
    #     run_inputs = {'max_iter': 10, 'algo': "SLSQP"}
    #
    #     dtype = "float64"
    #     mda_data = {'x_local': np.array([1.], dtype=dtype),
    #                 'x_shared': np.array([4., 3.], dtype=dtype),
    #                 'y_0': np.array([1.], dtype=dtype),
    #                 'y_1': np.array([1.], dtype=dtype)
    #                 }
    #
    #     # Run MDA
    #     print("\n ***** Run MDA\n")
    #     scenario.formulation.mda.execute(mda_data)
    #
    #     for disc in scenario.formulation.disciplines:
    #         print("\t " + str(disc.name))
    #         print("\t | n_calls: " + str(disc.n_calls) +
    #               ", n_calls_linearize: " + str(disc.n_calls))
    #         for k, v in disc.local_data.items():
    #             print("\t | " + str(k) + " " + str(v))
    #     print("\n \t in MDA")
    #     for k, v in scenario.formulation.mda.local_data.items():
    #         print("\t | " + str(k) + " " + str(v))
    #
    #     # run optimization
    #     print("\n ***** Run Optim\n")
    #     scenario.execute(run_inputs)
    #
    #     for disc in scenario.formulation.disciplines:
    #         print("\t " + str(disc.name))
    #         print("\t | n_calls: " + str(disc.n_calls) +
    #               ", n_calls_linearize: " + str(disc.n_calls))
    #         for k, v in disc.local_data.items():
    #             print("\t | " + str(k) + " " + str(v))
    #     print("\n \t in MDA")
    #     for k, v in scenario.formulation.mda.local_data.items():
    #         print("\t | " + str(k) + " " + str(v))
    #
    #     # run evaluate_function
    #     print("\n ***** Evaluate functions\n")
    #     scenario.formulation.opt_problem.evaluate_functions()
    #
    #     for disc in scenario.formulation.disciplines:
    #         print("\n \t " + str(disc.name))
    #         print("\t | n_calls: " + str(disc.n_calls) +
    #               ", n_calls_linearize: " + str(disc.n_calls))
    #         for k, v in disc.local_data.items():
    #             print("\t | " + str(k) + " " + str(v))
    #     print("\n \t in MDA")
    #     for k, v in scenario.formulation.mda.local_data.items():
    #         print("\t | " + str(k) + " " + str(v))

    # def _test_11_recursive_cache_activation(self):
    #
    #     self.study_name = 'optim'
    #     self.ns = f'{self.study_name}'
    #     self.sc_name = "SellarOptimScenario"
    #     self.c_name = "SellarCoupling"
    #     self.repo = 'sostrades_core.sos_processes.test'
    #     self.proc_name = 'test_sellar_opt_discopt'
    #
    #     # build sellar optim process
    #     self.ee = ExecutionEngine(self.study_name)
    #     factory = self.ee.factory
    #
    #     builder = factory.get_builder_from_process(repo=self.repo,
    #                                                mod_id=self.proc_name)
    #
    #     self.ee.factory.set_builders_to_coupling_builder(builder)
    #     self.ee.configure()
    #
    #     dspace_dict = {'variable': ['x'],
    #                    'value': [[1.]],
    #                    'lower_bnd': [[0.]],
    #                    'upper_bnd': [[10.]],
    #                    'enable_variable': [True],
    #                    'activated_elem': [[True]]}
    #     dspace = pd.DataFrame(dspace_dict)
    #
    #     disc_dict = {}
    #     disc_dict[f'{self.ns}.SellarOptimScenario.{self.c_name}.inner_mda_name'] = 'MDAGaussSeidel'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 2
    #     disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
    #     disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
    #     disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
    #         'c_1', 'c_2']
    #     disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
    #                                                                 "ineq_tolerance": 1e-6,
    #                                                                 "normalize_design_space": True}
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = np.array([
    #         1., 1.])
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = 10.
    #
    #     # execute sellar optim without cache and retrieve dm
    #     self.ee.load_study_from_input_dict(disc_dict)
    #
    #     def check_cache_name(disc, cache_name):
    #         self.assertEqual(disc.cache.__class__.__name__, cache_name)
    #         for sub_disc in disc.sos_disciplines:
    #             check_cache_name(sub_disc, cache_name)
    #         if isinstance(disc, ProxyCoupling):
    #             check_cache_name(disc.mdo_chain, cache_name)
    #             for sub_mda in disc.sub_mda_list:
    #                 check_cache_name(sub_mda, cache_name)
    #
    #     check_cache_name(self.ee.root_process, 'NoneType')
    #
    #     # activate root_process cache and check recursive activation
    #     disc_dict[f'{self.ns}.cache_type'] = 'SimpleCache'
    #     self.ee.load_study_from_input_dict(disc_dict)
    #
    #     check_cache_name(self.ee.root_process, 'SimpleCache')
    #
    # def _test_12_cache_on_sellar_optim(self):
    #
    #     self.study_name = 'optim'
    #     self.ns = f'{self.study_name}'
    #     self.sc_name = "SellarOptimScenario"
    #     self.c_name = "SellarCoupling"
    #     self.repo = 'sostrades_core.sos_processes.test'
    #     self.proc_name = 'test_sellar_opt_discopt'
    #
    #     # build sellar optim process
    #     exec_eng = ExecutionEngine(self.study_name)
    #     factory = exec_eng.factory
    #
    #     builder = factory.get_builder_from_process(repo=self.repo,
    #                                                mod_id=self.proc_name)
    #
    #     exec_eng.factory.set_builders_to_coupling_builder(builder)
    #     exec_eng.configure()
    #
    #     dspace_dict = {'variable': ['x'],
    #                    'value': [[1.]],
    #                    'lower_bnd': [[0.]],
    #                    'upper_bnd': [[10.]],
    #                    'enable_variable': [True],
    #                    'activated_elem': [[True]]}
    #     dspace = pd.DataFrame(dspace_dict)
    #
    #     disc_dict = {}
    #     disc_dict[f'{self.ns}.SellarOptimScenario.{self.c_name}.inner_mda_name'] = 'MDAGaussSeidel'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 2
    #     disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
    #     disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
    #     disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
    #     disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
    #         'c_1', 'c_2']
    #     disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
    #                                                                 "ineq_tolerance": 1e-6,
    #                                                                 "normalize_design_space": True}
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = np.array([1.])
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = np.array([
    #                                                                         1.])
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = np.array([
    #                                                                         1.])
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = np.array([
    #         1., 1.])
    #     disc_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = 10.
    #
    #     # execute sellar optim without cache and retrieve dm
    #     exec_eng.load_study_from_input_dict(disc_dict)
    #     exec_eng.execute()
    #     dm_without_cache = exec_eng.dm.get_data_dict_values()
    #
    #     # execute sellar optim with HDF5Cache and retrieve dm
    #     #         for cache_type_key in exec_eng.dm.get_all_namespaces_from_var_name('cache_type'):
    #     #             disc_dict[cache_type_key] = 'HDF5Cache'
    #     #         for cache_file_paht_key in exec_eng.dm.get_all_namespaces_from_var_name('cache_file_path'):
    #     #             disc_dict[cache_file_paht_key] = 'cache.h5'
    #     #         exec_eng.load_study_from_input_dict(disc_dict)
    #     #         exec_eng.execute()
    #     #         dm_with_HDF5_cache = exec_eng.dm.get_data_dict_values()
    #
    #     # execute sellar optim with MemoryFullCache and retrieve dm
    #     #         for cache_type_key in exec_eng.dm.get_all_namespaces_from_var_name('cache_type'):
    #     #             disc_dict[cache_type_key] = 'MemoryFullCache'
    #     #         exec_eng.load_study_from_input_dict(disc_dict)
    #     #         exec_eng.execute()
    #     #         dm_with_memory_full_cache = exec_eng.dm.get_data_dict_values()
    #
    #     # execute sellar optim with SimpleCache and retrieve dm
    #     for cache_type_key in exec_eng.dm.get_all_namespaces_from_var_name('cache_type'):
    #         disc_dict[cache_type_key] = 'SimpleCache'
    #     exec_eng.load_study_from_input_dict(disc_dict)
    #     exec_eng.execute()
    #     dm_with_simple_cache = exec_eng.dm.get_data_dict_values()
    #
    #     # remove cache_type keys from dm_with_cache and dm_without_cache
    #     for cache_type_key in exec_eng.dm.get_all_namespaces_from_var_name(
    #         'cache_type') + exec_eng.dm.get_all_namespaces_from_var_name(
    #             'cache_file_path') + exec_eng.dm.get_all_namespaces_from_var_name('residuals_history'):
    #         dm_with_simple_cache.pop(cache_type_key)
    #         #             dm_with_HDF5_cache.pop(cache_type_key)
    #         #             dm_with_memory_full_cache.pop(cache_type_key)
    #         dm_without_cache.pop(cache_type_key)
    #
    #     # compare values in dm_with_cache, dm_with_HDF5_cache,
    #     # dm_with_memory_full_cache and dm_without_cache
    #     dict_error = {}
    #     compare_dict(dm_with_simple_cache,
    #                  dm_without_cache, '', dict_error)
    #     #         compare_dict(dm_with_HDF5_cache,
    #     #                      dm_without_cache, '', dict_error)
    #     #         compare_dict(dm_with_memory_full_cache,
    #     #                      dm_without_cache, '', dict_error)
    #     self.assertDictEqual(dict_error, {})
    #
    # def _test_13_simple_cache_on_grid_search(self):
    #     """In this test we prove the ability of the cache to work properly on a grid search
    #     """
    #
    #     repo = 'sostrades_core.sos_processes.test'
    #     proc_name = 'test_grid_search'
    #     sa_builder = self.ee.factory.get_builder_from_process(
    #         repo, proc_name)
    #
    #     self.ee.factory.set_builders_to_coupling_builder(
    #         sa_builder)
    #     self.ee.load_study_from_input_dict({})
    #
    #     print(self.ee.display_treeview_nodes())
    #
    #     self.grid_search = 'GridSearch'
    #     self.study_name = 'SoSDisc'
    #
    #     eval_inputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.eval_inputs')
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.x', ['selected_input']] = True
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.j', ['selected_input']] = True
    #
    #     gather_outputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.gather_outputs')
    #     gather_outputs.loc[gather_outputs['full_name'] ==
    #                      f'{self.grid_search}.Disc1.y', ['selected_output']] = True
    #
    #     dspace = pd.DataFrame({
    #         'shortest_name': ['x', 'j'],
    #         'lower_bnd': [5., 20.],
    #         'upper_bnd': [7., 25.],
    #         'nb_points': [2, 2],
    #         'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
    #     })
    #
    #     dict_values = {
    #         # GRID SEARCH INPUTS
    #         f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
    #         f'{self.study_name}.{self.grid_search}.gather_outputs': gather_outputs,
    #         f'{self.study_name}.{self.grid_search}.design_space': dspace,
    #
    #         # DISC1 INPUTS
    #         f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
    #         f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
    #         f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
    #         f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.d': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.f': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.g': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.h': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.j': 3.,
    #
    #     }
    #
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     grid_search_disc = self.ee.dm.get_disciplines_with_name(
    #         f'{self.study_name}.{self.grid_search}')[0]
    #     disc1 = self.ee.dm.get_disciplines_with_name(
    #         f'{self.study_name}.{self.grid_search}.Disc1')[0]
    #
    #     # check cache is None
    #     self.assertEqual(
    #         grid_search_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(disc1.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(self.ee.root_process.cache, None)
    #     self.assertEqual(self.ee.root_process.mdo_chain.cache, None)
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #
    #     # check grid_search has run one extra time
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1 + 1)
    #
    #     # ACTIVATE SIMPLE CACHE ROOT PROCESS
    #
    #     dict_values[f'{self.name}.cache_type'] = 'SimpleCache'
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     # check that the root process distributes the cache to all its
    #     # sos_disciplines
    #     self.assertEqual(grid_search_disc.get_sosdisc_inputs(
    #         'cache_type'), 'SimpleCache')
    #     self.assertEqual(disc1.get_sosdisc_inputs('cache_type'), 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.mdo_chain.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.sos_disciplines[0].cache.__class__.__name__, 'SimpleCache')
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #
    #     # check that grid search has not run since no input was changed
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1)
    #
    #     # Third execute with a change of the design space
    #     dspace = pd.DataFrame({
    #         'shortest_name': ['x', 'j'],
    #         'lower_bnd': [5., 20.],
    #         'upper_bnd': [10., 30.],
    #         'nb_points': [3, 2],
    #         'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
    #     })
    #     dict_values[f'{self.study_name}.{self.grid_search}.design_space'] = dspace
    #     self.ee.load_study_from_input_dict(dict_values)
    #     res_3 = self.ee.execute()
    #
    #     # get number of calls after third call
    #     n_call_grid_search_3 = grid_search_disc.n_calls
    #
    #     # check that grid search has run
    #     self.assertEqual(n_call_grid_search_3, n_call_grid_search_2 + 1)
    #
    #     # Fourth execute with no change one more time
    #     res_4 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_4 = grid_search_disc.n_calls
    #
    #     # check that grid search has not run since no input was changed
    #     self.assertEqual(n_call_grid_search_4, n_call_grid_search_3)
    #
    #     # DESACTIVATE CACHE
    #
    #     dict_values[f'{self.name}.cache_type'] = 'None'
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     # check cache is None
    #     self.assertEqual(
    #         grid_search_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(disc1.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(self.ee.root_process.cache, None)
    #     self.assertEqual(self.ee.root_process.mdo_chain.cache, None)
    #
    #     #  execute one more time
    #     res_5 = self.ee.execute()
    #     # get number of calls after execute
    #     n_call_grid_search_5 = grid_search_disc.n_calls
    #
    #     # check that grid search has not run since cache is not activated
    #     self.assertEqual(n_call_grid_search_5, n_call_grid_search_4 + 1)
    #
    # def _test_14_simple_cache_on_grid_search_subprocess(self):
    #     """In this test we prove the ability of the cache to work properly on a grid search
    #     We activate simple cache, change a grid search subprocess input (here an input of Disc1)
    #     We expect the grid search to run since its subprocess has changed
    #     """
    #
    #     repo = 'sostrades_core.sos_processes.test'
    #     proc_name = 'test_grid_search'
    #     sa_builder = self.ee.factory.get_builder_from_process(
    #         repo, proc_name)
    #
    #     self.ee.factory.set_builders_to_coupling_builder(
    #         sa_builder)
    #     self.ee.load_study_from_input_dict({})
    #
    #     print(self.ee.display_treeview_nodes())
    #
    #     self.grid_search = 'GridSearch'
    #     self.study_name = 'SoSDisc'
    #
    #     eval_inputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.eval_inputs')
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.x', ['selected_input']] = True
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.j', ['selected_input']] = True
    #
    #     gather_outputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.gather_outputs')
    #     gather_outputs.loc[gather_outputs['full_name'] ==
    #                      f'{self.grid_search}.Disc1.y', ['selected_output']] = True
    #
    #     dspace = pd.DataFrame({
    #         'shortest_name': ['x', 'j'],
    #         'lower_bnd': [5., 20.],
    #         'upper_bnd': [7., 25.],
    #         'nb_points': [2, 2],
    #         'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
    #     })
    #
    #     dict_values = {
    #         # GRID SEARCH INPUTS
    #         f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
    #         f'{self.study_name}.{self.grid_search}.gather_outputs': gather_outputs,
    #         f'{self.study_name}.{self.grid_search}.design_space': dspace,
    #
    #         # DISC1 INPUTS
    #         f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
    #         f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
    #         f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
    #         f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.d': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.f': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.g': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.h': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.j': 3.,
    #
    #     }
    #
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     grid_search_disc = self.ee.dm.get_disciplines_with_name(
    #         f'{self.study_name}.{self.grid_search}')[0]
    #     disc1 = self.ee.dm.get_disciplines_with_name(
    #         f'{self.study_name}.{self.grid_search}.Disc1')[0]
    #
    #     # check cache is None
    #     self.assertEqual(
    #         grid_search_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(disc1.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(self.ee.root_process.cache, None)
    #     self.assertEqual(self.ee.root_process.mdo_chain.cache, None)
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #
    #     # check grid_search has run one extra time
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1 + 1)
    #
    #     # ACTIVATE SIMPLE CACHE ROOT PROCESS
    #
    #     dict_values[f'{self.name}.cache_type'] = 'SimpleCache'
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     # check that the root process distributes the cache to all its
    #     # sos_disciplines
    #     self.assertEqual(grid_search_disc.get_sosdisc_inputs(
    #         'cache_type'), 'SimpleCache')
    #     self.assertEqual(disc1.get_sosdisc_inputs('cache_type'), 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.mdo_chain.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.sos_disciplines[0].cache.__class__.__name__, 'SimpleCache')
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #
    #     # check that grid search has not run since no input was changed
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1)
    #
    #     # change an input of disc 1
    #     dict_values[f'{self.study_name}.{self.grid_search}.Disc1.d'] = 10.
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     # third execute without change of parameters
    #     res_3 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_3 = grid_search_disc.n_calls
    #
    #     # check that grid search has run since the subprocess has changed
    #     # self.assertEqual(n_call_grid_search_3, n_call_grid_search_2 + 1)
    #
    # def _test_15_set_recursive_cache_scatter(self):
    #
    #     ns_dict = {'ns_ac': self.name}
    #
    #     self.ee.ns_manager.add_ns_def(ns_dict)
    #
    #     mydict_build = {'input_name': 'name_list',
    #                     'input_type': 'string_list',
    #                     'input_ns': 'ns_barrierr',
    #                     'output_name': 'ac_name',
    #                     'scatter_ns': 'ns_ac'}
    #     self.ee.ns_manager.add_ns('ns_barrierr', self.name)
    #
    #     self.ee.scattermap_manager.add_build_map('name_list', mydict_build)
    #     mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
    #     builder_list = self.factory.get_builder_from_module('Disc1', mod_list)
    #
    #     scatter_builder = self.ee.factory.create_scatter_builder(
    #         'scatter', 'name_list', builder_list)
    #
    #     self.ee.factory.set_builders_to_coupling_builder(scatter_builder)
    #
    #     self.ee.configure()
    #     self.ee.display_treeview_nodes()
    #
    #     dict_values = {self.name + '.name_list': ['name_1', 'name_2'],
    #                    self.name + '.scatter.name_1.x': 1,
    #                    self.name + '.scatter.name_1.a': 2,
    #                    self.name + '.scatter.name_1.b': 3,
    #                    self.name + '.scatter.name_2.x': 2,
    #                    self.name + '.scatter.name_2.a': 0,
    #                    self.name + '.scatter.name_2.b': 5,
    #                    self.name + '.cache_type': 'SimpleCache'}
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     for cache_input in self.ee.dm.get_all_namespaces_from_var_name('cache_type'):
    #         self.assertTrue(self.ee.dm.get_value(cache_input), 'SimpleCache')
    #
    #     for disc in self.ee.factory.sos_disciplines:
    #         self.assertTrue(disc.cache.__class__.__name__, 'SimpleCache')
    #
    #     dict_values = {self.name + '.cache_type': 'None'}
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     for disc in self.ee.factory.sos_disciplines:
    #         self.assertTrue(disc.cache.__class__.__name__, None)
    #
    #     dict_values = {self.name + '.scatter.name_1.cache_type': 'SimpleCache'}
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     for cache_input in self.ee.dm.get_all_namespaces_from_var_name('cache_type'):
    #         if 'name_1' in cache_input:
    #             self.assertTrue(self.ee.dm.get_value(
    #                 cache_input), 'SimpleCache')
    #         else:
    #             self.assertTrue(self.ee.dm.get_value(cache_input), 'None')
    #
    #     dict_values = {self.name + '.cache_type': 'SimpleCache',
    #                    self.name + '.scatter.name_1.cache_type': 'None'}
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     for cache_input in self.ee.dm.get_all_namespaces_from_var_name('cache_type'):
    #         self.assertTrue(self.ee.dm.get_value(cache_input), 'SimpleCache')
    #
    # def _test_16_set_cache_recursively_on_sos_optim(self):
    #     """In this test we prove the ability of sosscenario discipline to recursively
    #     set its children cache
    #     """
    #     # create study sellar opt and load data from usecase
    #     study_1 = study_sellar_opt()
    #     study_1.load_data()
    #     # cache activation at the level of optim scenario
    #     dict_values = {
    #         f'{study_1.study_name}.SellarOptimScenario.cache_type': 'SimpleCache'}
    #     study_1.load_data(from_input_dict=dict_values)
    #
    #     # check that the cache type is set for all optim scenario sub
    #     # disciplines
    #     for discipline in study_1.execution_engine.factory.sos_disciplines:
    #         self.assertEqual(discipline.get_sosdisc_inputs(
    #             'cache_type'), 'SimpleCache')
    #
    # def test_17_cache_and_status_coupling(self):
    #     '''
    #     Checks that both SoSTrades and GEMSEO objects have status PENDING after a second prepare_execution and go DONE
    #     when the coupling loads cache.
    #     '''
    #     from gemseo.core.discipline import MDODiscipline
    #
    #     ns_dict = {'ns_ac': self.name}
    #
    #     self.ee.ns_manager.add_ns_def(ns_dict)
    #
    #     disc1_builder = self.factory.get_builder_from_module(
    #         'Disc1', self.mod1_path)
    #     disc2_builder = self.factory.get_builder_from_module(
    #         'Disc2', self.mod2_path)
    #
    #     self.factory.set_builders_to_coupling_builder(
    #         [disc1_builder, disc2_builder])
    #
    #     self.ee.configure()
    #     self.ee.display_treeview_nodes()
    #
    #     for key in self.ee.dm.data_id_map:
    #         print("   key", key)
    #     a = 1.0
    #     b = 3.0
    #     x = 99.0
    #     values_dict = {self.name + '.x': x,
    #                    self.name + '.Disc1.a': a,
    #                    self.name + '.Disc1.b': b,
    #                    self.name + '.Disc1.an_input_1': 'value_1',
    #                    self.name + '.Disc1.an_input_2': ['value_2', 'value_3'],
    #                    self.name + '.Disc2.constant': 1.5,
    #                    self.name + '.Disc2.power': 2,
    #                    self.name + '.cache_type': 'SimpleCache',
    #                    f'{self.name}.propagate_cache_to_children': True}
    #
    #     # set input data
    #     self.ee.load_study_from_input_dict(values_dict)
    #
    #     # get disciplines objects
    #     disc1 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc1')[0]
    #     disc2 = self.ee.dm.get_disciplines_with_name('SoSDisc.Disc2')[0]
    #     sos_coupl = self.ee.root_process
    #     n_calls_sosc = n_calls_disc1 = n_calls_disc2 = 0
    #
    #     # first execution
    #     self.ee.execute()
    #     # ref update
    #     n_calls_sosc += 1
    #     n_calls_disc1 += 1
    #     n_calls_disc2 += 1
    #
    #     # check that status are DONE after the first execution both on
    #     # SOSTRADES and GEMSEO side
    #     self.assertEqual(self.ee.root_process.status,
    #                      ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #
    #     # check that status are PENDING after the second prepare execution
    #     self.ee.prepare_execution()
    #     self.assertEqual(self.ee.root_process.status,
    #                      ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].status, ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].status, ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #
    #     # second execution w/o change
    #     self.ee.execute()
    #     # ref update
    #     n_calls_sosc += 0
    #     n_calls_disc1 += 0
    #     n_calls_disc2 += 0
    #
    #     # check that no discipline ran
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
    #     self.assertEqual(
    #         disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
    #     self.assertEqual(
    #         disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)
    #
    #     # check that the status are done after non-execution
    #     self.assertEqual(self.ee.root_process.status,
    #                      ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #
    #     # third execute with change of a private parameter of the second
    #     # discipline
    #     values_dict[f'{self.name}.Disc2.power'] = 3.
    #     self.ee.load_study_from_input_dict(values_dict)
    #
    #     # check that status are PENDING after the second prepare execution
    #     self.ee.prepare_execution()
    #     self.assertEqual(self.ee.root_process.status,
    #                      ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].status, ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].status, ProxyDiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_PENDING)
    #
    #     self.ee.execute()
    #
    #     # ref update
    #     n_calls_sosc += 1
    #     n_calls_disc1 += 0
    #     n_calls_disc2 += 1
    #
    #     # check that disc1 did not run
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_sosc)
    #     self.assertEqual(
    #         disc1.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc1)
    #     self.assertEqual(
    #         disc2.mdo_discipline_wrapp.mdo_discipline.n_calls, n_calls_disc2)
    #
    #     # check that the status are DONE after some disciplines executed and
    #     # some did not
    #     self.assertEqual(self.ee.root_process.status,
    #                      ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].status, ProxyDiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)
    #     self.assertEqual(
    #         self.ee.root_process.proxy_disciplines[1].mdo_discipline_wrapp.mdo_discipline.status, MDODiscipline.STATUS_DONE)


if __name__ == "__main__":
    cls = TestCache()
    cls.setUp()
    cls.test_3_test_cache_coupling_without_input_change()
