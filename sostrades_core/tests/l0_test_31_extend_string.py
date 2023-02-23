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
import pprint
import numpy as np
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from copy import deepcopy
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager


class TestExtendString(unittest.TestCase):
    """
    Extend string type for GEMSEO test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.name = 'EE'
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.dump_dir = join(gettempdir(), self.name)

        exec_eng = ExecutionEngine(self.name)
        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        self.disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)
        self.desc_in = deepcopy(self.disc5_builder.cls.DESC_IN)

    def tearDown(self):
        self.disc5_builder.cls.DESC_IN = self.desc_in
        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_sosdiscipline_simple_dict(self):
        print("\n Test 01 : SoSDiscipline with simple dict")
        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)
        disc5_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}  # add new string variable
        disc5_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}}  # add new string variable

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.z'] = [3., 0.]
        values_dict['EE.dict_out'] = {'key1': 0.5, 'key2': 0.5}
        values_dict['EE.Disc5.an_input_1'] = 'value0'
        values_dict['EE.Disc5.an_input_2'] = ['value1', 'value2']

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        # check inputs
        output_target = {'z': [3., 0.],
                         'dict_out': {'key1': 0.5, 'key2': 0.5},
                         'dict_of_dict_out': {'key1': {'1': 1, '2': 2}},
                         'an_input_1': 'value0',
                         'an_input_2': ['value1', 'value2']}
        disc5 = exec_eng.dm.get_disciplines_with_name('EE.Disc5')[0]
        outputs = disc5.get_sosdisc_inputs()
        self.assertDictEqual({key: value for key, value in outputs.items(
        ) if key not in disc5.NUM_DESC_IN}, output_target)

        # check outputs
        target = {
            'EE.z': [
                3.0, 0.0], 'EE.dict_out': [
                0.5, 0.5], 'EE.h': [
                0.75, 0.75]}

        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is np.array:
                self.assertListEqual(list(target[key]), list(res[key]))

    def test_02_sosdiscipline_string_dict_dump_load(self):
        print("\n Test 01 : SoSDiscipline with simple dict")
        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)
        # string variable
        disc5_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}
        # string list variable
        disc5_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}}
        # dict of string
        disc5_builder.cls.DESC_IN['an_input_3'] = {
            'type': 'dict', 'subtype_descriptor': {'dict': 'string'}}
        # dict of dict of string
        disc5_builder.cls.DESC_IN['an_input_4'] = {
            'type': 'dict', 'subtype_descriptor': {'dict': {'dict': {'dict': 'string'}}}}
        # dict of string list
        disc5_builder.cls.DESC_IN['an_input_5'] = {
            'type': 'dict', 'subtype_descriptor': {'dict': {'list': 'string'}}}
        # dict of dict of string list
        disc5_builder.cls.DESC_IN['an_input_6'] = {
            'type': 'dict', 'subtype_descriptor': {'dict': {'dict': {'list': 'string'}}}}
        disc5_builder.cls.DESC_IN['an_input_7'] = {'type': 'list', 'subtype_descriptor': {
            'list': {'dict': {'dict': {'list': 'string'}}}}}
        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.z'] = [3., 0.]
        values_dict['EE.dict_out'] = {'key1': 0.5, 'key2': 0.5}
        values_dict['EE.Disc5.an_input_1'] = 'STEPS-HEvbzeovbeo(-+=___________f roylgf'
        values_dict['EE.Disc5.an_input_2'] = ['STEPS-HE', 'eee']
        values_dict['EE.Disc5.an_input_3'] = {
            'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
            'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'}
        values_dict['EE.Disc5.an_input_4'] = {'value1': {
            'subkey': 'SagRoGBDIU(-_)=$*!%:;,verrvevfedvbdfjvbdbsvdsbvlksdnbvkmnripe'},
            'value2': {'subkey': 'STEPS-HE'}}
        values_dict['EE.Disc5.an_input_5'] = {
            'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'], 'scenario3': ['string', 1.0, 4.0]}
        values_dict['EE.Disc5.an_input_6'] = {'key_1': {
            'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
            'key_2': {
                'scenario1': ['AC1', 'AC2']}}
        values_dict['EE.Disc5.an_input_7'] = [{'key_1': {
            'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
            'key_2': {
                'scenario1': ['AC1', 'AC2']}}, {'key_11': {
            'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
            'key_22': {
                'scenario1': ['AC1', 'AC2']}}]
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        BaseStudyManager.static_dump_data(
            self.dump_dir, exec_eng, DirectLoadDump())

        ee2 = ExecutionEngine(self.name)

        ee2.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = ee2.factory.get_builder_from_module(
            'Disc5', mod_list)
        # string variable
        disc5_builder.cls.DESC_IN['an_input_1'] = {
            'type': 'string'}
        # string list variable
        disc5_builder.cls.DESC_IN['an_input_2'] = {
            'type': 'list', 'subtype_descriptor': {'list': 'string'}}
        # dict of string
        disc5_builder.cls.DESC_IN['an_input_3'] = {
            'type': 'dict'}
        # dict of dict of string
        disc5_builder.cls.DESC_IN['an_input_4'] = {
            'type': 'dict'}
        # dict of string list
        disc5_builder.cls.DESC_IN['an_input_5'] = {
            'type': 'dict'}
        # dict of dict of string list
        disc5_builder.cls.DESC_IN['an_input_6'] = {
            'type': 'dict'}
        disc5_builder.cls.DESC_IN['an_input_7'] = {'type': 'list', 'subtype_descriptor': {
            'list': {'dict': {'dict': {'list': 'string'}}}}}

        ee2.factory.set_builders_to_coupling_builder(disc5_builder)
        ee2.configure()

        BaseStudyManager.static_load_data(
            self.dump_dir, ee2, DirectLoadDump())

        # check inputs
        output_target = {'z': [3., 0.],
                         'dict_out': {'key1': 0.5, 'key2': 0.5},
                         'dict_of_dict_out': {'key1': {'1': 1, '2': 2}},
                         'an_input_1': 'STEPS-HEvbzeovbeo(-+=___________f roylgf',
                         'an_input_2': ['STEPS-HE', 'eee'],
                         'an_input_3': {'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
                                        'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'},
                         'an_input_4': {
                             'value1': {'subkey': 'SagRoGBDIU(-_)=$*!%:;,verrvevfedvbdfjvbdbsvdsbvlksdnbvkmnripe'},
                             'value2': {'subkey': 'STEPS-HE'}},
                         'an_input_5': {
                             'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'],
                             'scenario3': ['string', 1.0, 4.0]},
                         'an_input_6': {'key_1': {
                             'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                             'key_2': {
                                 'scenario1': ['AC1', 'AC2']}},
                         'an_input_7': [{'key_1': {
                             'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                             'key_2': {
                                 'scenario1': ['AC1', 'AC2']}}, {'key_11': {
                             'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                             'key_22': {
                                 'scenario1': ['AC1', 'AC2']}}]}
        disc5 = ee2.dm.get_disciplines_with_name('EE.Disc5')[0]
        outputs = disc5.get_sosdisc_inputs()
        self.maxDiff = None
        self.assertDictEqual({key: value for key, value in outputs.items(
        ) if key not in disc5.NUM_DESC_IN}, output_target)

        # check outputs
        target = {
            'EE.z': [
                3.0, 0.0], 'EE.dict_out': {'key1': 0.5, 'key2': 0.5}, 'EE.h': [
                0.75, 0.75],
            'EE.Disc5.an_input_1': 'STEPS-HEvbzeovbeo(-+=___________f roylgf',
            'EE.Disc5.an_input_2': ['STEPS-HE', 'eee'],
            'EE.Disc5.an_input_3': {'value1': 'STEPS_bzefivbzei))(((__)----+!!!:;=',
                                    'value2': 'ghzoiegfhzeoifbskcoevgzepgfzocfbuifgupzaihvjsbviupaegviuzabcubvepzgfbazuipbcva'},
            'EE.Disc5.an_input_4': {
                'value1': {'subkey': 'SagRoGBDIU(-_)=$*!%:;,verrvevfedvbdfjvbdbsvdsbvlksdnbvkmnripe'},
                'value2': {'subkey': 'STEPS-HE'}},
            'EE.Disc5.an_input_5': {'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4'],
                                    'scenario3': ['string', 1.0, 4.0]},
            'EE.Disc5.an_input_6': {'key_1': {
                'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                'key_2': {
                    'scenario1': ['AC1', 'AC2']}}}
        ee2.execute()
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if isinstance(res[key], dict):
                self.assertDictEqual(res[key], target[key])
            elif isinstance(res[key], list):
                self.assertListEqual(target[key], res[key])
            elif isinstance(res[key], np.ndarray):
                self.assertListEqual(list(target[key]), list(res[key]))
            else:
                self.assertEqual(res[key], target[key])

        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(self.dump_dir)

    def test_03_couple_strings_withany_types(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc9in_string_coupling.Disc9in'
        disc9in_builder = exec_eng.factory.get_builder_from_module(
            'Disc9in', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc9out_string_coupling.Disc9out'
        disc9out_builder = exec_eng.factory.get_builder_from_module(
            'Disc9out', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc9in_builder, disc9out_builder])
        exec_eng.configure()

        x0 = 2.0
        values_dict = {f'{self.name}.Disc9in.x': x0}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        self.assertEqual(exec_eng.dm.get_value(f'{self.name}.Disc9out.z'), x0)

        disc9in = exec_eng.dm.get_disciplines_with_name('EE.Disc9in')[0]
        outputs = disc9in.get_sosdisc_outputs()

        disc9out = exec_eng.dm.get_disciplines_with_name('EE.Disc9out')[0]
        inputs = disc9out.get_sosdisc_inputs()

        self.assertDictEqual(
            {key: value for key, value in inputs.items() if key not in disc9in.NUM_DESC_IN}, outputs)

        outputs_ref = {'string': 'x is > 0',
                       'string_list': ['&1234(-_)=+6789$%!ABCabc', f'{x0}{x0}{x0}{x0}{x0}{x0}{x0}', 'STEPS-HE'],
                       'string_dict': {'key0': 'STEPS-HE', 'key1': 'positive',
                                       'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'},
                       'string_dict_of_dict': {
                           'dict1': {'key1': 'STEPS-HE', 'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'},
                           'dict2': {'key1': 'positive', 'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}},
                       'dict_mix_types': {'AC1': {'string': 'NA', 'float': 8.0, 'integer': 1, 'list': [2.0, 4.0],
                                                  'dict': {'key1': 'positive',
                                                           'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}},
                                          'AC2': {'string': 'NA', 'float': 16.0, 'integer': 2, 'list': [2.0, 6.0],
                                                  'dict': {'key1': 'positive',
                                                           'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}}},
                       'dict_list': [{'key_1': {
                           'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                           'key_2': {
                               'scenario1': ['AC1', 'AC2']}}, {'key_11': {
                           'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                           'key_22': {
                               'scenario1': ['AC1', 'AC2']}}],
                       'dict_dict_dict_list_string': {'s1': {'key_1': {
                           'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                           'key_2': {
                               'scenario1': ['AC1', 'AC2']}},
                           's2': {'key_11': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_22': {
                                   'scenario1': ['AC1', 'AC2']}}}}
        self.maxDiff = None
        self.assertDictEqual(outputs, outputs_ref)

        for key in outputs_ref:
            full_key = f'{self.name}.{key}'
            key_value = exec_eng.dm.get_value(full_key)
            if isinstance(key_value, dict):
                self.assertDictEqual(key_value, outputs_ref[key])
            elif isinstance(key_value, list):
                self.assertListEqual(key_value, outputs_ref[key])

        for key, key_id in exec_eng.dm.data_id_map.items():
            if key.split('.')[-1] in outputs_ref:
                value = exec_eng.dm.data_dict[key_id]['value']
                coupling = exec_eng.dm.data_dict[key_id]['coupling']
                # all string are coupled strings
                self.assertTrue(coupling)
                ref_value = outputs_ref[key.split('.')[-1]]
                if isinstance(ref_value, dict):
                    self.assertDictEqual(value, ref_value)
                elif isinstance(ref_value, list):
                    self.assertListEqual(value, ref_value)
                else:
                    self.assertEqual(value, ref_value)

    def test_04_check_adding_newstring_in_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc9in_string_coupling.Disc9in'
        disc9in_builder = exec_eng.factory.get_builder_from_module(
            'Disc9in', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc9in_builder])
        exec_eng.configure()

        x0 = -1.0
        values_dict = {f'{self.name}.Disc9in.x': x0}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        self.assertDictEqual(exec_eng.dm.get_value(
            f'{self.name}.string_dict'), {})
        x0 = 2.0
        values_dict = {f'{self.name}.Disc9in.x': x0}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        self.assertDictEqual(exec_eng.dm.get_value(f'{self.name}.string_dict'), {
            'key0': 'STEPS-HE', 'key1': 'positive', 'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'})
