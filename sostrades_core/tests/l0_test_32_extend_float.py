'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from numpy import array
from numpy import float64 as np_float64
from numpy import int32 as np_int32
from numpy import int64 as np_int64

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.compare_data_manager_tooling import dict_are_equal
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import (
    convert_array_into_new_type,
    convert_new_type_into_array,
)


class TestExtendFloat(unittest.TestCase):
    """
    Extend float type for GEMSEO test class
    """

    def setUp(self):
        self.name = 'study'
        self.ee = ExecutionEngine(self.name)

    def test_01_int_and_float_coupling_variables(self):
        self.ee.ns_manager.add_ns('ns_disc1', f'{self.name}.Disc1')

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc0.Disc0'
        disc0_builder = self.ee.factory.get_builder_from_module(
            'Disc0', mod_list)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1_all_types.Disc1'
        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod_list)

        self.ee.factory.set_builders_to_coupling_builder(
            [disc0_builder, disc1_builder])

        self.ee.configure()

        b = 5
        values_dict = {}
        values_dict['study.Disc1.b'] = b
        values_dict['study.Disc1.name'] = 'A1'

        self.ee.load_study_from_input_dict(values_dict)

        # float
        r = 3.22222222222222222222222
        x = 0.22222222222222222222222
        a = 3
        values_dict['study.Disc0.r'] = r

        self.ee.load_study_from_input_dict(values_dict)

        self.assertTrue(isinstance(
            self.ee.dm.get_value('study.Disc0.r'), float))

        target = {'study.Disc0.r': array([r])}
        data_dm = {key: self.ee.dm.get_value(key) for key in target.keys()}

        reconverted_data_dm = {}
        converted_data_dm = {}
        for key, value in data_dm.items():
            red_dm = self.ee.dm.reduced_dm.get(key, {})
            converted_data_dm[key], new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            reconverted_data_dm[key] = convert_array_into_new_type(key, converted_data_dm[key], red_dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(converted_data_dm, target))

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))

        self.ee.execute()
        reconverted_data_dm = {}
        keys_to_convert = ['study.Disc0.r', 'study.Disc1.b', 'study.Disc1.y', 'study.Disc1.x']
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}

        for key, value in data_dm.items():
            red_dm = self.ee.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)
            red_dm.update(new_reduced_dm)
            reconverted_data_dm[key] = convert_array_into_new_type(key, converted_inputs, red_dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))

        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc0.r'], type(r)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], type(b)))
        self.assertEqual(reconverted_data_dm['study.Disc1.y'], a * x + b)
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.x'], type(x)))

        b = np_int32(5)
        values_dict['study.Disc1.b'] = b
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        keys_to_convert = ['study.Disc0.r', 'study.Disc1.b', 'study.Disc1.y', 'study.Disc1.x']
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        for key, value in data_dm.items():
            red_dm = self.ee.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)
            red_dm.update(new_reduced_dm)
            reconverted_data_dm[key] = convert_array_into_new_type(key, converted_inputs, red_dm)

        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc0.r'], type(r)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], type(b)))
        self.assertEqual(reconverted_data_dm['study.Disc1.y'], a * x + b)
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.x'], type(x)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.y'], float))

        # np_int64
        b = np_int64(5)

        values_dict['study.Disc1.b'] = b

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        keys_to_convert = ['study.Disc0.r', 'study.Disc1.b', 'study.Disc1.y', 'study.Disc1.x']
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        for key, value in data_dm.items():
            red_dm = self.ee.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)
            red_dm.update(new_reduced_dm)
            reconverted_data_dm[key] = convert_array_into_new_type(key, converted_inputs, red_dm)

        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc0.r'], type(r)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], type(b)))
        self.assertEqual(reconverted_data_dm['study.Disc1.y'], a * x + b)
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.x'], type(x)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.y'], float))

        # np_float64
        r = np_float64(3.22222222222222222222222)
        x = 0.22222222222222222222222
        a = 3
        values_dict['study.Disc0.r'] = r

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        keys_to_convert = ['study.Disc0.r', 'study.Disc1.b', 'study.Disc1.y', 'study.Disc1.x']
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        for key, value in data_dm.items():
            red_dm = self.ee.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)
            red_dm.update(new_reduced_dm)
            reconverted_data_dm[key] = convert_array_into_new_type(key, converted_inputs, red_dm)
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc0.r'], type(r)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], type(b)))
        self.assertEqual(reconverted_data_dm['study.Disc1.y'], a * x + b)
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.x'], type(x)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.y'], float))
