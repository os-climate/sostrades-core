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

from numpy import allclose, array
from numpy import float64 as np_float64
from numpy import int32 as np_int32

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestExtendFloat(unittest.TestCase):
    """Extend float type for GEMSEO test class"""

    def setUp(self):
        self.study_name = 'study'
        self.ee = ExecutionEngine(self.study_name)

    def test_01_int_and_float_coupling_variables(self):
        self.ee.ns_manager.add_ns('ns_ac', f'{self.study_name}')

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1_disc2_float_coupled.Disc2'
        disc0_builder = self.ee.factory.get_builder_from_module(
            'Disc2', mod_list)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1_disc2_float_coupled.Disc1'
        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod_list)

        self.ee.factory.set_builders_to_coupling_builder(
            [disc0_builder, disc1_builder])

        self.ee.configure()
        x = 10.
        a = int(5)
        y = 25.
        b = 25.
        values_dict = {
            self.study_name + '.x': x,
            self.study_name + '.a': a,
            self.study_name + '.Disc1.b': b,
            self.study_name + '.y': y,
            self.study_name + '.Disc2.indicator': 12.}

        self.ee.load_study_from_input_dict(values_dict)

        self.assertTrue(isinstance(
            self.ee.dm.get_value('study.x'), float))
        self.assertTrue(isinstance(
            self.ee.dm.get_value('study.a'), int))

        # do the prepare execution
        self.ee.prepare_execution()

        target = {'study.x': array([x]), 'study.a':array([a]), 'study.y':array([y])}
        data_dm = {key: self.ee.dm.get_value(key) for key in target.keys()}

        inner_mda_input_data_converter = self.ee.root_process.discipline_wrapp.discipline.inner_mdas[0].input_grammar.data_converter
        self.__convert_data(data_dm, inner_mda_input_data_converter, target)

        self.ee.execute()
        inner_mda_output_data_converter = self.ee.root_process.discipline_wrapp.discipline.inner_mdas[0].output_grammar.data_converter
        keys_to_convert = ['study.x', 'study.Disc1.b', 'study.y', 'study.a']
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        target = {'study.x':array([-3.571428571428573]),
                    'study.Disc1.b':array([25.]),
                    'study.y' :array([-3.571428571428573]),
                    'study.a' : array([8])}
        reconverted_data_dm = self.__convert_data(data_dm, inner_mda_output_data_converter, target)

        self.assertTrue(isinstance(
            reconverted_data_dm['study.x'], type(x)))
        self.assertEqual(reconverted_data_dm['study.y'], reconverted_data_dm['study.a'] * reconverted_data_dm['study.x'] + reconverted_data_dm['study.Disc1.b'])
        self.assertTrue(isinstance(
            reconverted_data_dm['study.y'], type(y)))

        # int32
        a = np_int32(5)
        values_dict['study.a'] = a
        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        keys_to_convert = ['study.x', 'study.a', 'study.y', 'study.Disc1.b']
        inner_mda_output_data_converter = self.ee.root_process.discipline_wrapp.discipline.inner_mdas[0].output_grammar.data_converter
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        reconverted_data_dm = self.__convert_data(data_dm, inner_mda_input_data_converter, target)

        self.assertTrue(isinstance(
            reconverted_data_dm['study.x'], type(x)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.a'], type(a)), f"the type of study.a is {type(reconverted_data_dm['study.a'])} instead of {type(a)}" )
        self.assertEqual(reconverted_data_dm['study.y'], reconverted_data_dm['study.a'] * reconverted_data_dm['study.x'] + reconverted_data_dm['study.Disc1.b'])
        self.assertTrue(isinstance(
            reconverted_data_dm['study.y'], type(y)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], float))

        # np_int64 test has been removed because gemseo doesn't deal with int64, only int32

        # np_float64
        x = np_float64(10.0)
        values_dict['study.x'] = x

        self.ee.load_study_from_input_dict(values_dict)
        self.ee.execute()

        keys_to_convert = ['study.x', 'study.a', 'study.y', 'study.Disc1.b']
        inner_mda_output_data_converter = self.ee.root_process.discipline_wrapp.discipline.inner_mdas[0].output_grammar.data_converter
        data_dm = {key: self.ee.dm.get_value(key) for key in keys_to_convert}
        reconverted_data_dm = self.__convert_data(data_dm, inner_mda_output_data_converter, target)

        self.assertTrue(isinstance(
            reconverted_data_dm['study.x'], type(x)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.a'], type(a)))
        self.assertEqual(reconverted_data_dm['study.y'], reconverted_data_dm['study.a'] * reconverted_data_dm['study.x'] + reconverted_data_dm['study.Disc1.b'])
        self.assertTrue(isinstance(
            reconverted_data_dm['study.y'], type(y)))
        self.assertTrue(isinstance(
            reconverted_data_dm['study.Disc1.b'], float))

    def __convert_data(self, data_dm, data_converter, target):
        reconverted_data_dm = {}
        converted_data_dm = {}
        for key, value in data_dm.items():
            red_dm = {key:value}
            converted_data_dm[key] = data_converter.convert_data_to_array(red_dm.keys(),red_dm)
            red_dm[key] = converted_data_dm[key]

            new_red_dm = data_converter.convert_array_to_data(converted_data_dm[key], {key:slice(0,1,None)})
            reconverted_data_dm[key] = new_red_dm[key]

        # check new_types conversion into array
        for key, value in target.items():
            allclose(value, converted_data_dm.get(key), rtol=0.0001, atol=0.0001)


        # check array conversion into new_types
        for key, value in data_dm.items():
            self.assertAlmostEqual(value, reconverted_data_dm.get(key))

        return reconverted_data_dm

if __name__ == '__main__':
    test = TestExtendFloat()
    test.setUp()
    test.test_01_int_and_float_coupling_variables()
