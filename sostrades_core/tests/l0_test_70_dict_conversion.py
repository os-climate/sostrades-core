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
import pprint
import unittest
from os.path import join
from tempfile import gettempdir

from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import (
    convert_array_into_new_type,
    convert_new_type_into_array,
)


class TestExtendString(unittest.TestCase):
    """
    Extend string type for GEMSEO test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.name = 'EE'
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)
        self.dump_dir = join(gettempdir(), self.name)

        self.exec_eng = ExecutionEngine(self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc_list_conversion.Disc'
        self.disc_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc', mod_list)

    def test_01_simple_dict_conversion(self):
        """ This test proves the ability to convert simple dict
        {'dict':'float'}, {'dict':'dataframe'} ... into array and
        to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        dict_float = self.exec_eng.dm.get_value('EE.Disc.dict_float')
        var_dict = {'EE.Disc.dict_float': dict_float}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            self.assertDictEqual(conversion_back, dict_float)

        dict_df = self.exec_eng.dm.get_value('EE.Disc.dict_dataframe')
        var_dict = {'EE.Disc.dict_dataframe': dict_df}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key_inner in [f'key{i}' for i in range(1, 6)]:
                assert_frame_equal(conversion_back[key_inner], dict_df[key_inner], check_dtype=False)

        dict_df_array = self.exec_eng.dm.get_value('EE.Disc.dict_dataframe_array')
        var_dict = {'EE.Disc.dict_dataframe_array': dict_df_array}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key_inner in [f'key{i}' for i in range(1, 6)]:
                assert_frame_equal(conversion_back[key_inner], dict_df_array[key_inner],
                                   check_dtype=False)

        dict_array = self.exec_eng.dm.get_value('EE.Disc.dict_array')
        var_dict = {'EE.Disc.dict_array': dict_array}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key_inner in [f'key{i}' for i in range(1, 6)]:
                assert_array_equal(conversion_back[key_inner], dict_array[key_inner])

    def test_02_recursive_dict_conversion(self):
        """ This test proves the ability to convert recursive  dict
        {'dict':{'dict':'float'}}, {'dict':{'dict':'dataframe'}}... into array and
                to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        dict_dict_float = self.exec_eng.dm.get_value('EE.Disc.dict_dict_float')
        var_dict = {'EE.Disc.dict_dict_float': dict_dict_float}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            self.assertDictEqual(conversion_back, dict_dict_float)

        dict_dict_dict_array = self.exec_eng.dm.get_value('EE.Disc.dict_dict_dict_array')
        var_dict = {'EE.Disc.dict_dict_dict_array': dict_dict_dict_array}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key1 in [f'key{i}' for i in range(1, 6)]:
                for key2 in [f'key{i}' for i in range(1, 6)]:
                    for key3 in [f'key{i}' for i in range(1, 6)]:
                        assert_array_equal(conversion_back[key1][key2][key3],
                                           dict_dict_dict_array[key1][key2][key3])

        dict_dict_dataframe = self.exec_eng.dm.get_value('EE.Disc.dict_dict_dataframe')
        var_dict = {'EE.Disc.dict_dict_dataframe': dict_dict_dataframe}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key1 in [f'key{i}' for i in range(1, 6)]:
                for key2 in [f'key{i}' for i in range(1, 6)]:
                    assert_frame_equal(conversion_back[key1][key2],
                                       dict_dict_dataframe[key1][key2],
                                       check_dtype=False)

    def test_03_recursive_dict_list_conversion(self):
        """ This test proves the ability to convert recursive  dict of list
        {'dict':{'list':'float'}}, {'dict':{'list':'dataframe'}}... into array and
                to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        dict_list_float = self.exec_eng.dm.get_value('EE.Disc.dict_list_float')
        var_dict = {'EE.Disc.dict_list_float': dict_list_float}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            self.assertDictEqual(conversion_back, dict_list_float)

        dict_list_list_dataframe = self.exec_eng.dm.get_value('EE.Disc.dict_list_list_dataframe')
        var_dict = {'EE.Disc.dict_list_list_dataframe': dict_list_list_dataframe}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key1 in [f'key{i}' for i in range(1, 6)]:
                for i in range(3):
                    for j in range(5):
                        assert_frame_equal(conversion_back[key1][i][j],
                                           dict_list_list_dataframe[key1][i][j],
                                           check_dtype=False)

        dict_list_dict_dataframe = self.exec_eng.dm.get_value('EE.Disc.dict_list_dict_dataframe')
        var_dict = {'EE.Disc.dict_list_dict_dataframe': dict_list_dict_dataframe}

        for key, value in var_dict.items():
            red_dm = self.exec_eng.dm.reduced_dm.get(key, {})
            converted_inputs, new_reduced_dm = convert_new_type_into_array(key, value, red_dm)

            red_dm.update(new_reduced_dm)
            conversion_back = convert_array_into_new_type(key, converted_inputs, red_dm)
            for key1 in [f'key{i}' for i in range(1, 6)]:
                for i in range(3):
                    for key2 in [f'key{i}' for i in range(1, 6)]:
                        assert_frame_equal(conversion_back[key1][i][key2],
                                           dict_list_dict_dataframe[key1][i][key2],
                                           check_dtype=False)
