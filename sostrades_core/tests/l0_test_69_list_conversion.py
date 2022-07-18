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
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import pprint
from os.path import join

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir


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

    def test_01_simple_list_conversion(self):
        """ This test proves the ability to convert simple list
        {'list':'float'}, {'list':'dataframe'} ... into array and
        to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        list_float = self.exec_eng.dm.get_value('EE.Disc.list_float')
        var_dict = {'EE.Disc.list_float': list_float}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_float': conversion_into_array['EE.Disc.list_float']})
        self.assertListEqual(conversion_back['EE.Disc.list_float'], list_float)

        list_df = self.exec_eng.dm.get_value('EE.Disc.list_dataframe')
        var_dict = {'EE.Disc.list_dataframe': list_df}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_dataframe': conversion_into_array['EE.Disc.list_dataframe']})
        for i in range(5):
            assert_frame_equal(conversion_back['EE.Disc.list_dataframe'][i], list_df[i], check_dtype=False)

        list_array = self.exec_eng.dm.get_value('EE.Disc.list_array')
        var_dict = {'EE.Disc.list_array': list_array}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_array': conversion_into_array['EE.Disc.list_array']})
        for i in range(5):
            assert_array_equal(conversion_back['EE.Disc.list_array'][i], list_array[i])

    def test_02_recursive_list_conversion(self):
        """ This test proves the ability to convert recursive  list
        {'list':{'list':'float'}}, {'list':{'list':'dataframe'}}... into array and
                to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        list_list_float = self.exec_eng.dm.get_value('EE.Disc.list_list_float')
        var_dict = {'EE.Disc.list_list_float': list_list_float}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_list_float': conversion_into_array['EE.Disc.list_list_float']})
        self.assertListEqual(conversion_back['EE.Disc.list_list_float'], list_list_float)

        list_list_list_array = self.exec_eng.dm.get_value('EE.Disc.list_list_list_array')
        var_dict = {'EE.Disc.list_list_list_array': list_list_list_array}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_list_list_array': conversion_into_array['EE.Disc.list_list_list_array']})
        for i in range(5):
            for j in range(4):
                for k in range(5):
                    assert_array_equal(conversion_back['EE.Disc.list_list_list_array'][i][j][k],
                                       list_list_list_array[i][j][k])

        list_list_dataframe = self.exec_eng.dm.get_value('EE.Disc.list_list_dataframe')
        var_dict = {'EE.Disc.list_list_dataframe': list_list_dataframe}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_list_dataframe': conversion_into_array['EE.Disc.list_list_dataframe']})
        for i in range(3):
            for j in range(5):
                assert_frame_equal(conversion_back['EE.Disc.list_list_dataframe'][i][j], list_list_dataframe[i][j],
                                   check_dtype=False)

    def test_03_recursive_list_dict_conversion(self):
        """ This test proves the ability to convert recursive  list of dict
        {'list':{'dict':'float'}}, {'list':{'dict':'dataframe'}}... into array and
                to reconvert it back afterward
        """

        builder = self.disc_builder

        # Set builder in factory and configure
        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})
        print(self.exec_eng.display_treeview_nodes())
        disc = self.exec_eng.dm.get_disciplines_with_name('EE.Disc')[0]

        list_dict_float = self.exec_eng.dm.get_value('EE.Disc.list_dict_float')
        var_dict = {'EE.Disc.list_dict_float': list_dict_float}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_dict_float': conversion_into_array['EE.Disc.list_dict_float']})
        self.assertListEqual(conversion_back['EE.Disc.list_dict_float'], list_dict_float)

        list_list_dict_float = self.exec_eng.dm.get_value('EE.Disc.list_list_dict_float')
        var_dict = {'EE.Disc.list_list_dict_float': list_list_dict_float}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_list_dict_float': conversion_into_array['EE.Disc.list_list_dict_float']})
        self.assertListEqual(conversion_back['EE.Disc.list_list_dict_float'], list_list_dict_float)

        list_dict_list_array = self.exec_eng.dm.get_value('EE.Disc.list_dict_list_array')
        var_dict = {'EE.Disc.list_dict_list_array': list_dict_list_array}

        conversion_into_array = disc._convert_new_type_into_array(var_dict)
        conversion_back = disc._convert_array_into_new_type(
            {'EE.Disc.list_dict_list_array': conversion_into_array['EE.Disc.list_dict_list_array']})

        for i in range(3):
            for key1 in [f'key{i}' for i in range(1, 6)]:
                for k in range(5):
                    assert_array_equal(conversion_back['EE.Disc.list_dict_list_array'][i][key1][k],
                                       list_dict_list_array[i][key1][k])
