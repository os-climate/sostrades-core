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
from copy import deepcopy
from os import makedirs
from os.path import join, dirname, basename
from pathlib import Path
from shutil import rmtree, unpack_archive
from time import sleep
from sys import platform
from multiprocessing import cpu_count

from numpy import array
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.tree.serializer import DataSerializer, CSV_SEP, FILE_URL, generate_unique_data_csv
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from tempfile import gettempdir
from sostrades_core.study_manager.base_study_manager import BaseStudyManager

N_CPUS = cpu_count()


def init_dict(dtype, unit=None, value=None, st_name=None, visi=None, ns=None, var_name=None):
    if st_name is not None:
        disc_dep = [st_name]
    else:
        disc_dep = []

    def_dict = deepcopy({'default': None, 'possible_values': None,
                         'range': None, 'user_level': 1,
                         'editable': True, 'io_type': 'in', 'model_origin': 'TestDiscAllTypes.DiscAllTypes',
                         'coupling': False, 'type_metadata': None,
                         'description': None, 'optional': False, 'numerical': False,
                         'disciplines_dependencies': disc_dep, 'meta_input': False})
    def_dict['type'] = dtype
    def_dict['unit'] = unit
    def_dict['value'] = value
    def_dict['visibility'] = visi
    def_dict['namespace'] = ns
    def_dict['var_name'] = var_name
    return def_dict


class TestSerializeDF(unittest.TestCase):
    """
    SerializeDF test class
    """

    def setUp(self):
        self.ns_test = 'TestSerializeDF'
        self.ns_test_key = 'ns_test'
        self.dir_to_del = []
        self.ref_dir = join(dirname(__file__), 'data', 'ref_output')
        self.out_dir = join(dirname(__file__), 'data', 'test_output')

        self.h_data = array([0., 0., 0., 0.])
        self.z_list = [0., 0., 0., 0.]
        self.dict_in_data = {'key0': 0., 'key1': 0.}
        self.df_in_data = DataFrame(array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1],
                                           [0.2, 1.2, 2.2], [-9., -8.7, 1e3]]),
                                    columns=['variable', 'c2', 'c3'])
        self.dict_string_in = {'key_C': '1st string',
                               'key_D': '2nd string'}
        self.list_dict_string_in = [self.dict_string_in, self.dict_string_in]

        self.dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                     'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}
        a_df = DataFrame(array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                         columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': a_df * 3.1416}

        self.root_dir = gettempdir()

    def tearDown(self):
        for dir_to_del in self.dir_to_del:
            rmtree(dir_to_del)
            sleep(0.5)
        sleep(0.5)

    def set_TestDiscAllTypes_ee(self, st_name, proc_n,
                                ns_dict=None, db_dir=None, rw_obj=None):
        if ns_dict is None:
            ns_dict = {self.ns_test_key: self.ns_test}
        exec_eng = self.init_and_configure_ee(st_name, ns_dict, db_dir, rw_obj)
        return self.set_ee_data(exec_eng, st_name, proc_n)

    def init_and_configure_ee(self, st_name, ns_dict, db_dir=None, rw_obj=None):
        exec_eng = ExecutionEngine(st_name, rw_object=rw_obj,
                                   root_dir=db_dir)
        exec_eng.ns_manager.add_ns_def(ns_dict)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc_all_types.DiscAllTypes'

        disc_builder = exec_eng.factory.get_builder_from_module(
            'DiscAllTypes', mod_list)
        exec_eng.factory.set_builders_to_coupling_builder(disc_builder)

        exec_eng.configure()
        return exec_eng

    def set_ee_data(self, exec_eng, st_name, proc_n):
        values_dict = {}
        values_dict[st_name + '.' + proc_n + '.h'] = self.h_data
        values_dict[self.ns_test + '.z_list'] = self.z_list
        values_dict[st_name + '.' + proc_n + '.dict_in'] = self.dict_in_data
        values_dict[st_name + '.' + proc_n + '.df_in'] = self.df_in_data
        values_dict[st_name + '.' + proc_n +
                    '.dict_of_dict_in'] = self.dict_of_dict_in_data
        values_dict[st_name + '.' + proc_n +
                    '.dict_of_df_in'] = self.dict_of_df_in_data
        values_dict[st_name + '.' + proc_n +
                    '.dict_string_in'] = self.dict_string_in
        values_dict[st_name + '.' + proc_n +
                    '.list_dict_string_in'] = self.list_dict_string_in

        exec_eng.load_study_from_input_dict(values_dict)
        return exec_eng

    def test_01_extract_only_dataframe(self):
        arr_array = array([1., 2., 3., 5.])
        list_val = ['a', 'b', 'c']
        df_col = ['col1', 'col2', 'col3', 'col4']
        df_array = array([[1., 2., 4., 8.],
                          [2., 4., 8., 16.], [4., 8., 16., 32.]])
        origin_dict = {'x': {'type': 'float', 'unit': 'meters', 'value': 1.23},
                       'arr': {'type': 'array', 'unit': 'modes', 'value': arr_array},
                       'list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': '#names', 'value': list_val},
                       'df': {'type': 'dataframe', 'unit': None, 'value': DataFrame(df_array, columns=df_col)}}
        test_extract_DF = join(self.out_dir, 'test_extract_DF')
        if not Path(test_extract_DF).is_dir():
            makedirs(test_extract_DF)
            sleep(0.1)

        ds = DataSerializer()
        df = ds.export_data_dict_to_csv(origin_dict, test_extract_DF)

        # test x
        var = 'x'
        assert 1.23 == df.loc[var]['value']
        assert 'meters' == df.loc[var]['unit']
        # assert 'array' == df.loc['arr']['type']
        var_file = join(test_extract_DF, var + '.csv')
        assert not Path(var_file).is_file()
        # test arr
        var = 'arr'
        var_file = join(test_extract_DF, var + '.csv')
        assert FILE_URL + basename(var_file) == df.loc[var]['value']
        assert 'modes' == df.loc[var]['unit']
        # assert 'array' == df.loc['arr']['type']
        assert Path(var_file).is_file()
        var_file_data = read_csv(var_file, delimiter=CSV_SEP, header=0)
        assert_frame_equal(var_file_data, DataFrame(arr_array,
                                                    columns=['value']))
        # test list
        var = 'list'
        var_file = join(test_extract_DF, var + '.csv')
        assert FILE_URL + basename(var_file) == df.loc[var]['value']
        assert '#names' == df.loc[var]['unit']
        # assert 'list' == df.loc[var]['type']
        assert Path(var_file).is_file()
        var_file_data = read_csv(var_file, delimiter=CSV_SEP, header=0)
        assert_frame_equal(var_file_data, DataFrame(list_val,
                                                    columns=['value']))
        # test df
        var = 'df'
        var_file = join(test_extract_DF, var + '.csv')
        assert FILE_URL + basename(var_file) == df.loc[var]['value']
        assert None == df.loc[var]['unit']
        # assert 'dataframe' == df.loc['df']['type']
        var_file_data = read_csv(var_file, delimiter=CSV_SEP, header=0)
        assert_frame_equal(var_file_data, DataFrame(df_array, columns=df_col))

        self.dir_to_del.append(self.out_dir)

    def test_02_parameter_data_getter(self):
        test_name = 'test_parameter_data_getter'
        st_name = 'TestDiscAllTypes'
        proc_n = 'DiscAllTypes'
        exec_eng = self.set_TestDiscAllTypes_ee(st_name, proc_n)
        dm = exec_eng.dm
        exec_eng.execute()
        serializer = DataSerializer()

        serializer.put_dict_from_study(
            self.out_dir, DirectLoadDump(), exec_eng.get_anonimated_data_dict())
        test_dir = join(self.out_dir, test_name)
        if not Path(test_dir).is_dir():
            makedirs(test_dir)
        cr_char = '\n' if platform == 'linux' else '\r\n'
        # z scalar
        param = 'z'
        exp_val = f'value{cr_char}90.0{cr_char}'
        o_f_s = dm.get_parameter_data(self.ns_test + '.' + param)
        o_v = o_f_s.getvalue().decode()
        self.assertEqual(exp_val, o_v,
                         f'diff for variable {param} ref vs out:\n{exp_val}\nVS\n{o_v}')
        # h array
        param = 'h'
        o_f_s = dm.get_parameter_data(st_name + '.' + proc_n + '.' + param)
        o_v = o_f_s.getvalue().decode()
        exp_arr = ['value'] + [str(v) for v in self.h_data] + ['']
        c_o_v = o_v.split(cr_char)
        self.assertListEqual(exp_arr, c_o_v,
                             f'diff for variable {param} ref vs out\n{exp_arr}\nVS\n{o_v}')
        # df_in dataframe
        param = 'df_in'
        exp_val = self.df_in_data.astype(str)
        o_f_s = dm.get_parameter_data(st_name + '.' + proc_n + '.' + param)
        o_v = o_f_s.getvalue().decode()
        o_df = DataFrame.from_records(
            [e.split(CSV_SEP) for e in o_v.split(cr_char)][1:-1], columns=exp_val.columns)
        self.assertTrue(exp_val.equals(o_df),
                        f'diff for variable {param} ref vs out\n{exp_val}\nVS\n{o_df}')
        # dict_in dict of scalars
        param = 'dict_in'
        o_f_s = dm.get_parameter_data(st_name + '.' + proc_n + '.' + param)
        o_v = o_f_s.getvalue().decode()
        c_o_v = [e.split(CSV_SEP) for e in o_v.split(cr_char)][:-1]
        o_dict = dict(c_o_v)
        o_dict.pop('variable')
        exp_val = {}
        for k, v in self.dict_in_data.items():
            exp_val[k] = str(v)
        self.assertDictEqual(exp_val, o_dict,
                             f'diff for variable {param} ref vs out\n{exp_val}\nVS\n{o_dict}')

    def test_03_unique_data_csv_generator(self):
        test_name = 'test_unique_data_csv_generator'
        test_dir = join(self.out_dir, test_name)
        if not Path(test_dir).is_dir():
            makedirs(test_dir)
        # dict_of_df_in dict of dataframe
        generate_unique_data_csv(self.dict_of_df_in_data,
                                 join(test_dir, 'dict_of_df_in.csv'))
        self.dir_to_del.append(test_dir)

    def test_04_test_csv_files_export(self):
        test_name = 'test_8_csv_files_export'
        st_name = 'TestDiscAllTypes'
        proc_n = 'DiscAllTypes'
        exec_eng = self.set_TestDiscAllTypes_ee(st_name, proc_n,
                                                db_dir=self.out_dir)
        export_dir = join(self.out_dir, test_name)
        export_dir_zip = exec_eng.export_data_dict_and_zip(export_dir)
        self.assertTrue(Path(export_dir_zip).is_file())
        self.assertFalse(Path(export_dir).is_dir())
        # Create a ZipFile Object and load sample.zip in it
        unpack_archive(export_dir_zip, self.out_dir)
        sleep(0.1)
        self.assertTrue(Path(export_dir).is_dir())
        dm_val_file = join(export_dir, DataSerializer.val_filename)
        exp_dm_val_file = join(export_dir, 'dm_values.csv')
        self.assertEqual(exp_dm_val_file, dm_val_file)

        prefix_f = st_name + '.' + proc_n + '.'
        for a_file in ['dm_values',
                       prefix_f + 'df_in', prefix_f + 'dict_in',
                       prefix_f + 'dict_of_df_in',
                       prefix_f + 'dict_of_dict_in', prefix_f + 'h']:
            out_f_p = join(export_dir, a_file + '.csv')
            self.assertTrue(Path(out_f_p).is_file(),
                            f'file {out_f_p} does not exist')
            out_df = read_csv(out_f_p, delimiter=CSV_SEP,
                              header=0, index_col=0)
            exp_f_p = join(export_dir, a_file + '.csv')
            ref_df = read_csv(exp_f_p, delimiter=CSV_SEP,
                              header=0, index_col=0)
            if a_file == 'dm_values':
                n_p = ref_df.loc[st_name + '.n_processes']['value']
                key_ns = 'TestDiscAllTypes.n_processes'
                out_df.loc[key_ns]['value'] = n_p

            self.assertTrue(ref_df.equals(out_df),
                            f'exported csv files differ:\n{exp_f_p}\nVS\n{out_f_p}')
        self.dir_to_del.append(self.out_dir)

    def _test_05_load_study_after_execute(self):
        # load process in GUI
        self.name = 'MyCase'
        self.repo = 'sostrades_core.sos_processes.test'
        self.exec_eng = ExecutionEngine(self.name)

        builders = self.exec_eng.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)

        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.name +
                       '.name_list': ['name_1', 'name_2']}
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        constant1 = 10
        constant2 = 20
        power1 = 2
        power2 = 3

        private_val = {}
        private_val[self.name +
                    '.Disc2.name_1.constant'] = constant1
        private_val[self.name + '.Disc2.name_1.power'] = power1
        private_val[self.name +
                    '.Disc2.name_2.constant'] = constant2
        private_val[self.name + '.Disc2.name_2.power'] = power2

        x1 = 2
        a1 = 3
        b1 = 4
        x2 = 4
        a2 = 6
        b2 = 2
        private_val[self.name + '.name_1.x'] = x1
        private_val[self.name + '.name_2.x'] = x2
        private_val[self.name + '.Disc1.name_1.a'] = a1
        private_val[self.name + '.Disc1.name_2.a'] = a2
        private_val[self.name + '.Disc1.name_1.b'] = b1
        private_val[self.name + '.Disc1.name_2.b'] = b2
        self.exec_eng.load_study_from_input_dict(private_val)

        self.exec_eng.execute()

        y1 = self.exec_eng.dm.get_value(self.name + '.name_1.y')
        y2 = self.exec_eng.dm.get_value(self.name + '.name_2.y')
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)

        z1 = self.exec_eng.dm.get_value(self.name + '.name_1.z')
        z2 = self.exec_eng.dm.get_value(self.name + '.name_2.z')
        self.assertEqual(z1, constant1 + y1 ** power1)
        self.assertEqual(z2, constant2 + y2 ** power2)

        z_dict = self.exec_eng.dm.get_value(
            self.name + '.z_dict')
        # Check gather disciplines
        self.assertDictEqual(z_dict, {'name_1': z1, 'name_2': z2})

        y_dict = self.exec_eng.dm.get_value(
            self.name + '.y_dict')
        # Check gather disciplines
        self.assertDictEqual(y_dict, {'name_1': y1, 'name_2': y2})

        dump_dir = join(self.root_dir, self.name)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.name)
        builders = exec_eng2.factory.get_builder_from_process(
            repo=self.repo, mod_id='test_coupling_of_scatter')
        exec_eng2.factory.set_builders_to_coupling_builder(builders)

        exec_eng2.configure()

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())

        y1 = exec_eng2.dm.get_value(self.name + '.name_1.y')
        y2 = exec_eng2.dm.get_value(self.name + '.name_2.y')
        self.assertEqual(exec_eng2.dm.get_value(
            self.name + '.Disc1.name_1.a'), a1)
        self.assertEqual(y1, a1 * x1 + b1)
        self.assertEqual(y2, a2 * x2 + b2)
        self.dir_to_del.append(
            dump_dir)
