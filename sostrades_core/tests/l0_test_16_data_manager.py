'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/25-2023/11/03 Copyright 2023 Capgemini

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
import hashlib
from time import sleep
from shutil import rmtree
from os import makedirs
from copy import copy
from os.path import join, dirname
from pathlib import Path
from pickle import dump as pkl_dump

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.tree.serializer import DataSerializer
from sostrades_core.tests.l0_test_06_dict_pickle_import_export_dm import init_dict
from sostrades_core.tools.rw.load_dump_dm_data import CryptedLoadDump, DirectLoadDump


def init_execution_engine_disc1(name, encryption_dir=None):
    # , rw_object=None => no encryption strategy
    if encryption_dir is not None:
        priv_key_f = join(encryption_dir, 'private_key.pem')
        pub_key_f = join(encryption_dir, 'public_key.pem')
        encrypt_ds = CryptedLoadDump(private_key_file=priv_key_f,
                                     public_key_file=pub_key_f)
    else:
        encrypt_ds = None

    exec_eng = ExecutionEngine(name, rw_object=encrypt_ds)
    repo = 'sostrades_core.sos_processes.test'
    exec_eng.select_root_process(repo, 'test_disc1')
    data_dict = {}
    data_dict[name + '.x'] = 2
    data_dict[name + '.Disc1.a'] = 10.
    data_dict[name + '.Disc1.b'] = 5.

    exec_eng.load_study_from_input_dict(data_dict)
    return exec_eng


def init_execution_engine_coupling_disc1_disc2(name):
    exec_eng = ExecutionEngine(name)
    repo = 'sostrades_core.sos_processes.test'
    exec_eng.select_root_process(repo,
                                 'test_disc1_disc2_coupling')
    # modify DM
    data_dict = {}
    data_dict[f'{name}.x'] = 5.
    data_dict[f'{name}.Disc1.a'] = 10.
    data_dict[f'{name}.Disc1.b'] = 20.
    data_dict[f'{name}.Disc2.power'] = 2
    data_dict[f'{name}.Disc2.constant'] = -10.
    exec_eng.load_study_from_input_dict(data_dict)
    return exec_eng


def get_hexdigest(file):
    BLOCK_SIZE = 65536
    file_hash = hashlib.sha256()
    with open(file, 'rb') as f:
        fb = f.read(BLOCK_SIZE)
        while len(fb) > 0:
            file_hash.update(fb)
            fb = f.read(BLOCK_SIZE)

    return file_hash.hexdigest()


class TestDataManagerGenerator(unittest.TestCase):
    """
    Data manager generator test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.ref_dir = join(dirname(__file__), 'data', 'ref_output')

    def tearDown(self):
        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def ignore_fields(self, dict_to_pop):
        for k in ['ns_reference', 'disciplines_dependencies', 'model_origin', 'namespace', 'dataframe_descriptor',
                  'dataframe_edition_locked']:
            for key in dict_to_pop:
                dict_to_pop[key].pop(k, None)
        return dict_to_pop

    def test_01_load_DM(self):
        IO_TYPE = ProxyDiscipline.IO_TYPE
        namespace = 'NPS.CH19_Kero'
        # empty DM to pass to discipline
        ee = init_execution_engine_disc1(namespace)
        ref_study_dir = join(self.ref_dir, namespace)
        disc_dir_to_load_2 = ref_study_dir + '_2'
        dm_data_dict_2 = {}
        dm_data_dict_1 = ee.dm.convert_dict_with_maps(ee.dm.data_dict,
                                                      ee.dm.data_id_map,
                                                      keys='full_names')
        for k, v in dm_data_dict_1.items():
            k_2 = k.replace(namespace, namespace + '_2')
            dm_data_dict_2[k_2] = v
        if Path(disc_dir_to_load_2).is_dir():
            rmtree(disc_dir_to_load_2)
            sleep(0.1)
        makedirs(disc_dir_to_load_2)
        sleep(0.1)
        pkl_dump(dm_data_dict_2, open(join(disc_dir_to_load_2,
                                           DataSerializer.pkl_filename), 'wb'))
        sleep(0.1)

        serializer = DataSerializer()
        get_dm_data_dict_2 = serializer.get_dict_from_study(
            disc_dir_to_load_2, DirectLoadDump())

        assert dm_data_dict_1 != get_dm_data_dict_2

        ns_2 = namespace + '_2'
        ref_dm_pkl_file = join(self.ref_dir, ns_2, DataSerializer.pkl_filename)
        ref_dict = {ns_2 + '.x': init_dict('float'),
                    ns_2 + '.y': init_dict('float'),
                    ns_2 + '.Disc1.a': init_dict('float'),
                    ns_2 + '.Disc1.b': init_dict('float'),
                    ns_2 + '.Disc1.indicator': init_dict('float'),
                    ns_2 + '.Disc1.linearization_mode': init_dict('string'),
                    ns_2 + '.Disc1.cache_type': init_dict('string'),
                    ns_2 + '.Disc1.cache_file_path': init_dict('string'),
                    ns_2 + '.Disc1.debug_mode': init_dict('string'),
                    ns_2 + '.linearization_mode': init_dict('string'),
                    ns_2 + '.linear_solver_MDA': init_dict('string'),
                    ns_2 + '.linear_solver_MDA_preconditioner': init_dict('string'),
                    ns_2 + '.linear_solver_MDO': init_dict('string'),
                    ns_2 + '.linear_solver_MDO_preconditioner': init_dict('string'),
                    ns_2 + '.linear_solver_MDA_options': init_dict('dict'),
                    ns_2 + '.linear_solver_MDO_options': init_dict('dict'),
                    ns_2 + '.cache_type': init_dict('string'),
                    ns_2 + '.cache_file_path': init_dict('string'),
                    ns_2 + '.debug_mode': init_dict('string'),
                    ns_2 + '.warm_start': init_dict('string'),
                    ns_2 + '.acceleration': init_dict('string'),
                    ns_2 + '.inner_mda_name': init_dict('string'),
                    ns_2 + '.max_mda_iter': init_dict('int'),
                    ns_2 + '.epsilon0': init_dict('float'),
                    ns_2 + '.warm_start_threshold': init_dict('float'),
                    ns_2 + '.residuals_history': init_dict('dataframe'),
                    ns_2 + '.n_subcouplings_parallel': init_dict('int'),
                    ns_2 + '.group_mda_disciplines': init_dict('bool'),
                    ns_2 + '.propagate_cache_to_children': init_dict('bool'),
                    ns_2 + '.tolerance_gs': init_dict('float'),
                    ns_2 + '.relax_factor': init_dict('float'), }

        val_dict = {'default': None, 'type': 'string', 'unit': None,
                    'possible_values': None, 'range': None, 'user_level': 1,
                    'visibility': 'Private', 'editable': True, IO_TYPE: 'IN',
                    'model_origin': 'NPS.CH19_Kero.Disc1', 'value': None}
        for var_id in ['n_processes', 'warm_start_threshold',
                       'chain_linearize', 'tolerance', 'use_lu_fact',
                       'linearization_mode', 'cache_type', 'cache_file_path', 'debug_mode']:
            var_n = ns_2 + '.' + var_id
            ref_dict[var_n] = copy(val_dict)
        data_id_map_2 = {}
        for k, v in ee.dm.data_id_map.items():
            k_2 = k.replace(namespace, namespace + '_2')
            data_id_map_2[k_2] = v

        ds = DataSerializer()
        ds.dm_pkl_file = ref_dm_pkl_file
        ds.load_from_pickle(ref_dict, DirectLoadDump())
        ref_dict = self.ignore_fields(ref_dict)
        dm_data_dict_2 = self.ignore_fields(dm_data_dict_2)

        for key in ref_dict:
            if ref_dict[key][ProxyDiscipline.TYPE] != "dataframe":
                self.assertDictEqual(
                    ref_dict[key], dm_data_dict_2[key], msg=f'{key}')
        self.dirs_to_del.append(disc_dir_to_load_2)

    def test_02_DM_with_soscoupling(self):
        study_name = 'EETests'
        exec_engine = init_execution_engine_coupling_disc1_disc2(study_name)
        tv_to_display = exec_engine.display_treeview_nodes()
        exp_disp_tv_list = ['Nodes representation for Treeview EETests',
                            '|_ EETests',
                            '\t|_ Disc1',
                            '\t|_ Disc2']
        self.assertEqual('\n'.join(exp_disp_tv_list),
                         tv_to_display)

        ns = study_name
        x_in = ns + '.x'

        # check data in data manager
        self.assertTrue(exec_engine.dm.check_data_in_dm(x_in))
        self.assertDictEqual(
            exec_engine.dm.get_all_var_name_with_ns_key('x'), {x_in: 'ns_ac'})
        self.assertIn(x_in, exec_engine.dm.get_data_dict_values())

        exec_engine.dm.set_values_from_dict({x_in: 3.})
        exec_engine.execute()
        res = exec_engine.dm.data_dict
        # ref data
        ns_pv_disc1 = ns + '.Disc1'
        ns_pv_disc2 = ns + '.Disc2'
        z = 2490.
        res_target = {
            x_in: 3.,
            ns + '.y': 50.,
            ns + '.z': z,
            ns_pv_disc1 + '.a': 10.,
            ns_pv_disc1 + '.b': 20.,
            ns_pv_disc1 + '.indicator': 200.,
            ns_pv_disc2 + '.constant': -10.,
            ns_pv_disc2 + '.power': 2}

        # check outputs
        for key in res_target:
            key_id = exec_engine.dm.get_data_id(key)
            self.assertEqual(res[key_id]['value'], res_target[key])

        # check data with data manager methods
        self.assertEqual(
            exec_engine.dm.get_data_dict_attr('value')[ns + '.z'], z)
        self.assertListEqual(exec_engine.dm.export_couplings()[
                                 'var_name'].values.tolist(), [ns + '.y'])
        y_id = exec_engine.dm.get_data_id(ns + '.y')
        self.assertTrue(exec_engine.dm.get_var_name_from_uid(y_id), 'y')
        self.assertTrue(exec_engine.dm.get_var_full_name(y_id), ns + '.y')

        # check disciplines with data manager methods
        self.assertListEqual(list(exec_engine.dm.get_io_data_of_disciplines(
            exec_engine.root_process.proxy_disciplines).keys()), ['value', 'type_metadata', 'local_data'])
        self.assertListEqual(list(exec_engine.dm.convert_disciplines_dict_with_full_name(
        ).keys()), ['EETests', 'EETests.Disc1', 'EETests.Disc2'])

        # check status with data manager method
        status_dict_from_dm = exec_engine.dm.build_disc_status_dict()
        for disc in status_dict_from_dm.values():
            self.assertEqual(list(disc.values())[
                                 0], ProxyDiscipline.STATUS_DONE)
        # check status with execution engine method
        status_dict_from_ee = exec_engine.get_anonimated_disciplines_status_dict()
        for disc in status_dict_from_ee.values():
            self.assertEqual(list(disc.values())[
                                 0], ProxyDiscipline.STATUS_DONE)


''' HOW TO UPDATE dm.pkl file (reference dm.data_dict):
go to ref dir (sostrades_core\tests\data\ref_output\<STUDY_DIR>)
import pickle
a_d=pickle.load(open('dm.pkl','rb'))
a_d.update(<DICT>)
pickle.dump(a_d, open('dm.pkl', 'wb'))
'''

if '__main__' == __name__:
    testcls = TestDataManagerGenerator()
    testcls.setUp()
    testcls.test_01_load_DM()
