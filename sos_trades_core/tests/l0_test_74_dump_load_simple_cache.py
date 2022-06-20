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
from shutil import rmtree
from os.path import join, dirname, exists
from pathlib import Path
from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
from sos_trades_core.sos_processes.test.test_sellar_opt_w_design_var.usecase import Study as study_sellar_opt
from sos_trades_core.sos_processes.test.test_sellar_coupling.usecase import Study as study_sellar_mda
from sos_trades_core.sos_processes.test.test_disc1_disc2_coupling.usecase_coupling_2_disc_test import Study as study_disc1_disc2


class TestLoadSimpleCache(unittest.TestCase):
    """
    Test of SimpleCache dump and load from files
    """

    def setUp(self):
        
        self.dir_to_del = []
        self.ref_dir = join(dirname(__file__), 'data')
        self.repo_name = 'sos_trades_core.sos_processes.test'
        self.proc_name_disc1_disc2 = 'test_disc1_disc2_coupling'
        self.proc_name_sellar_opt = 'test_sellar_opt_w_design_var'
        self.proc_name_sellar_mda = 'test_sellar_coupling'
        self.dump_dir = join(self.ref_dir, 'dumped_cache_test_74')
        self.dir_to_del = []
        
    def tearDown(self):
        for dir in self.dir_to_del:
            if Path(dir).is_dir():
                rmtree(dir)
        
    def test_01_cache_map(self):
        
        study_1 = study_disc1_disc2()
        study_1.load_data()
        study_1.ee.build_cache_map()
        
        study_2 = study_disc1_disc2()
        study_2.load_data()
        study_2.ee.build_cache_map()
        
        # check cache_map equality
        self.assertEqual(len(study_1.ee.dm.cache_map), len(study_2.ee.dm.cache_map))
        
        for study_1_elem, study_2_elem in zip(study_1.ee.dm.cache_map.items(), study_2.ee.dm.cache_map.items()):
            self.assertEqual(study_1_elem[0], study_2_elem[0])
            self.assertEqual(study_1_elem[1], study_2_elem[1])
            self.assertEqual(study_1_elem[1], None)
            
        # activate cache
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)
        
        dict_values = {f'{study_2.study_name}.cache_type': 'SimpleCache'}
        study_2.load_data(from_input_dict=dict_values)
        
        # check cache_map equality
        self.assertEqual(len(study_1.ee.dm.cache_map), len(study_2.ee.dm.cache_map))
        
        for study_1_elem, study_2_elem in zip(study_1.ee.dm.cache_map.items(), study_2.ee.dm.cache_map.items()):
            self.assertEqual(study_1_elem[0], study_2_elem[0])
            self.assertEqual(study_1_elem[1].__class__.__name__, study_2_elem[1].__class__.__name__)
            self.assertEqual(study_1_elem[1].__class__.__name__, 'SimpleCache')
        
    def test_02_dump_and_load_cache_on_disc1_process(self):
        
        dump_dir = join(self.dump_dir, 'test_02')
        
        # run study, then dump dm and disciplines status
        study_dump = study_disc1_disc2()
        study_dump.set_dump_directory(
            dump_dir)
        study_dump.load_data()
        # cache activation
        dict_values = {f'{study_dump.study_name}.cache_type': 'SimpleCache'}
        study_dump.load_data(from_input_dict=dict_values)
        
        # run with dump cache_map
        study_dump.load_cache()
        study_dump.run(dump_study=True)
        
        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sos_trades_core.sos_processes.test', 'test_disc1_disc2_coupling', study_dump.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))
        cache_map_from_pkl = study_dump.setup_cache_map_dict(dump_dir)
         
        # load dumped dm in a new study
        study_load = BaseStudyManager(self.repo_name, self.proc_name_disc1_disc2, study_dump.study_name)
        study_load.load_data(from_path=dump_dir)
        study_load.load_disciplines_data(study_folder_path=dump_dir)
        
        # as in launch_calculation script, load cache and run
        study_load.load_cache(study_folder_path=dump_dir)
        
        # compare cache map of study_dump and study_load
        self.assertListEqual(list(study_load.ee.dm.cache_map.keys()), list(cache_map_from_pkl.keys()))
        
        disc_cache_dump = list(cache_map_from_pkl.values())[0]
        disc_cache_load = list(study_load.ee.dm.cache_map.values())[0]
        
        for disc_cache_id in cache_map_from_pkl.keys():
            disc_cache_dump = cache_map_from_pkl[disc_cache_id]
            disc_cache_load = study_load.ee.dm.cache_map[disc_cache_id]
            
            self.assertListEqual(disc_cache_dump.inputs_names, disc_cache_load.inputs_names)
            self.assertListEqual(disc_cache_dump.outputs_names, disc_cache_load.outputs_names)
            self.assertDictEqual(disc_cache_dump.get_last_cached_inputs(), disc_cache_load.get_last_cached_inputs())
            self.assertDictEqual(disc_cache_dump.get_last_cached_outputs(), disc_cache_load.get_last_cached_outputs())
            self.assertListEqual(list(disc_cache_dump.get_outputs(disc_cache_load.get_last_cached_inputs(), disc_cache_dump.inputs_names)),
                                 list(disc_cache_load.get_outputs(disc_cache_load.get_last_cached_inputs(), disc_cache_load.inputs_names)))
        
        # run study and check if cache is used
        study_load.run()
        for disc_id in study_dump.ee.dm.gemseo_disciplines_id_map.keys():
            disc_dump = study_dump.ee.dm.gemseo_disciplines_id_map[disc_id]
            disc_load = study_load.ee.dm.gemseo_disciplines_id_map[disc_id]
            self.assertEqual(disc_load.n_calls, disc_dump.n_calls - 1)
                
        self.dir_to_del.append(self.dump_dir)
                
    def test_03_dump_and_load_cache_None(self):
        
        dump_dir = join(self.dump_dir, 'test_03')
        
        # run study, then dump dm and disciplines status
        study_dump = study_disc1_disc2()
        study_dump.set_dump_directory(
            dump_dir)
        study_dump.load_data()
        
        # load empty cache
        self.assertEqual(study_dump.ee.dm.gemseo_disciplines_id_map, None)
        self.assertEqual(study_dump.ee.dm.cache_map, None)
        study_dump.load_cache()
        self.assertEqual(study_dump.ee.dm.gemseo_disciplines_id_map, {})
        self.assertEqual(study_dump.ee.dm.cache_map, {})

        # run with dump cache_map
        study_dump.run(dump_study=True)
        
        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sos_trades_core.sos_processes.test', 'test_disc1_disc2_coupling', study_dump.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))
         
        # load dumped dm in a new study
        study_load = BaseStudyManager(self.repo_name, self.proc_name_disc1_disc2, study_dump.study_name)
        study_load.load_data(from_path=dump_dir)
        study_load.load_disciplines_data(study_folder_path=dump_dir)
        study_load.load_cache(study_folder_path=dump_dir)
        
        # run study and check if cache is used
        study_load.run()
        
        self.assertEqual(study_load.ee.dm.gemseo_disciplines_id_map, {})
        self.assertEqual(study_load.ee.dm.cache_map, {})
        
        for disc in study_load.ee.factory.sos_disciplines:
            self.assertEqual(disc.n_calls, 1)
        
        self.dir_to_del.append(self.dump_dir)
            
    def test_04_dump_and_load_disc1_cache(self):
        
        dump_dir = join(self.dump_dir, 'test_04')
        
        # run study, then dump dm and disciplines status
        study_1 = study_disc1_disc2()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()
        # cache activation for Disc1
        dict_values = {f'{study_1.study_name}.Disc1.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)
        
        # run with dump cache_map
        study_1.load_cache()
        study_1.run(dump_study=True)
        
        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sos_trades_core.sos_processes.test', 'test_disc1_disc2_coupling', study_1.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))
        cache_map_from_pkl = study_1.setup_cache_map_dict(dump_dir)
        self.assertEqual(len(cache_map_from_pkl), 1)
        
        # load dumped dm in a new study
        study_2 = BaseStudyManager(self.repo_name, self.proc_name_disc1_disc2, study_1.study_name)
        study_2.load_data(from_path=dump_dir)
        study_2.load_disciplines_data(study_folder_path=dump_dir)
        study_2.load_cache(study_folder_path=dump_dir)
        
        # run study and check if cache is used
        study_2.run()
        
        for disc in study_2.ee.factory.sos_disciplines:
            if disc.name == 'Disc1':
                self.assertEqual(disc.n_calls, 0)
            else:
                self.assertEqual(disc.n_calls, 1)
                
        self.dir_to_del.append(self.dump_dir)
        
    def test_05_set_recursive_cache_coupling(self):
        
        dump_dir = join(self.dump_dir, 'test_05')

        study = study_disc1_disc2()
        study.load_data()
        
        values_dict = {f'{study.study_name}.cache_type': 'SimpleCache'}
        study.load_data(from_input_dict=values_dict)
        study.load_cache(dump_dir)
        study.dump_cache(dump_dir)
        
        self.assertEqual(len(study.ee.dm.cache_map), 4)
        
        values_dict = {f'{study.study_name}.cache_type': 'None'}
        study.load_data(from_input_dict=values_dict)
        study.load_cache(dump_dir)
        study.dump_cache(dump_dir)
        
        self.assertEqual(len(study.ee.dm.cache_map), 0)
        
        study.run()
        self.assertEqual(len(study.ee.dm.cache_map), 0)
        
        self.dir_to_del.append(self.dump_dir)
        
    def test_06_load_cache_on_sellar_mda(self):
        
        dump_dir = join(self.dump_dir, 'test_06')
        
        # create study sellar MDA and load data from usecase
        study_1 = study_sellar_mda()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()
        
        # cache activation
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)
        study_1.load_cache(dump_dir)
        
        # check cache_map completed but each cache is empty
        self.assertEqual(len(study_1.ee.dm.cache_map), 8)
        for cache in study_1.ee.dm.cache_map.values():
            self.assertEqual(cache.get_last_cached_inputs(), None)
        
        # run MDA
        study_1.run()
        
        # check n_calls
        for disc in study_1.ee.dm.gemseo_disciplines_id_map.values():
            if disc.name in ['Sellar_1', 'Sellar_2']:
                self.assertEqual(disc.n_calls, 7)
            else:
                self.assertEqual(disc.n_calls, 1)
                
        # check cache are filled with last cached inputs and outputs 
        for cache in study_1.ee.dm.cache_map.values():
            self.assertNotEqual(cache.get_last_cached_inputs(), None)
            self.assertNotEqual(cache.get_last_cached_outputs(), None)
            
        # run study_1 with converged MDA
        study_1.run()
        
        # dump data and cache
        study_1.dump_data(dump_dir)
        study_1.dump_cache(dump_dir)
        
        # create new study from dumped data and cache of study_1 and run
        study_2 = BaseStudyManager(self.repo_name, self.proc_name_sellar_mda, study_1.study_name)
        study_2.load_data(from_path=dump_dir)
        study_2.load_cache(dump_dir)
        study_2.run()
        
        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls, 0)
        
        # run again
        study_2.run()
        
        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls, 0)
            
        self.dir_to_del.append(self.dump_dir)

    def test_07_load_cache_on_sellar_opt(self):
        
        dump_dir = join(self.dump_dir, 'test_07')
        
        # create study sellar opt and load data from usecase
        study_1 = study_sellar_opt()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()
        
        # cache activation
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)
        study_1.load_cache(dump_dir)
        
        # check cache_map completed but each cache is empty
        self.assertEqual(len(study_1.ee.dm.cache_map), 11)
        for cache in study_1.ee.dm.cache_map.values():
            self.assertEqual(cache.get_last_cached_inputs(), None)
        
        # run sellar opt
        study_1.run()
        
        # check n_calls
        for disc in study_1.ee.dm.gemseo_disciplines_id_map.values():
            print(disc.name, disc.n_calls)
                
        # check cache are filled with last cached inputs and outputs 
        for cache in study_1.ee.dm.cache_map.values():
            self.assertNotEqual(cache.get_last_cached_inputs(), None)
            self.assertNotEqual(cache.get_last_cached_outputs(), None)
            
        # run study_1 with converged opt
#         study_1.run()
        
        # check n_calls
#         for disc in study_1.ee.dm.gemseo_disciplines_id_map.values():
#             print(disc.name, disc.n_calls)
                
        # dump data and cache
        study_1.dump_data(dump_dir)
        study_1.dump_cache(dump_dir)
        
        # create new study from dumped data and cache of study_1 and run
        study_2 = BaseStudyManager(self.repo_name, self.proc_name_sellar_opt, study_1.study_name)
        study_2.load_data(from_path=dump_dir)
        study_2.load_cache(dump_dir)
        study_2.run()
        
        # check n_calls
#         for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
#             print(disc.name, disc.n_calls)

        # run again
#         study_2.run()
        
        # check n_calls
#         for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
#             print(disc.name, disc.n_calls)
            
        
if '__main__' == __name__:
    cls = TestLoadSimpleCache()
    cls.setUp()
#     cls.test_01_cache_map()
#     cls.test_02_dump_and_load_cache_on_disc1_process()
#     cls.test_03_dump_and_load_cache_None()
#     cls.test_04_dump_and_load_disc1_cache()
#     cls.test_05_set_recursive_cache_coupling()
    cls.test_06_load_cache_on_sellar_mda()
#     cls.test_07_load_cache_on_sellar_opt()
    cls.tearDown()

