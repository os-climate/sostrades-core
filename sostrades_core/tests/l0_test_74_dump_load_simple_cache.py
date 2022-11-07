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
from gemseo.caches.simple_cache import SimpleCache

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from shutil import rmtree
from os.path import join, dirname, exists
from pathlib import Path
from time import sleep
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
#from sostrades_core.sos_processes.test.test_sellar_opt_w_design_var.usecase import Study as study_sellar_opt
from sostrades_core.sos_processes.test.test_sellar_coupling.usecase import Study as study_sellar_mda
from sostrades_core.sos_processes.test.test_disc1_disc2_coupling.usecase_coupling_2_disc_test import Study as study_disc1_disc2


class TestLoadSimpleCache(unittest.TestCase):
    """
    Test of SimpleCache dump and load from files
    """

    def setUp(self):

        self.dir_to_del = []
        self.ref_dir = join(dirname(__file__), 'data')
        self.repo_name = 'sostrades_core.sos_processes.test'
        self.proc_name_disc1_disc2 = 'test_disc1_disc2_coupling'
        self.proc_name_sellar_opt = 'test_sellar_opt_w_design_var'
        self.proc_name_sellar_mda = 'test_sellar_coupling'
        self.dump_dir = join(self.ref_dir, 'dumped_cache_test_74')
        self.dir_to_del = []

    def tearDown(self):
        for dir in self.dir_to_del:
            if Path(dir).is_dir():
                rmtree(dir)
                sleep(2)

    def test_01_dump_and_load_cache_None(self):
        '''
        Test when cache is None that file is not written + map are empty
        '''
        dump_dir = join(self.dump_dir, 'test_01')

        # run study, then dump dm and disciplines status
        study_dump = study_disc1_disc2()
        study_dump.set_dump_directory(
            dump_dir)
        study_dump.load_data()
        dict_values = {f'{study_dump.study_name}.cache_type': 'None'}
        study_dump.load_data(from_input_dict=dict_values)
        # Still no call to any build map
        self.assertEqual(study_dump.ee.dm.gemseo_disciplines_id_map, None)
        self.assertEqual(study_dump.ee.dm.cache_map, None)

        # run with dump cache_map
        study_dump.run(dump_study=True)
        # cache_type is None then cache map is {}
        self.assertEqual(study_dump.ee.dm.cache_map, {})
        # check dumped cache pickle non existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study_dump.study_name, 'cache.pkl')
        # Do not create the cache.pkl file if dict is empty
        self.assertFalse(exists(cache_pkl_path))

        # load dumped dm in a new study
        study_load = BaseStudyManager(
            self.repo_name, self.proc_name_disc1_disc2, study_dump.study_name)
        study_load.load_data(from_path=dump_dir)
        study_load.load_disciplines_data(study_folder_path=dump_dir)
        study_dump.read_cache_pickle(study_folder_path=dump_dir)
        # run study and check if cache is used
        study_load.run()

        self.assertEqual(study_dump.ee.dm.cache_map, {})

        for disc in study_load.ee.factory.proxy_disciplines:
            self.assertEqual(
                disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 1)

        self.dir_to_del.append(self.dump_dir)

    def test_02_same_cache_map_with_2_different_studies(self):
        '''
        The objective is to test that two separate studies give the same key_uid for the same disciplines
        And to test that the modification of the cache_type is transmitted after a prepare execution to the build_map (need to be reconstructed if it was None and none is not an object
        '''
        study_1 = study_disc1_disc2()
        study_1.load_data()

        study_2 = study_disc1_disc2()
        study_2.load_data()

        # activate cache
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)
        study_1.ee.prepare_execution()
        study_1.ee.build_cache_map()

        dict_values = {f'{study_2.study_name}.cache_type': 'SimpleCache'}
        study_2.load_data(from_input_dict=dict_values)
        study_2.ee.prepare_execution()
        study_2.ee.build_cache_map()

        # check cache_map equality
        self.assertEqual(len(study_1.ee.dm.cache_map),
                         len(study_2.ee.dm.cache_map))

        for study_1_elem, study_2_elem in zip(study_1.ee.dm.cache_map.items(), study_2.ee.dm.cache_map.items()):
            self.assertEqual(study_1_elem[0], study_2_elem[0])
            self.assertEqual(
                study_1_elem[1].__class__.__name__, study_2_elem[1].__class__.__name__)
            self.assertTrue(isinstance(study_1_elem[1], SimpleCache))

    def test_03_dump_simple_cache_check_anonymization(self):
        '''
        Check anonymization before dumping the pickle file 
        And check unanonymization
        '''
        dump_dir = join(self.dump_dir, 'test_03')

        # run study, then dump dm and disciplines status
        study_dump = study_disc1_disc2()
        study_dump.load_data()
        # cache activation
        dict_values = {f'{study_dump.study_name}.cache_type': 'SimpleCache'}
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.set_dump_directory(dump_dir)

        # run with dump cache_map with anonymzation
        study_dump.execution_engine.execute()
        study_dump.execution_engine.build_cache_map()
        initial_cache_map = study_dump.execution_engine.dm.cache_map
        anonymized_cache_map = study_dump.execution_engine.get_cache_map_to_dump()

        # Check structure of anonymized_serialized_cache
        for disc_id, serialized_cache in anonymized_cache_map.items():
            self.assertTrue(isinstance(serialized_cache, dict))
            self.assertTrue(list(serialized_cache[1].keys()), [
                            'inputs', 'outputs'])

        for disc_id, disc_cache in initial_cache_map.items():
            self.assertTrue(disc_id in anonymized_cache_map)
            disc_cache_data = disc_cache.get_all_data()[1]
            for var_type in ['inputs', 'outputs']:
                for input_var, input_value in disc_cache_data[var_type].items():
                    input_var_anonymized = input_var.replace(
                        study_dump.study_name, '<study_ph>')
                    self.assertTrue(
                        input_var_anonymized in anonymized_cache_map[disc_id][1][var_type])
                    serialized_input_value = anonymized_cache_map[
                        disc_id][1][var_type][input_var_anonymized]
                    if isinstance(input_value, (float, int, str)):
                        self.assertEqual(input_value, serialized_input_value)
        study_dump.manage_dump_cache()
        study_dump.dump_cache(dump_dir)
        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))

        unanonymized_cache_map = study_dump.execution_engine.unanonymize_caches_in_cache_map(
            anonymized_cache_map)
        for disc_id, disc_cache in initial_cache_map.items():
            for var_type in ['inputs', 'outputs']:

                self.assertDictEqual(disc_cache.get_all_data()[
                                     1][var_type], unanonymized_cache_map[disc_id][1][var_type])
        self.dir_to_del.append(self.dump_dir)

    def test_04_dump_and_load_simple_cache_on_process(self):
        '''
        Dump and load a simple cache on the entire process with same name to check all values
        '''
        dump_dir = join(self.dump_dir, 'test_04')

        # run study, then dump dm and disciplines status
        study_dump = study_disc1_disc2()
        study_dump.load_data()
        # cache activation
        dict_values = {f'{study_dump.study_name}.cache_type': 'SimpleCache'}
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.set_dump_directory(dump_dir)

        # run with dump cache_map
        study_dump.run(dump_study=True)

        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study_dump.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))
        cache_map_from_pkl = study_dump.execution_engine.dm.cache_map

        # load dumped dm in a new study
        study_load = BaseStudyManager(
            self.repo_name, self.proc_name_disc1_disc2, study_dump.study_name)
        study_load.load_data(from_path=dump_dir)
        study_load.load_disciplines_data(study_folder_path=dump_dir)

        # as in launch_calculation script, load cache and run
        study_load.read_cache_pickle(study_folder_path=dump_dir)

        # compare cache map of study_dump and study_load
        self.assertListEqual(
            list(study_load.loaded_cache.keys()), list(cache_map_from_pkl.keys()))

        # run study and check if cache is used
        study_load.run()

        self.dir_to_del.append(self.dump_dir)
        for disc_id in study_dump.ee.dm.gemseo_disciplines_id_map.keys():
            disc_dump = study_dump.ee.dm.gemseo_disciplines_id_map[disc_id]
            disc_load = study_load.ee.dm.gemseo_disciplines_id_map[disc_id]
            self.assertEqual(disc_load.n_calls, 0)
            self.assertEqual(disc_dump.n_calls, 1)

        disc_cache_dump = list(cache_map_from_pkl.values())[0]
        disc_cache_load = list(study_load.ee.dm.cache_map.values())[0]

        for disc_cache_id in cache_map_from_pkl.keys():
            disc_cache_dump = cache_map_from_pkl[disc_cache_id]
            disc_cache_load = study_load.ee.dm.cache_map[disc_cache_id]

            self.assertListEqual(disc_cache_dump.inputs_names,
                                 disc_cache_load.inputs_names)
            self.assertListEqual(disc_cache_dump.outputs_names,
                                 disc_cache_load.outputs_names)
            cache_dump_outputs = disc_cache_dump.get_last_cached_outputs()
            cache_load_outputs = disc_cache_load.get_last_cached_outputs()
            self.assertDictEqual({key: value for key, value in cache_dump_outputs.items() if not key.endswith('residuals_history')},
                                 {key: value for key, value in cache_load_outputs.items() if not key.endswith('residuals_history')})
            cache_dump_get_outputs = list(disc_cache_dump.get_outputs(
                disc_cache_load.get_last_cached_inputs(), disc_cache_dump.inputs_names))[0]
            cache_load_get_outputs = list(disc_cache_load.get_outputs(
                disc_cache_load.get_last_cached_inputs(), disc_cache_load.inputs_names))[0]
            self.assertDictEqual({key: value for key, value in cache_dump_get_outputs.items() if not key.endswith('residuals_history')},
                                 {key: value for key, value in cache_load_get_outputs.items() if not key.endswith('residuals_history')})

        self.dir_to_del.append(self.dump_dir)

    def test_05_dump_and_load_simple_cache_only_one_disc(self):
        '''
        Dump and load a simple cache on ony one disc to test the configuration +  different process names
        '''
        dump_dir = join(self.dump_dir, 'test_05')

        # run study, then dump dm and disciplines status
        study_1 = study_disc1_disc2()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()
        # cache activation for Disc1
        dict_values = {f'{study_1.study_name}.Disc1.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)

        # run with dump cache_map
        study_1.run(dump_study=True)

        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study_1.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))
        study_1.read_cache_pickle(dump_dir)
        cache_map_from_pkl = study_1.loaded_cache
        self.assertEqual(len(cache_map_from_pkl), 1)

        # load dumped dm in a new study
        study_2 = BaseStudyManager(
            self.repo_name, self.proc_name_disc1_disc2, 'new_study')
        study_2.load_data(from_path=dump_dir)
        study_2.load_disciplines_data(study_folder_path=dump_dir)
        study_2.read_cache_pickle(study_folder_path=dump_dir)

        # run study and check if cache is used
        study_2.run()

        for disc in study_2.ee.factory.proxy_disciplines:
            if disc.get_disc_full_name() == f'{study_2.study_name}.Disc1':
                self.assertEqual(
                    disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 0)
            else:
                self.assertEqual(
                    disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 1)

        self.dir_to_del.append(self.dump_dir)

    def test_06_set_different_cache_type(self):
        '''
        Test different cache type to verify reconfiguration rest_cache mode 
        and to verify that pkl is deleted when cache is None
        '''
        dump_dir = join(self.dump_dir, 'test_06')

        study = study_disc1_disc2()
        study.load_data()

        values_dict = {f'{study.study_name}.cache_type': 'SimpleCache'}
        study.load_data(from_input_dict=values_dict)
        study.set_dump_directory(
            dump_dir)
        study.run(dump_study=True)

        for disc in study.ee.factory.proxy_disciplines:
            self.assertEqual(
                disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 1)

        self.assertEqual(len(study.ee.dm.cache_map), 4)
        study.read_cache_pickle(dump_dir)
        study.run(dump_study=True)

        for disc in study.ee.factory.proxy_disciplines:
            self.assertEqual(
                disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 1)

        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))

        values_dict = {f'{study.study_name}.cache_type': 'None'}
        study.load_data(from_input_dict=values_dict)
        study.read_cache_pickle(dump_dir)
        study.run(dump_study=True)

        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study.study_name, 'cache.pkl')
        self.assertFalse(exists(cache_pkl_path))

        self.assertEqual(len(study.ee.dm.cache_map), 0)

        for disc in study.ee.factory.proxy_disciplines:
            self.assertEqual(
                disc.mdo_discipline_wrapp.mdo_discipline.n_calls, 2)

        study.run()
        self.assertEqual(len(study.ee.dm.cache_map), 0)

        self.dir_to_del.append(self.dump_dir)

    def test_07_load_cache_on_sellar_mda(self):

        dump_dir = join(self.dump_dir, 'test_07')

        # create study sellar MDA and load data from usecase
        study_1 = study_sellar_mda()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()

        # cache activation
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache'}
        study_1.load_data(from_input_dict=dict_values)

        # run MDA
        study_1.run(dump_study=True)

        # check cache are filled with last cached inputs and outputs
        for cache in study_1.ee.dm.cache_map.values():
            self.assertNotEqual(cache.get_last_cached_inputs(), None)
            self.assertNotEqual(cache.get_last_cached_outputs(), None)

        # run study_1 with converged MDA
        study_1.run(dump_study=True)

        # create new study from dumped data and cache of study_1 and run
        study_2 = BaseStudyManager(
            self.repo_name, self.proc_name_sellar_mda, 'new_study')
        study_2.load_data(from_path=dump_dir)
        study_2.read_cache_pickle(dump_dir)
        study_2.run()

        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(
                disc.n_calls, 0)

        # run again
        study_2.run()

        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(
                disc.n_calls, 0)

        self.dir_to_del.append(self.dump_dir)

    def test_08_copy_cache_with_copy_study(self):

        dump_dir = join(self.dump_dir, 'test_08')

        study = study_disc1_disc2()
        study.load_data()

        values_dict = {f'{study.study_name}.cache_type': 'SimpleCache'}
        study.load_data(from_input_dict=values_dict)
        study.set_dump_directory(
            dump_dir)
        study.run(dump_study=True)
        study.read_cache_pickle(dump_dir)
        cache = study.loaded_cache
        # check dumped cache pickle existence
        cache_pkl_path = join(dump_dir, 'sostrades_core.sos_processes.test',
                              'test_disc1_disc2_coupling', study.study_name, 'cache.pkl')
        self.assertTrue(exists(cache_pkl_path))

        # create new study in new directory and new name
        new_dump_dir = join(self.dump_dir, 'new_test_08')
        study_2 = BaseStudyManager(
            self.repo_name, self.proc_name_disc1_disc2, 'new_study')

        # load the data from the old directory
        study_2.load_data(dump_dir, display_treeview=False)

        study_2.set_dump_directory(
            new_dump_dir)
        # study_2.load_disciplines_data(dump_dir)
        study_2.read_cache_pickle(dump_dir)

        # check dumped cache pickle existence
        new_cache_pkl_path = join(new_dump_dir, 'sostrades_core.sos_processes.test',
                                  'test_disc1_disc2_coupling', study_2.study_name, 'cache.pkl')
        self.assertFalse(exists(new_cache_pkl_path))
        # Save it in the new directory
        study_2.dump_study(study_2.dump_directory)
        # check dumped cache pickle existence
        new_cache_pkl_path = join(new_dump_dir, 'sostrades_core.sos_processes.test',
                                  'test_disc1_disc2_coupling', study_2.study_name, 'cache.pkl')
        self.assertTrue(exists(new_cache_pkl_path))
        study_2.read_cache_pickle(new_dump_dir)
        new_cache = study_2.loaded_cache
        self.assertListEqual(list(cache.keys()), list(new_cache.keys()))
        for key in cache.keys():
            self.assertDictEqual(
                {key_out: value for key_out, value in cache[key][1]['outputs'].items(
                ) if not key_out.endswith('residuals_history')},
                {key_out: value for key_out, value in new_cache[key][1]['outputs'].items() if not key_out.endswith('residuals_history')})

        study_2.run()

        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(
                disc.n_calls, 0)

        self.dir_to_del.append(dump_dir)
        self.dir_to_del.append(new_dump_dir)

    def test_09_load_cache_on_sellar_mda_newton_raphson(self):

        dump_dir = join(self.dump_dir, 'test_07')

        # create study sellar MDA and load data from usecase
        study_1 = study_sellar_mda()
        study_1.set_dump_directory(
            dump_dir)
        study_1.load_data()

        # cache activation
        dict_values = {f'{study_1.study_name}.cache_type': 'SimpleCache',
                       f'{study_1.study_name}.SellarCoupling.sub_mda_class': 'MDANewtonRaphson'}
        study_1.load_data(from_input_dict=dict_values)

        # run MDA
        study_1.run(dump_study=True)

        # check cache are filled with last cached inputs and outputs
        study_1.read_cache_pickle(dump_dir)

        serialized_cache_map = study_1.loaded_cache
        unanonymized_cache_map = study_1.execution_engine.unanonymize_caches_in_cache_map(
            serialized_cache_map)
        # check that jacobian is well anonymized
        for disc_id, serialized_cache in serialized_cache_map.items():
            if 'jacobian' in serialized_cache[1]:
                for key_out, injacobian in serialized_cache[1]['jacobian'].items():
                    self.assertTrue(key_out.startswith('<study_ph>'))
                    key_out_unanonymized = key_out.replace(
                        '<study_ph>', study_1.study_name)
                    self.assertTrue(
                        key_out_unanonymized in unanonymized_cache_map[disc_id][1]['jacobian'])
                    for key_in, jac in injacobian.items():
                        key_in_unanonymized = key_in.replace(
                            '<study_ph>', study_1.study_name)
                        self.assertTrue(key_in.startswith('<study_ph>'))
                        self.assertTrue(
                            unanonymized_cache_map[disc_id][1]['jacobian'][key_out_unanonymized][key_in_unanonymized] == jac)
        # run study_1 with converged MDA
        study_1.run(dump_study=True)

        # create new study from dumped data and cache of study_1 and run
        study_2 = BaseStudyManager(
            self.repo_name, self.proc_name_sellar_mda, study_1.study_name)
        study_2.load_data(from_path=dump_dir)
        study_2.read_cache_pickle(dump_dir)

        # run study_2 with cache of study_1
        study_2.run()

        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls, 0)
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls_linearize, 0)
        # run again
        study_2.run()

        # check n_calls == 0
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls, 0)
        for disc in study_2.ee.dm.gemseo_disciplines_id_map.values():
            self.assertEqual(disc.n_calls_linearize, 0)
        self.dir_to_del.append(self.dump_dir)

#     def _test_08_load_cache_on_sellar_opt(self):
#
#         dump_dir = join(self.dump_dir, 'test_08')
#
#         # create study sellar opt and load data from usecase
#         #study_opt = study_sellar_opt()
#         study_opt.set_dump_directory(
#             dump_dir)
#         study_opt.load_data()
#
#         # cache activation
#         dict_values = {f'{study_opt.study_name}.cache_type': 'SimpleCache',
#                        f'{study_opt.study_name}.SellarOptimScenario.max_iter': 10,
#                        f'{study_opt.study_name}.SellarOptimScenario.SellarCoupling.sub_mda_class': 'MDANewtonRaphson'}
#         study_opt.load_data(from_input_dict=dict_values)
#         study_opt.load_cache(dump_dir)
#
#         # run sellar opt
#         study_opt.run()
#
#         # check n_calls
#         n_calls_1 = {}
#         for i, disc in enumerate(study_opt.ee.dm.gemseo_disciplines_id_map.values()):
#             n_calls_1[(i, disc[0].name)] = disc[0].n_calls
#
#         # run study_opt after convergence
#         study_opt.run()
#
#         # check n_calls
#         n_calls_2 = {}
#         for i, disc in enumerate(study_opt.ee.dm.gemseo_disciplines_id_map.values()):
#             n_calls_2[(i, disc[0].name)] = disc[0].n_calls
#             if disc[0].name == 'DesignVar':
#                 self.assertEqual(
#                     n_calls_2[(i, disc[0].name)], n_calls_1[(i, disc[0].name)])
#             else:
#                 self.assertEqual(
#                     n_calls_2[(i, disc[0].name)], n_calls_1[(i, disc[0].name)] + 1)
#
#         self.dir_to_del.append(self.dump_dir)


if '__main__' == __name__:
    cls = TestLoadSimpleCache()
    cls.setUp()
    cls.test_08_copy_cache_with_copy_study()
    cls.tearDown()
