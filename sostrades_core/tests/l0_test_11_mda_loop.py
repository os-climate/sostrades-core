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
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

import numpy as np
from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager


class TestMDALoop(unittest.TestCase):
    """
    MDA test class
    """

    def setUp(self):
        self.dirs_to_del = []
        self.name = 'EE'
        self.root_dir = gettempdir()

    def tearDown(self):
        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_mda_loop(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.n_processes'] = 1
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        target = {'EE.h': array([0.70710678,
                                 0.70710678]),
                  'EE.x': array([0., 0.707107, 0.707107])}

        # check output keys
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))
        max_mda_iter = exec_eng.dm.get_value('EE.max_mda_iter')
        residual_history = exec_eng.root_mdo_discipline.sub_mda_list[0].residual_history

        # check residual history
        tolerance = exec_eng.dm.get_value('EE.tolerance')
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1][0], tolerance)

        disc6 = exec_eng.root_process.proxy_disciplines[0]
        disc7 = exec_eng.root_process.proxy_disciplines[1]

        x_in = disc6.get_sosdisc_inputs('x')
        x_target = array([np.sqrt(2.) / 2., np.sqrt(2.) / 2.])
        x_out = disc7.get_sosdisc_outputs('x')
        x_dm = exec_eng.dm.get_value('EE.x')
        self.assertAlmostEqual(x_dm[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_in[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_out[0], x_target[0], delta=tolerance)

    def test_02_mda_numerical_options(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        ''' Default values for numerical options
        'max_mda_iter': 200.,
        'n_processes': max_cpu,
        'chain_linearize': False,
        'tolerance': 1.e-6,
        'use_lu_fact': False
        '''

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-8
        values_dict['EE.chain_linearize'] = True
        values_dict['EE.n_processes'] = 2
        values_dict['EE.max_mda_iter'] = 50
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        self.assertEqual(values_dict['EE.use_lu_fact'], mda.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'], mda.tolerance)

        self.assertEqual(values_dict['EE.n_processes'], mda.n_processes)
        self.assertEqual(values_dict['EE.max_mda_iter'], mda.max_mda_iter)
        exec_eng.execute()

        self.assertEqual(values_dict['EE.use_lu_fact'], mda.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'], mda.tolerance)

        self.assertEqual(values_dict['EE.n_processes'], mda.n_processes)
        self.assertEqual(values_dict['EE.max_mda_iter'], mda.max_mda_iter)

        target = {'EE.h': array([0.70710678,
                                 0.70710678]),
                  'EE.x': array([0., 0.707107, 0.707107])}

        # check output keys
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))
                self.assertListEqual(
                    exec_eng.root_mdo_discipline.local_data[key], target[key])

#     def test_03_mda_loop_after_dump_load(self):
# 
#         exec_eng = ExecutionEngine(self.name)
# 
#         exec_eng.ns_manager.add_ns('ns_protected', self.name)
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
#         disc7_builder = exec_eng.factory.get_builder_from_module(
#             'Disc7', mod_list)
# 
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
#         disc6_builder = exec_eng.factory.get_builder_from_module(
#             'Disc6', mod_list)
# 
#         exec_eng.factory.set_builders_to_coupling_builder(
#             [disc6_builder, disc7_builder])
#         exec_eng.configure()
#         # additional test to verify that values_in are used
#         values_dict = {}
#         values_dict['EE.h'] = array([8., 9.])
#         values_dict['EE.x'] = array([5., 3.])
#         values_dict['EE.n_processes'] = 1
#         exec_eng.load_study_from_input_dict(values_dict)
# 
#         target = {'EE.h': array([0.70710678,
#                                  0.70710678]),
#                   'EE.x': array([0., 0.707107, 0.707107])}
#         res = {}
#         for key in target:
#             res[key] = exec_eng.dm.get_value(key)
#             print('exec_1 before exe', key, res[key])
#         exec_eng.execute()
#         norm0 = exec_eng.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residual after first exe', norm0, normed_residual)
#         exec_eng.execute()
#         norm0 = exec_eng.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residualafter second exe', norm0, normed_residual)
# 
#         # check output keys
#         res = {}
#         for key in target:
#             res[key] = exec_eng.dm.get_value(key)
#             print('exec_1', key, res[key])
#             if target[key] is dict:
#                 self.assertDictEqual(res[key], target[key])
#             elif target[key] is array:
#                 self.assertListEqual(
#                     list(target[key]), list(res[key]))
# 
#         residual_history = exec_eng.root_mdo_discipline.sub_mda_list[0].residual_history
#         residual_history_output = exec_eng.dm.get_disciplines_with_name('EE')[0].get_sosdisc_outputs(
#             'residuals_history')[exec_eng.root_mdo_discipline.sub_mda_list[0].name].values.tolist()
#         self.assertEqual(residual_history, residual_history_output)
# 
#         dump_dir = join(self.root_dir, self.name)
# 
#         BaseStudyManager.static_dump_data(
#             dump_dir, exec_eng, DirectLoadDump())
# 
#         exec_eng2 = ExecutionEngine(self.name)
# 
#         exec_eng2.ns_manager.add_ns('ns_protected', self.name)
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
#         disc7_builder = exec_eng2.factory.get_builder_from_module(
#             'Disc7', mod_list)
# 
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
#         disc6_builder = exec_eng2.factory.get_builder_from_module(
#             'Disc6', mod_list)
# 
#         exec_eng2.factory.set_builders_to_coupling_builder(
#             [disc6_builder, disc7_builder])
#         exec_eng2.configure()
# 
#         BaseStudyManager.static_load_data(
#             dump_dir, exec_eng2, DirectLoadDump())
# 
#         res = {}
#         for key in target:
#             res[key] = exec_eng2.dm.get_value(key)
#             print('exec_2 before exe', key, res[key])
#         norm0 = exec_eng2.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng2.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residual before third exe', norm0, normed_residual)
#         exec_eng2.execute()
#         norm0 = exec_eng2.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng2.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residual after third exe', norm0, normed_residual)
#         res = {}
#         for key in target:
#             res[key] = exec_eng2.dm.get_value(key)
#             print('exec_2', key, res[key])
#             if target[key] is dict:
#                 self.assertDictEqual(res[key], target[key])
#             elif target[key] is array:
#                 self.assertListEqual(
#                     list(target[key]), list(res[key]))
#         residual_history_2 = exec_eng2.root_mdo_discipline.sub_mda_list[0].residual_history
#         residual_history_output_2 = exec_eng2.dm.get_disciplines_with_name('EE')[0].get_sosdisc_outputs(
#             'residuals_history')[exec_eng2.root_mdo_discipline.sub_mda_list[0].name].values.tolist()
#         self.assertEqual(residual_history_2, residual_history_output_2)
# 
#         BaseStudyManager.static_dump_data(
#             dump_dir, exec_eng2, DirectLoadDump())
# 
#         exec_eng3 = ExecutionEngine(self.name)
# 
#         exec_eng3.ns_manager.add_ns('ns_protected', self.name)
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
#         disc7_builder = exec_eng3.factory.get_builder_from_module(
#             'Disc7', mod_list)
# 
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
#         disc6_builder = exec_eng3.factory.get_builder_from_module(
#             'Disc6', mod_list)
# 
#         exec_eng3.factory.set_builders_to_coupling_builder(
#             [disc6_builder, disc7_builder])
#         exec_eng3.configure()
# 
#         BaseStudyManager.static_load_data(
#             dump_dir, exec_eng3, DirectLoadDump())
# 
#         res = {}
#         for key in target:
#             res[key] = exec_eng3.dm.get_value(key)
#             print('exec_3 before exe', key, res[key])
#         norm0 = exec_eng3.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng3.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residual before exe 4', norm0, normed_residual)
#         exec_eng3.execute()
#         norm0 = exec_eng3.root_mdo_discipline.sub_mda_list[0].norm0
#         normed_residual = exec_eng3.root_mdo_discipline.sub_mda_list[0].normed_residual
#         print('norm0 and norm_residual after exe 4', norm0, normed_residual)
#         res = {}
#         for key in target:
#             res[key] = exec_eng3.dm.get_value(key)
#             print('exec_3', key, res[key])
#             if target[key] is dict:
#                 self.assertDictEqual(res[key], target[key])
#             elif target[key] is array:
#                 self.assertListEqual(
#                     list(target[key]), list(res[key]))
#         residual_history_3 = exec_eng3.root_mdo_discipline.sub_mda_list[0].residual_history
#         residual_history_output_3 = exec_eng3.dm.get_disciplines_with_name('EE')[0].get_sosdisc_outputs(
#             'residuals_history')[exec_eng3.root_mdo_discipline.sub_mda_list[0].name].values.tolist()
#         self.assertEqual(residual_history_3, residual_history_output_3)
# 
#         self.assertEqual(residual_history_3[-1][0], residual_history_2[-1][0])
# 
#         BaseStudyManager.static_dump_data(
#             dump_dir, exec_eng3, DirectLoadDump())
# 
#         exec_eng4 = ExecutionEngine(self.name)
# 
#         exec_eng4.ns_manager.add_ns('ns_protected', self.name)
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
#         disc7_builder = exec_eng4.factory.get_builder_from_module(
#             'Disc7', mod_list)
# 
#         mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
#         disc6_builder = exec_eng4.factory.get_builder_from_module(
#             'Disc6', mod_list)
# 
#         exec_eng4.factory.set_builders_to_coupling_builder(
#             [disc6_builder, disc7_builder])
#         exec_eng4.configure()
# 
#         BaseStudyManager.static_load_data(
#             dump_dir, exec_eng4, DirectLoadDump())
# 
#         res = {}
#         for key in target:
#             res[key] = exec_eng4.dm.get_value(key)
#             print('exec_4 before exe', key, res[key])
# 
#         exec_eng4.execute()
#         residual_history_4 = exec_eng4.root_mdo_discipline.sub_mda_list[0].residual_history
#         residual_history_output_4 = exec_eng4.dm.get_disciplines_with_name('EE')[0].get_sosdisc_outputs(
#             'residuals_history')[exec_eng4.root_mdo_discipline.sub_mda_list[0].name].values.tolist()
#         self.assertEqual(residual_history_4, residual_history_output_4)
#         res = {}
#         for key in target:
#             res[key] = exec_eng4.dm.get_value(key)
#             print('exec_4', key, res[key])
#             if target[key] is dict:
#                 self.assertDictEqual(res[key], target[key])
#             elif target[key] is array:
#                 self.assertListEqual(
#                     list(target[key]), list(res[key]))
#         # Clean the dump folder at the end of the test
#         self.dirs_to_del.append(
#             join(self.root_dir, self.name))

    def test_04_two_mdas_loop_comparison(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.n_processes'] = 1
        exec_eng.load_study_from_input_dict(values_dict)

        target = {'EE.h': array([0.70710678,
                                 0.70710678]),
                  'EE.x': array([0., 0.707107, 0.707107])}

        exec_eng.execute()

        # -- check output keys
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            print('exec_1', key, res[key])
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))
        residual_history = exec_eng.root_mdo_discipline.sub_mda_list[0].residual_history

        exec_eng2 = ExecutionEngine(self.name)

        exec_eng2.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng2.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng2.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng2.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng2.configure()

        exec_eng2.load_study_from_input_dict(values_dict)

        exec_eng2.execute()

        # -- check output keys
        res = {}
        for key in target:
            res[key] = exec_eng2.dm.get_value(key)
            print('exec_1', key, res[key])
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))

        residual_history2 = exec_eng2.root_mdo_discipline.sub_mda_list[0].residual_history

        self.assertListEqual(residual_history, residual_history2)

    def test_05_mda_loop_with_string_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)
        disc7_builder.cls.DESC_OUT['string_dict'] = {
            'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_protected'}
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)
        disc6_builder.cls.DESC_IN['string_dict'] = {
            'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_protected'}
        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.string_dict'] = {'key0': 'toto'}
        values_dict['EE.n_processes'] = 1
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        target = {'EE.h': array([0.70710678,
                                 0.70710678]),
                  'EE.x': array([0., 0.707107, 0.707107]),
                  'EE.string_dict': {'key0': 'toto'}}
        # -- check output keys
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))
        max_mda_iter = exec_eng.dm.get_value('EE.max_mda_iter')
        residual_history = exec_eng.root_mdo_discipline.sub_mda_list[0].residual_history
        print('residual_history', residual_history)
        # Check residual history
        tolerance = exec_eng.dm.get_value('EE.tolerance')
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1][0], tolerance)

        disc6 = exec_eng.root_process.proxy_disciplines[0]
        disc7 = exec_eng.root_process.proxy_disciplines[1]

        x_in = disc6.get_sosdisc_inputs('x')
        x_target = array([np.sqrt(2.) / 2., np.sqrt(2.) / 2.])
        x_out = disc7.get_sosdisc_outputs('x')
        x_dm = exec_eng.dm.get_value('EE.x')
        self.assertAlmostEqual(x_dm[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_in[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_out[0], x_target[0], delta=tolerance)

        disc6_builder.cls.DESC_IN.pop('string_dict')
        disc7_builder.cls.DESC_OUT.pop('string_dict')

    def _test_06_mda_loop_with_discipline_grouping(self):
        '''
        Test temporary commented because the MDOChain built under sub mda can not access conversion method (in not instance of SoSDiscipline)
        '''

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])

        exec_eng.configure()

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.n_processes'] = 1
        values_dict['EE.group_mda_disciplines'] = True
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        target = {'EE.h': array([0.70710678,
                                 0.70710678]),
                  'EE.x': array([0., 0.707107, 0.707107])}
        # -- check output keys
        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))
        max_mda_iter = exec_eng.dm.get_value('EE.max_mda_iter')
        residual_history = exec_eng.root_mdo_discipline.sub_mda_list[0].residual_history
        print('residual_history', residual_history)
        # Check residual history
        tolerance = exec_eng.dm.get_value('EE.tolerance')
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1][0], tolerance)

        disc6 = exec_eng.root_process.proxy_disciplines[0]
        disc7 = exec_eng.root_process.proxy_disciplines[1]

        x_in = disc6.get_sosdisc_inputs('x')
        x_target = array([np.sqrt(2.) / 2., np.sqrt(2.) / 2.])
        x_out = disc7.get_sosdisc_outputs('x')
        x_dm = exec_eng.dm.get_value('EE.x')
        self.assertAlmostEqual(x_dm[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_in[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_out[0], x_target[0], delta=tolerance)

        # - check that the group option has been taken into account

        # we check that in the MDA into the MDOChain
        # there is the MDOChain (ie group of subdiscs) instead of 2 disciplines
        mdo_chain = exec_eng.root_mdo_discipline.mdo_chain.disciplines[0].disciplines[0]
        assert mdo_chain.__class__.__name__ == "MDOChain"

        option = exec_eng.root_process.get_sosdisc_inputs(
            "group_mda_disciplines")
        assert option == True

    def test_07_check_no_self_coupled_soscoupling(self):

        exec_eng = ExecutionEngine(self.name)

        # add disciplines Sellaroupling
        coupling_name = "SellarCoupling"
        mda_builder = exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling')
        exec_eng.factory.set_builders_to_coupling_builder(mda_builder)
        exec_eng.configure()

        # Sellar inputs
        disc_dict = {}
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.

        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.execute()

        # we check that in the root coupling, the subcoupling is NOT a (selfcoupled) MDA with an SoSCoupling inside
        # but a SoSCoupling directly
        sub_coupling = exec_eng.root_mdo_discipline.mdo_chain.disciplines[0]
        assert sub_coupling.__class__.__name__ == "SoSMDAChain"

    def test_08_mda_numerical_options_NR(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 2000,
            'tol': 1.0e-10}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]

        self.assertEqual(values_dict['EE.use_lu_fact'],
                         sub_mda_class.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.n_processes'],
                         sub_mda_class.n_processes)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        values_dict = {}
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.n_processes'] = 2
        values_dict['EE.max_mda_iter'] = 100
        values_dict['EE.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.use_lu_fact'],
                         sub_mda_class.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.n_processes'],
                         sub_mda_class.n_processes)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

#         exec_eng.execute()

    def test_09_mda_numerical_options_GS(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'MDAGaussSeidel'

        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 100
        values_dict['EE.sub_mda_class'] = 'MDAGaussSeidel'

        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]

        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)

    def test_10_mda_numerical_options_GSNR(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        ''' Default values for numerical options
        'max_mda_iter': 200.,
        'n_processes': max_cpu,
        'chain_linearize': False,
        'tolerance': 1.e-6,
        'use_lu_fact': False
        '''

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.tolerance_gs'] = 1.0
        values_dict['EE.sub_mda_class'] = 'GSNewtonMDA'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.max_mda_iter'] = 150
        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(values_dict['EE.tolerance_gs'],
                         sub_mda_class.mda_sequence[0].tolerance)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[1].tolerance)

        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.mda_sequence[1].max_mda_iter)

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        NR = sub_mda_class.mda_sequence[1]

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         NR.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         NR.linear_solver_options['max_iter'])

    def test_11_mda_numerical_options_GSorNR(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        ''' Default values for numerical options
        'max_mda_iter': 200.,
        'n_processes': max_cpu,
        'chain_linearize': False,
        'tolerance': 1.e-6,
        'use_lu_fact': False
        '''

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'GSorNewtonMDA'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.max_mda_iter'] = 150
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[0].tolerance)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[1].tolerance)

        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.mda_sequence[1].max_mda_iter)

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        NR = sub_mda_class.mda_sequence[1]

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         NR.linear_solver_options['max_iter'])

    def test_14_mda_numerical_options_GSPureNR(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        ''' Default values for numerical options
        'max_mda_iter': 200.,
        'n_processes': max_cpu,
        'chain_linearize': False,
        'tolerance': 1.e-6,
        'use_lu_fact': False
        '''

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'GSPureNewtonMDA'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.max_mda_iter'] = 150
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(10.0,
                         sub_mda_class.mda_sequence[0].tolerance)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[1].tolerance)

        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.mda_sequence[1].max_mda_iter)

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        NR = sub_mda_class.mda_sequence[1]

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         NR.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         NR.linear_solver_options['max_iter'])

    def test_15_mda_numerical_options_PureNR(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.epsilon0'] = 1.0
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'PureNewtonRaphson'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 2000,
            'tol': 1.0e-10}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]

        self.assertEqual(values_dict['EE.use_lu_fact'],
                         sub_mda_class.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        values_dict = {}
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-13

        values_dict['EE.max_mda_iter'] = 100
        values_dict['EE.sub_mda_class'] = 'PureNewtonRaphson'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)
        
        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.use_lu_fact'],
                         sub_mda_class.use_lu_fact)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        import tracemalloc
        tracemalloc.start()
# 
#         exec_eng.execute()
#         current, peak = tracemalloc.get_traced_memory()
#         print(
#             f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
# 
#         tracemalloc.stop()

    def test_16_mda_numerical_options_GSPureNRorGSMDA(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        ''' Default values for numerical options
        'max_mda_iter': 200.,
        'n_processes': max_cpu,
        'chain_linearize': False,
        'tolerance': 1.e-6,
        'use_lu_fact': False
        '''

        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.x'] = array([5., 3.])
        values_dict['EE.use_lu_fact'] = True
        values_dict['EE.tolerance'] = 1.e-15
        values_dict['EE.n_processes'] = 4
        values_dict['EE.max_mda_iter'] = 50
        values_dict['EE.sub_mda_class'] = 'GSPureNewtonorGSMDA'
        values_dict['EE.linear_solver_MDA_options'] = {
            'max_iter': 5000,
            'tol': 1e-15}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        values_dict['EE.tolerance'] = 1.e-20
        values_dict['EE.max_mda_iter'] = 150
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        mda = exec_eng.root_mdo_discipline

        sub_mda_class = mda.sub_mda_list[0]
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.tolerance)
        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.max_mda_iter)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[0].tolerance)
        self.assertEqual(values_dict['EE.tolerance'],
                         sub_mda_class.mda_sequence[1].tolerance)

        self.assertEqual(values_dict['EE.max_mda_iter'],
                         sub_mda_class.mda_sequence[1].max_mda_iter)

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['tol'],
                         sub_mda_class.linear_solver_tolerance)
        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         sub_mda_class.linear_solver_options['max_iter'])

        NR = sub_mda_class.mda_sequence[1]

        self.assertEqual(values_dict['EE.linear_solver_MDA_options']['max_iter'],
                         NR.linear_solver_options['max_iter'])


if '__main__' == __name__:
    cls = TestMDALoop()
    cls.setUp()
    cls.test_06_mda_loop_with_discipline_grouping()
