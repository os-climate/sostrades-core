'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/02-2024/05/16 Copyright 2023 Capgemini

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
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from time import sleep

import numpy as np
from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestMDAPrerun(unittest.TestCase):
    """
    Class to test pre-run of MDAChain
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

    def test_01_mda_no_init_values(self):
        '''
        Must raise an exception since no disciplines have input filled
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
            [disc7_builder, disc6_builder])
        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.n_processes'] = 1
        exec_eng.load_study_from_input_dict(values_dict)

        with self.assertRaises(Exception) as cm:
            exec_eng.execute()
        error_message = 'The MDA cannot be pre-runned, some input values are missing to run the MDA'
        self.assertTrue(str(cm.exception).startswith(error_message))

    def test_02_mda_init_h(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc7_builder, disc6_builder])
        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = array([8., 9.])
        values_dict['EE.n_processes'] = 1
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
        residual_history = exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.sub_mda_list[0].residual_history
        # Check residual history
        tolerance = exec_eng.dm.get_value('EE.tolerance')
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1], tolerance)

        disc6 = exec_eng.dm.get_disciplines_with_name('EE.Disc6')[0]
        disc7 = exec_eng.dm.get_disciplines_with_name('EE.Disc7')[0]

        x_in = disc6.get_sosdisc_inputs('x')
        x_target = array([np.sqrt(2.) / 2., np.sqrt(2.) / 2.])
        x_out = disc7.get_sosdisc_outputs('x')
        x_dm = exec_eng.dm.get_value('EE.x')
        self.assertAlmostEqual(abs(x_dm[0]), x_target[0], delta=tolerance)
        self.assertAlmostEqual(abs(x_in[0]), x_target[0], delta=tolerance)
        self.assertAlmostEqual(abs(x_out[0]), x_target[0], delta=tolerance)

    def test_03_mda_init_x(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc7_builder, disc6_builder])
        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.x'] = array([8., 9.])
        values_dict['EE.n_processes'] = 1
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
        residual_history = exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.sub_mda_list[0].residual_history
        # Check residual history
        tolerance = exec_eng.dm.get_value('EE.tolerance')
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1], tolerance)

        disc6 = exec_eng.dm.get_disciplines_with_name('EE.Disc6')[0]
        disc7 = exec_eng.dm.get_disciplines_with_name('EE.Disc7')[0]

        x_in = disc6.get_sosdisc_inputs('x')
        x_target = array([np.sqrt(2.) / 2., np.sqrt(2.) / 2.])
        x_out = disc7.get_sosdisc_outputs('x')
        x_dm = exec_eng.dm.get_value('EE.x')
        self.assertAlmostEqual(x_dm[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_in[0], x_target[0], delta=tolerance)
        self.assertAlmostEqual(x_out[0], x_target[0], delta=tolerance)

    def test_04_prerun_coupling_of_coupling(self):
        '''
        Test the prerun when a coupling is given as output of a process and is build into the root process which is a coupling
        In this case we need to give the local_data of the high leevel coupling to the subcoupling
        Test to check that everything works

        '''
        exec_eng = ExecutionEngine(self.name)

        repo_test = 'sostrades_core.sos_processes.test'
        builder = exec_eng.factory.get_builder_from_process(repo=repo_test,
                                                            mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)
        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        coupling_name = "SellarCoupling"

        # additional test to verify that values_in are used
        disc_dict = {}
        # Sellar inputs
        disc_dict[f'{self.name}.{coupling_name}.x'] = array([1.])
        # disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        # disc_dict[f'{self.name}.{coupling_name}.y_2'] = array([1.])
        disc_dict[f'{self.name}.{coupling_name}.z'] = array([1., 1.])
        disc_dict[f'{self.name}.{coupling_name}.Sellar_Problem.local_dv'] = 10.
        disc_dict[f'{self.name}.{coupling_name}.max_mda_iter'] = 100
        disc_dict[f'{self.name}.{coupling_name}.tolerance'] = 1e-12
        exec_eng.load_study_from_input_dict(disc_dict)

        with self.assertRaises(Exception) as cm:
            exec_eng.execute()
        error_message = "The MDA cannot be pre-runned, some input values are missing to run the MDA " + \
                        "\nEE.SellarCoupling.Sellar_2 : ['EE.SellarCoupling.y_1']" + \
                        "\nEE.SellarCoupling.Sellar_1 : ['EE.SellarCoupling.y_2']"
        self.assertTrue(str(cm.exception).startswith(error_message))

        disc_dict[f'{self.name}.{coupling_name}.y_1'] = array([1.])
        exec_eng.load_study_from_input_dict(disc_dict)
        exec_eng.execute()


if '__main__' == __name__:
    cls = TestMDAPrerun()
    cls.setUp()
    cls.test_01_mda_no_init_values()
