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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
"""

import unittest
import pandas as pd
from numpy import array
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine

from os import remove


class TestMDANumericalParameters(unittest.TestCase):
    """
    Class to test MDA numerical parameters
    """

    def setUp(self):
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.c_name = "SellarCoupling"
        self.sc_name = "SellarOptimScenario"
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_coupling'
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [1., [5., 2.], 1., 1.],
                       'lower_bnd': [0., [-10., 0.], -100., -100.],
                       'upper_bnd': [10., [10., 10.], 100., 100.]}
        self.dspace = pd.DataFrame(dspace_dict)

        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1.Disc1'

    def tearDown(self):
        try:
            remove('.\cache.h5')
        except OSError:
            pass

    def test_01_optim_scenario_with_warm_start_for_mda(self):
        print("\n Test 1 : check the influence of warm_start option")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        #-- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [1., [5., 2.]],
                       'lower_bnd': [0., [-10., 0.]],
                       'upper_bnd': [10., [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        #-- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-7,
                                                                    "ineq_tolerance": 1e-7,
                                                                    "normalize_design_space": False}

        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                                                                         1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()
        with self.assertLogs('SoS.EE.Coupling', level='INFO') as cm:
            exec_eng.execute()
            nb_ite_wo_warm_start = len(cm.output)

        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=proc_name_discopt)

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        exec_eng2.dm.set_values_from_dict(disc_dict)

        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.warm_start'] = True

        exec_eng2.dm.set_values_from_dict(values_dict)

        exec_eng2.configure()
        with self.assertLogs('SoS.EE.Coupling', level='INFO') as cm:
            exec_eng2.execute()
            nb_ite_warm_start = len(cm.output)

        # Check that we have less iterations
        self.assertLessEqual(nb_ite_warm_start, nb_ite_wo_warm_start)

        # check that optim is not impacted by warm_start
        opt_disc = exec_eng2.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        for x, x_th in zip(opt_disc.optimization_result.x_opt, exp_x):
            self.assertAlmostEqual(x, x_th, delta=1.0e-4,
                                   msg="Wrong optimal x solution")

    def test_02_chech_cache_option(self):
        print("\n Test 2 : check cache_option")

        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee.ns_manager.add_ns('ns_ac', self.name)
        self.ee.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b,
                       self.name + '.Disc1.cache_type': 'SimpleCache'}

        self.ee.load_study_from_input_dict(values_dict)

        self.ee.display_treeview_nodes()
        self.ee.execute()

        values_dict = {self.name + '.x': 3.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()

        values_dict = {self.name + '.x': 1.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()

        values_dict = {self.name + '.x': 1.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee.dm.set_values_from_dict(values_dict)
        self.ee.execute()

        n_calls_simple_cache = self.ee.root_process.disciplines[0].n_calls

        self.assertEqual(n_calls_simple_cache, 3)
        # Now with memory full cache
        self.ee2 = ExecutionEngine(self.name)

        disc1_builder = self.ee2.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee2.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee2.ns_manager.add_ns('ns_ac', self.name)
        self.ee2.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b,
                       self.name + '.Disc1.cache_type': 'MemoryFullCache'}

        self.ee2.load_study_from_input_dict(values_dict)

        self.ee2.display_treeview_nodes()
        self.ee2.execute()

        values_dict = {self.name + '.x': 3.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee2.dm.set_values_from_dict(values_dict)
        self.ee2.execute()

        values_dict = {self.name + '.x': 1.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee2.dm.set_values_from_dict(values_dict)
        self.ee2.execute()

        n_calls_full_memory_cache = self.ee2.root_process.disciplines[0].n_calls
        self.assertEqual(n_calls_full_memory_cache, 2)

        self.ee3 = ExecutionEngine(self.name)

        disc1_builder = self.ee3.factory.get_builder_from_module(
            'Disc1', self.mod1_path)
        self.ee3.factory.set_builders_to_coupling_builder(disc1_builder)

        self.ee3.ns_manager.add_ns('ns_ac', self.name)
        self.ee3.configure()
        a = 1.0
        b = 2.0
        x = 1.0
        values_dict = {self.name + '.x': x,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b,
                       self.name + '.Disc1.cache_type': 'HDF5Cache'}

        with self.assertRaises(Exception) as cm:
            self.ee3.load_study_from_input_dict(values_dict)
        error_message = 'if the cache type is set to HDF5Cache the cache_file path must be set'
        self.assertTrue(str(cm.exception) == error_message)

        values_dict[self.name + '.Disc1.cache_file_path'] = 'cache.h5'
        self.ee3.load_study_from_input_dict(values_dict)
        self.ee3.execute()

        values_dict = {self.name + '.x': 3.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee3.dm.set_values_from_dict(values_dict)
        self.ee3.execute()

        values_dict = {self.name + '.x': 1.0,
                       self.name + '.Disc1.a': a,
                       self.name + '.Disc1.b': b}

        self.ee3.dm.set_values_from_dict(values_dict)
        self.ee3.execute()

        n_calls_hdf5_cache = self.ee3.root_process.disciplines[0].n_calls
        self.assertEqual(n_calls_hdf5_cache, 2)


if __name__ == "__main__":
    cls = TestMDANumericalParameters()
    cls.setUp()
    cls.test_02_chech_cache_option()
