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
from copy import deepcopy
from gemseo.utils.compare_data_manager_tooling import compare_dict, \
    delete_keys_from_dict
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine

import unittest
import pandas as pd
import numpy as np


class TestParallelExecution(unittest.TestCase):
    """
    Tests on parallel execution
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.c_name = "SellarCoupling"
        self.sc_name = "SellarOptimScenario"
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_coupling'
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [1., [5., 2.], 1., 1.],
                       'lower_bnd': [0., [-10., 0.], -100., -100.],
                       'upper_bnd': [10., [10., 10.], 100., 100.],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        self.dspace = pd.DataFrame(dspace_dict)

    def test_01_parallel_execution_NR_2procs(self):
        """
        1 proc
        """
        n_proc = 1
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng.load_study_from_input_dict(values_dict)

        mda = exec_eng.root_process.sos_disciplines[0]

        self.assertEqual(mda.n_processes, n_proc)
        exec_eng.execute()

        dm_dict_1 = deepcopy(exec_eng.get_anonimated_data_dict())
        """
        2 procs
        """
        n_proc = 2
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng2.load_study_from_input_dict(values_dict)

        mda2 = exec_eng2.root_process.sos_disciplines[0]

        self.assertEqual(mda2.n_processes, n_proc)
        exec_eng2.execute()
        dm_dict_2 = deepcopy(exec_eng2.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_2)
        compare_dict(dm_dict_1,
                     dm_dict_2, '', dict_error)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.SellarCoupling.n_processes.value': "1 and 2 don't match"})

        for disc1, disc2 in zip(exec_eng.root_process.sos_disciplines[0].sos_disciplines, exec_eng2.root_process.sos_disciplines[0].sos_disciplines):
            if disc1.jac is not None:
                self.assertDictEqual(disc1.jac, disc2.jac)

    def test_02_parallel_execution_NR_64procs(self):
        """
        1 proc
        """
        n_proc = 1
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng.load_study_from_input_dict(values_dict)

        mda = exec_eng.root_process.sos_disciplines[0]

        self.assertEqual(mda.n_processes, n_proc)
        exec_eng.execute()
        dm_dict_1 = deepcopy(exec_eng.get_anonimated_data_dict())
        """
        64 procs
        """
        n_proc = 64
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "MDANewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng2.load_study_from_input_dict(values_dict)

        mda2 = exec_eng2.root_process.sos_disciplines[0]

        self.assertEqual(mda2.n_processes, n_proc)
        exec_eng2.execute()
        dm_dict_2 = deepcopy(exec_eng2.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_2)
        compare_dict(dm_dict_1,
                     dm_dict_2, '', dict_error)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.SellarCoupling.n_processes.value': "1 and 64 don't match"})

        for disc1, disc2 in zip(exec_eng.root_process.sos_disciplines[0].sos_disciplines, exec_eng2.root_process.sos_disciplines[0].sos_disciplines):
            if disc1.jac is not None:
                self.assertDictEqual(disc1.jac, disc2.jac)

    def test_03_parallel_execution_PureNR_2procs(self):
        """
        1 proc
        """
        n_proc = 1
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "PureNewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng.load_study_from_input_dict(values_dict)

        mda = exec_eng.root_process.sos_disciplines[0]

        self.assertEqual(mda.n_processes, n_proc)
        exec_eng.execute()

        dm_dict_1 = deepcopy(exec_eng.get_anonimated_data_dict())
        """
        2 procs
        """
        n_proc = 2
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "PureNewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng2.load_study_from_input_dict(values_dict)

        mda2 = exec_eng2.root_process.sos_disciplines[0]

        self.assertEqual(mda2.n_processes, n_proc)
        exec_eng2.execute()
        dm_dict_2 = deepcopy(exec_eng2.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_2)
        compare_dict(dm_dict_1,
                     dm_dict_2, '', dict_error)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.SellarCoupling.n_processes.value': "1 and 2 don't match"})

        for disc1, disc2 in zip(exec_eng.root_process.sos_disciplines[0].sos_disciplines, exec_eng2.root_process.sos_disciplines[0].sos_disciplines):
            if disc1.jac is not None:
                self.assertDictEqual(disc1.jac, disc2.jac)

    def test_04_parallel_execution_pureNR_64procs(self):
        """
        1 proc
        """
        n_proc = 1
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "PureNewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng.load_study_from_input_dict(values_dict)

        mda = exec_eng.root_process.sos_disciplines[0]

        self.assertEqual(mda.n_processes, n_proc)
        exec_eng.execute()
        dm_dict_1 = deepcopy(exec_eng.get_anonimated_data_dict())
        """
        64 procs
        """
        n_proc = 64
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "PureNewtonRaphson"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng2.load_study_from_input_dict(values_dict)

        mda2 = exec_eng2.root_process.sos_disciplines[0]

        self.assertEqual(mda2.n_processes, n_proc)
        exec_eng2.execute()
        dm_dict_2 = deepcopy(exec_eng2.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_2)
        compare_dict(dm_dict_1,
                     dm_dict_2, '', dict_error)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.SellarCoupling.n_processes.value': "1 and 64 don't match"})

        for disc1, disc2 in zip(exec_eng.root_process.sos_disciplines[0].sos_disciplines, exec_eng2.root_process.sos_disciplines[0].sos_disciplines):
            if disc1.jac is not None:
                self.assertDictEqual(disc1.jac, disc2.jac)

    def test_05_parallel_execution_GSPureNR_2procs(self):
        """
        1 proc
        """
        n_proc = 1
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng.load_study_from_input_dict(values_dict)

        mda = exec_eng.root_process.sos_disciplines[0]

        self.assertEqual(mda.n_processes, n_proc)
        exec_eng.execute()

        dm_dict_1 = deepcopy(exec_eng.get_anonimated_data_dict())
        """
        2 procs
        """
        n_proc = 2
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id='test_sellar_coupling')

        exec_eng2.factory.set_builders_to_coupling_builder(builder)

        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = "GSPureNewtonMDA"
        values_dict[f'{self.ns}.{self.c_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = np.array([1., 1.])
        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.n_processes'] = n_proc
        exec_eng2.load_study_from_input_dict(values_dict)

        mda2 = exec_eng2.root_process.sos_disciplines[0]

        self.assertEqual(mda2.n_processes, n_proc)
        exec_eng2.execute()
        dm_dict_2 = deepcopy(exec_eng2.get_anonimated_data_dict())
        dict_error = {}
        # to delete modelorigin and discipline dependencies which are not the
        # same
        delete_keys_from_dict(dm_dict_1)
        delete_keys_from_dict(dm_dict_2)
        compare_dict(dm_dict_1,
                     dm_dict_2, '', dict_error)
        # The only different value is n_processes
        self.assertDictEqual(dict_error, {
                             '.<study_ph>.SellarCoupling.n_processes.value': "1 and 2 don't match"})

        for disc1, disc2 in zip(exec_eng.root_process.sos_disciplines[0].sos_disciplines, exec_eng2.root_process.sos_disciplines[0].sos_disciplines):
            if disc1.jac is not None:
                self.assertDictEqual(disc1.jac, disc2.jac)


if '__main__' == __name__:

    cls = TestParallelExecution()
    cls.setUp()
    cls.test_01_parallel_execution_NR_2procs()
