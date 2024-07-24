'''
Copyright 2022 Airbus SAS
Modifications on 2023/01/24-2024/05/16 Copyright 2023 Capgemini

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
import os
import unittest
from copy import deepcopy

import pandas as pd
from gemseo.core.mdo_scenario import MDOScenario
from numpy import array, set_printoptions
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.test.test_sellar_opt_discopt.usecase import (
    Study as study_sellar_opt_discopt,
)
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for optimization scenario
"""


class TestSoSOptimScenario(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarOptimScenario"
        self.c_name = "SellarCoupling"

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        self.dspace = pd.DataFrame(dspace_dict)
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_discopt'

    def _test_01_optim_scenario_check_treeview(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-10,
                                                                    "ineq_tolerance": 2e-3,
                                                                    "normalize_design_space": False}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        # XDSMize test

    #         exec_eng.root_process.xdsmize()
    # to visualize in an internet browser :
    # - download XDSMjs at https://github.com/OneraHub/XDSMjs and unzip
    # - replace existing xdsm.json inside by yours
    # - in the same folder, type in terminal 'python -m http.server 8080'
    # - open in browser http://localhost:8080/xdsm.html

    def _test_02_optim_scenario_execution_mdf(self):
        '''
        TEST COMMENTED BECAUSE MDF FORMULATION BUILD A MDACHAIN INSTEAD OF SOSCOUPLING
        '''
        print("\n Test 2 : Sellar optim solution check with MDF formulation")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_MMA"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-5,
                                                                    "ineq_tolerance": 1e-5,
                                                                    "normalize_design_space": False}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339395 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, 4, msg="Wrong objective value")
        exp_x = array([8.45997174e-15, 1.97763888, 0.0])
        assert_array_almost_equal(
            exp_x, opt_disc.optimization_result.x_opt, decimal=4, err_msg="Wrong optimal x solution")

    def _test_03_optim_scenario_execution_idf(self):
        print("\n Test 3 : Sellar optim solution check with IDF formulation")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'IDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       '\t\t|_ Sellar_Problem',
                       '\t\t|_ Sellar_2',
                       '\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.1800 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
        exp_x = array([1.6653e-16, 2.1339, 0., 3.16, 3.911598])
        assert_array_almost_equal(
            exp_x, opt_disc.optimization_result.x_opt, decimal=4, err_msg="Wrong optimal x solution")

    def test_04_optim_scenario_execution_disciplinaryopt(self):
        print("\n Test 4 : Sellar optim solution check with DisciplinaryOpt formulation")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.f_opt, places=4, msg="Wrong objective value")
        exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
        assert_array_almost_equal(
            exp_x, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.x_opt, decimal=4,
            err_msg="Wrong optimal x solution")

    def _test_05_optim_scenario_execution_disciplinaryopt_complex_step(self):
        print("\n Test 5 : Sellar optim solution check with DisciplinaryOpt formulation  with complex step")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = MDOScenario.COMPLEX_STEP
        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.f_opt, places=4, msg="Wrong objective value")
        exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
        assert_array_almost_equal(
            exp_x, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.x_opt, decimal=4, err_msg="Wrongoptimal x solution")

    def test_05b_optim_scenario_execution_disciplinaryopt_analyticgrad(self):
        print("\n Test 5 : Sellar optim solution check with DisciplinaryOpt formulation  with user defined gradients")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = 'user'
        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        # check optimal x and f
        sellar_obj_opt = 3.18339 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.f_opt, places=4, msg="Wrong objective value")
        exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
        assert_array_almost_equal(
            exp_x, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.x_opt, decimal=4, err_msg="Wrongoptimal x solution")

    def _test_06_optim_scenario_execution_fd_parallel(self):
        if os.name == 'nt':
            print("\n Test 6 : skipped, multi-proc not handled on windows")
        else:
            print("\n Test 6 : Sellar optim with FD in parallel execution")
            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            repo_discopt = 'sostrades_core.sos_processes.test'
            proc_name_discopt = 'test_sellar_opt_discopt'
            builder = factory.get_builder_from_process(repo=repo_discopt,
                                                       mod_id=proc_name_discopt)

            exec_eng.factory.set_builders_to_coupling_builder(builder)

            exec_eng.configure()

            dspace_dict = {'variable': ['x', 'z'],
                           'value': [[1.], [5., 2.]],
                           'lower_bnd': [[0.], [-10., 0.]],
                           'upper_bnd': [[10.], [10., 10.]],
                           'enable_variable': [True, True],
                           'activated_elem': [[True], [True, True]]}
            dspace = pd.DataFrame(dspace_dict)

            # -- set up disciplines in Scenario
            disc_dict = {}
            # Optim inputs
            disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
            disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
            disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
            disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
            disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
            disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
                'c_1', 'c_2']

            disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                        "ineq_tolerance": 1e-6,
                                                                        "normalize_design_space": True}
            # parallel inputs
            disc_dict[f'{self.ns}.SellarOptimScenario.parallel_options'] = {"parallel": True,
                                                                            "n_processes": 2,
                                                                            "use_threading": False,
                                                                            "wait_time_between_fork": 0}

            exec_eng.dm.set_values_from_dict(disc_dict)

            # Sellar inputs
            local_dv = 10.
            values_dict = {}
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([
                                                                             1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                               1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                               1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
                1., 1.])
            values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
            exec_eng.dm.set_values_from_dict(values_dict)

            exec_eng.configure()

            exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                           '|_ optim',
                           f'\t|_ {self.sc_name}',
                           f'\t\t|_ {self.c_name}',
                           '\t\t\t|_ Sellar_Problem',
                           '\t\t\t|_ Sellar_2',
                           '\t\t\t|_ Sellar_1', ]
            exp_tv_str = '\n'.join(exp_tv_list)
            exec_eng.display_treeview_nodes(True)
            assert exp_tv_str == exec_eng.display_treeview_nodes()

            res = exec_eng.execute()

            # retrieve discipline to check the result...
            opt_disc = exec_eng.dm.get_disciplines_with_name(
                "optim." + self.sc_name)[0]

            # check optimal x and f
            sellar_obj_opt = 3.18339 + local_dv
            self.assertAlmostEqual(
                sellar_obj_opt, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.f_opt, places=4, msg="Wrong objective value")
            exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
            assert_array_almost_equal(
                exp_x, opt_disc.mdo_discipline_wrapp.mdo_discipline.optimization_result.x_opt, decimal=4, err_msg="Wrong optimal x solution")

    def test_07_test_options(self):
        print("\n Test 07 : Sellar optim solution check options")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
            1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
            1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.prepare_execution()

        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0]

        algo_options = opt_disc.get_sosdisc_inputs('algo_options')

        assert ("maxcor" in algo_options.keys())
        assert ("max_ls_step_nb" in algo_options.keys())

    def test_08_optim_scenario_eval_mode(self):
        print("\n Test 8 : Sellar optim with eval_mode")
        set_printoptions(precision=20)
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[2.], [2., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.eval_mode'] = True
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            2., 2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.tolerance'] = 1e-16
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()

        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarOptimScenario.algo_options', 'editable'))
        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarOptimScenario.algo', 'editable'))

        exec_eng.execute()

        # Check that the jacobian has not been executed
        self.assertEqual(
            exec_eng.root_process.proxy_disciplines[0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.jac, None)
        # Exec_eng with only the coupling
        exec_eng2 = ExecutionEngine(self.study_name)
        factory = exec_eng2.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id='test_sellar_coupling')

        factory.set_builders_to_coupling_builder(builder)
        exec_eng2.configure()

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_1'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.y_2'] = array([2.])
        values_dict[f'{self.ns}.{self.c_name}.z'] = array([
            2., 2.])

        values_dict[f'{self.ns}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        values_dict[f'{self.ns}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.c_name}.tolerance'] = 1e-16

        exec_eng2.load_study_from_input_dict(values_dict)
        exec_eng2.prepare_execution()
        exec_eng2.execute()

        for var in ['x', 'y_1', 'y_2', 'z', 'obj', 'c_1', 'c_2']:

            eval_value = exec_eng.dm.get_value(
                f'{self.ns}.{self.sc_name}.{self.c_name}.{var}')
            coupling_value = exec_eng2.dm.get_value(
                f'{self.ns}.{self.c_name}.{var}')
            try:
                self.assertEqual(coupling_value, eval_value)
            except:
                self.assertListEqual(list(coupling_value), list(eval_value))

    def test_08b_optim_scenario_eval_mode_no_post_proc(self):
        print("\n Test 8b : Sellar optim with eval_mode with no output design space post proc")
        set_printoptions(precision=20)
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[2.], [2., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}

        # desactivate the optim post processings
        disc_dict[f'{self.ns}.SellarOptimScenario.desactivate_optim_out_storage'] = True

        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.eval_mode'] = True
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            2., 2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.tolerance'] = 1e-16
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()

        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarOptimScenario.algo_options', 'editable'))
        self.assertFalse(exec_eng.dm.get_data(
            f'{self.ns}.SellarOptimScenario.algo', 'editable'))

        exec_eng.execute()

        # check that the post processings can be executed
        from sostrades_core.tools.post_processing.post_processing_factory import (
            PostProcessingFactory,
        )
        ppf = PostProcessingFactory()
        for disc in exec_eng.root_process.proxy_disciplines:
            filters = ppf.get_post_processing_filters_by_discipline(
                disc)
            graph_list = ppf.get_post_processing_by_discipline(
                disc, filters, as_json=False)

        assert KeyError(exec_eng.dm.get_value("optim.SellarOptimScenario.post_processing_mdo_data"))

    def test_09_optim_scenario_eval_mode_with_eval_jac(self):
        print("\n Test 9 : Sellar optim with eval_mode and eval_jac")

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[2.], [2., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.eval_mode'] = True
        disc_dict[f'{self.ns}.SellarOptimScenario.eval_jac'] = True
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.sub_mda_class'] = 'MDANewtonRaphson'
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           2.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            2., 2.])

        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.prepare_execution()
        exec_eng.execute()

        # Get the jacobian of each functions (constraints + objective)
        computed_jac = exec_eng.root_process.proxy_disciplines[
            0].proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline.jac

        self.assertListEqual(sorted(list(computed_jac.keys())), sorted([
            f'{self.ns}.{self.sc_name}.{self.c_name}.{var}' for var in ['obj', 'c_1', 'c_2']]))

    def _test_10_update_dspace(self):
        '''
        TEST COMMENTED BECAUSE MDF FORMULATION BUILD A MDACHAIN INSTEAD OF SOSCOUPLING
        '''
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 12.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-10,
                                                                    "ineq_tolerance": 2e-3,
                                                                    "normalize_design_space": False}
        # Sellar inputs
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = 10.

        values_dict.update(disc_dict)

        try:
            exec_eng.load_study_from_input_dict(values_dict)
        except:
            pass

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 5.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        dspace = pd.DataFrame(dspace_dict)

        values_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace

        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.execute()

    def test_11_update_dspace_from_usecase(self):

        uc_cls = study_sellar_opt_discopt()
        uc_cls.setup_usecase()
        uc_cls.load_data()

        dspace = deepcopy(uc_cls.execution_engine.dm.get_value(
            f'{uc_cls.study_name}.SellarOptimScenario.design_space'))
        dspace['value'] = [[2.], [7., 4.]]

        values_dict = {
            f'{uc_cls.study_name}.SellarOptimScenario.design_space': dspace}

        try:
            uc_cls.load_data(from_input_dict=values_dict)

        except:
            dspace = deepcopy(uc_cls.execution_engine.dm.get_value(
                f'{uc_cls.study_name}.SellarOptimScenario.design_space'))
            dspace['value'] = [[1.], [5., 2.]]

            values_dict = {
                f'{uc_cls.study_name}.SellarOptimScenario.design_space': dspace}

            uc_cls.load_data(from_input_dict=values_dict)

    def test_12_optim_scenario_execution_disciplinaryopt(self):
        print("\n Test 12 : Sellar optim solution check with DisciplinaryOpt formulation, check optimum")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 2
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0].mdo_discipline_wrapp.mdo_discipline
        opt_array = array([1., 5., 2.])
        # check that design space in GEMS contains the optimal value (not last
        # iteration)
        assert_array_almost_equal(
            opt_disc.formulation.design_space.get_current_x(), opt_array,
            err_msg="design space does not have optimal value")

        # check that in dm we have xopt value
        z = exec_eng.dm.get_value(f'{self.ns}.{self.sc_name}.{self.c_name}.z')
        opt_z = array([5., 2.])
        print('before array_almost_equal, z=', z)
        assert_array_almost_equal(
            z, opt_z, err_msg="the value of z in dm does not have the optimal value")

        x = exec_eng.dm.get_value(f'{self.ns}.{self.sc_name}.{self.c_name}.x')
        opt_x = array([1.])
        assert_array_almost_equal(
            x, opt_x, err_msg="the value of x in dm does not have the optimal value")

    def test_13_optim_scenario_execution_disciplinaryopt_other_dspace(self):
        print("\n Test 13 : Sellar optim solution check with DisciplinaryOpt formulation, check optimum")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2., 3.]],
                       'lower_bnd': [[0.], [-10., 0., 0.]],
                       'upper_bnd': [[10.], [10., 10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True, False]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 2
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

    def test_16_test_post_run(self):
        print("\n Test 16 : Sellar optim check post run exception")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2., 3.]],
                       'lower_bnd': [[0.], [-10., 0., 0.]],
                       'upper_bnd': [[10.], [10., 10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True, False]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 10
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        disc_dict[f'{self.ns}.SellarOptimScenario.execute_at_xopt'] = False
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        # execute without post run
        res = exec_eng.execute()

        assert isinstance(exec_eng.dm.get_value("optim.SellarOptimScenario.post_processing_mdo_data"), dict)

        # get sosoptimscenario discipline
        disc = exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        disc.formulation.opt_problem.nonproc_constraints = []
        disc.formulation.opt_problem.nonproc_objective = None

        # execute postrun to trigger exception
        disc._post_run()
        dm = exec_eng.dm
        x_first_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.x')
        z_first_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.z')

        # use nominal execution
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)
        disc_dict[f'{self.ns}.SellarOptimScenario.execute_at_xopt'] = True
        exec_eng.dm.set_values_from_dict(disc_dict)
        exec_eng.configure()
        exec_eng.prepare_execution()
        res = exec_eng.execute()
        dm = exec_eng.dm
        x_nominal_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.x')
        z_nominal_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.z')
        assert x_first_execution == x_nominal_execution
        assert_array_equal(z_first_execution, z_nominal_execution)

    def test_16b_test_post_run(self):
        print("\n Test 16b : Sellar optim check post run if no post processing optim outputs")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2., 3.]],
                       'lower_bnd': [[0.], [-10., 0., 0.]],
                       'upper_bnd': [[10.], [10., 10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True, False]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}

        # desactivate the optim post processings
        disc_dict[f'{self.ns}.SellarOptimScenario.desactivate_optim_out_storage'] = True

        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 10
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        disc_dict[f'{self.ns}.SellarOptimScenario.execute_at_xopt'] = False
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        # execute without post run
        res = exec_eng.execute()

        assert KeyError(exec_eng.dm.get_value("optim.SellarOptimScenario.post_processing_mdo_data"))

         # check that the post processings can be executed
        from sostrades_core.tools.post_processing.post_processing_factory import (
            PostProcessingFactory,
        )
        ppf = PostProcessingFactory()
        for disc in exec_eng.root_process.proxy_disciplines:
            filters = ppf.get_post_processing_filters_by_discipline(
                disc)
            graph_list = ppf.get_post_processing_by_discipline(
                disc, filters, as_json=False)

    def _test_17_optim_scenario_execution_disciplinaryopt_complex_step_with_custom_step(self):
        print("\n Test 17 : Sellar optim solution check with DisciplinaryOpt formulation with complex step and a finite differences step")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        disc_dict[f'{self.ns}.SellarOptimScenario.differentiation_method'] = MDOScenario.COMPLEX_STEP
        fd_step = 1.e-15
        disc_dict[f'{self.ns}.SellarOptimScenario.fd_step'] = fd_step
        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        # retrieve discipline to get information to check
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            "optim." + self.sc_name)[0].mdo_discipline_wrapp.mdo_discipline

        assert opt_disc.formulation.opt_problem.fd_step == fd_step

        # check optimal x and f
        sellar_obj_opt = 3.18339 + local_dv
        self.assertAlmostEqual(
            sellar_obj_opt, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
        exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
        assert_array_almost_equal(
            exp_x, opt_disc.optimization_result.x_opt, decimal=4, err_msg="Wrongoptimal x solution")

    def test_18_optim_scenario_mdo_graphs(self):
        print("\n Test 18 : Sellar optim check mdo graphs")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        # -- set up design space
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2., 3.]],
                       'lower_bnd': [[0.], [-10., 0., 0.]],
                       'upper_bnd': [[10.], [10., 10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True, False]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 10
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        disc_dict[f'{self.ns}.SellarOptimScenario.execute_at_xopt'] = False
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        exec_eng.prepare_execution()
        # execute without post run
        res = exec_eng.execute()

        assert isinstance(exec_eng.dm.get_value("optim.SellarOptimScenario.post_processing_mdo_data"), dict)

        # get sosoptimscenario discipline
        disc = exec_eng.root_process.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline
        disc.formulation.opt_problem.nonproc_constraints = []
        disc.formulation.opt_problem.nonproc_objective = None

        # execute postrun to trigger exception
        disc._post_run()
        dm = exec_eng.dm
        x_first_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.x')
        z_first_execution = dm.get_value(
            f'{self.ns}.{self.sc_name}.{self.c_name}.z')

        # use nominal execution
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
                                                                           1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.dm.set_values_from_dict(values_dict)
        disc_dict[f'{self.ns}.SellarOptimScenario.execute_at_xopt'] = True
        exec_eng.dm.set_values_from_dict(disc_dict)
        exec_eng.configure()
        exec_eng.prepare_execution()
        res = exec_eng.execute()

        ppf = PostProcessingFactory()
        disc = exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.SellarOptimScenario')
        filters = ppf.get_post_processing_filters_by_discipline(
            disc[0])
        graph_list = ppf.get_post_processing_by_discipline(
            disc[0], filters, as_json=False)

        self.assertIn("Fitness function", filters[0].filter_values)
        self.assertIn("Design variables", filters[0].filter_values)

        for graph in graph_list:
            # graph.to_plotly().show()
            pass

    def test_19_proxyoptim_data_integrity(self):
        """
        Checks data integrity errors are raised for dspace, ineq_constraints and objective_name variables in ProxyOptim.
        """
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        repo_discopt = 'sostrades_core.sos_processes.test'
        proc_name_discopt = 'test_sellar_opt_discopt'
        builder = factory.get_builder_from_process(repo=repo_discopt,
                                                   mod_id=proc_name_discopt)
        exec_eng.factory.set_builders_to_coupling_builder(builder)
        exec_eng.configure()

        # -- set up design space OK
        dspace_dict = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        # -- set up design space wrong
        dspace_dict_wrong_var = {'variable': ['w', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace_wrong_var = pd.DataFrame(dspace_dict_wrong_var)

        dspace_dict_wrong_act = {'variable': ['x', 'z'],
                       'value': [[1.], [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True, False]]}
        dspace_wrong_act = pd.DataFrame(dspace_dict_wrong_act)

        dspace_dict_wrong_val = {'variable': ['x', 'z'],
                       'value': ['hello', [5., 2.]],
                       'lower_bnd': [[0.], [-10., 0.]],
                       'upper_bnd': [[10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace_wrong_val = pd.DataFrame(dspace_dict_wrong_val)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": True}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_1'] = array([
            1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y_2'] = array([
            1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.z'] = array([
            1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ optim',
                       f'\t|_ {self.sc_name}',
                       f'\t\t|_ {self.c_name}',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace_wrong_var
        exec_eng.load_study_from_input_dict(disc_dict)
        with self.assertRaises(ValueError) as cm:
            exec_eng.execute()
        error_message = "Variable optim.SellarOptimScenario.design_space : Variable w is not among subprocess inputs."
        self.assertTrue(str(cm.exception).startswith(error_message))

        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace_wrong_act
        exec_eng.load_study_from_input_dict(disc_dict)
        with self.assertRaises(ValueError) as cm:
            exec_eng.execute()
        error_message = "Variable optim.SellarOptimScenario.design_space : Columns lower_bnd, upper_bnd, " \
                        "activated_elem and value should have coherent shapes for variable z."
        self.assertTrue(str(cm.exception).startswith(error_message))

        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace_wrong_val
        exec_eng.load_study_from_input_dict(disc_dict)
        with self.assertRaises(ValueError) as cm:
            exec_eng.execute()
        error_message = "Variable optim.SellarOptimScenario.design_space : Columns value, lower_bnd and upper_bnd must be arrays or lists for variable x."
        self.assertTrue(str(cm.exception).startswith(error_message))

        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = ['w']
        exec_eng.load_study_from_input_dict(disc_dict)
        with self.assertRaises(ValueError) as cm:
            exec_eng.execute()
        error_message = "Variable optim.SellarOptimScenario.ineq_constraints : Variable w is not among subprocess outputs."
        self.assertTrue(str(cm.exception).startswith(error_message))

        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = []
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'wrong_obj'
        exec_eng.load_study_from_input_dict(disc_dict)
        with self.assertRaises(ValueError) as cm:
            exec_eng.execute()
        error_message = "Variable optim.SellarOptimScenario.objective_name : Variable wrong_obj is not among subprocess outputs."
        self.assertTrue(str(cm.exception).startswith(error_message))


if '__main__' == __name__:
    cls = TestSoSOptimScenario()
    cls.setUp()
    cls.test_16b_test_post_run()
