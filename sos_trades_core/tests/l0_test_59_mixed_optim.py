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
unit test for optimization scenario
"""

import unittest
from numpy import array, set_printoptions
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal
import os
from copy import deepcopy

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from gemseo.algos.design_space import DesignSpace



class TestMixedOptimAlgorithms(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        # namespaces
        self.study_name = 'mixed_opt'
        self.ns = f'{self.study_name}'
        self.sc_name = "MixedOptimScenario"
        self.c_name = "MixedCoupling"
        self.disc_name = "DiscMixedOpt"
        # paths and processes
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name_linear_feasible = 'test_mixedopt_linear_feasible'
        
    def _get_basic_solver_options(self, dspace):
        # Optim inputs
        opt_dict = {}
        opt_dict[f'{self.ns}.{self.sc_name}.max_iter'] = 100
        opt_dict[f'{self.ns}.{self.sc_name}.algo'] = "OuterApproximation"
        opt_dict[f'{self.ns}.{self.sc_name}.design_space'] = dspace
        opt_dict[f'{self.ns}.{self.sc_name}.formulation'] = 'DisciplinaryOpt'
        opt_dict[f'{self.ns}.{self.sc_name}.objective_name'] = 'obj'
        opt_dict[f'{self.ns}.{self.sc_name}.ineq_constraints'] = ['constr']
        opt_dict[f'{self.ns}.{self.sc_name}.differentiation_method'] = 'user'
        
        algo_options_master = {}
        
        algo_options_slave = {"ftol_rel": 1e-10,
                              "ineq_tolerance": 2e-3,
                              "normalize_design_space": False}
        
        opt_dict[f'{self.ns}.{self.sc_name}.algo_options'] = {"ftol_abs": 1e-10,
                                                     "ineq_tolerance": 2e-3,
                                                     "normalize_design_space": False,
                                                     "algo_NLP": "SLSQP",
                                                     "algo_options_NLP": algo_options_slave,
                                                     "algo_options_MILP": algo_options_master}
        
        return opt_dict
        
    
    def test_01_mixed_optim_outer_approximation_linear_pb_feasible_NLP(self):
        print("\n Test 1 : Mixed optim case with monolevel Outer Approximation on linear problem with feasible NLPs")
        
        #-- set up exec engine with the process
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name_linear_feasible)
        exec_eng.factory.set_builders_to_coupling_builder(builder)
        exec_eng.configure()

        #-- set up design space
        dspace_dict = {'variable': ['x1', 'x2'],
                       'value': [[2], [3.]],
                       'lower_bnd': [[0], [0.]],
                       'upper_bnd': [[999], [999.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True]],
                       'variable_type' : [DesignSpace.INTEGER, DesignSpace.FLOAT]}
        dspace = pd.DataFrame(dspace_dict)

        #-- configure
        opt_dict = self._get_basic_solver_options(dspace)
        
        #-- set up disciplines inputs in Scenario
        opt_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.{self.disc_name}.x1'] = array([2.])
        opt_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.{self.disc_name}.x2'] = array([4])

        exec_eng.load_study_from_input_dict(opt_dict)
        
        #-- execution
        res = exec_eng.execute()

        #-- retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            self.study_name + '.' + self.sc_name)[0]
        
        # check optimal x, y and objective value
        obj_opt_ref = -30.5
        self.assertAlmostEqual(
            obj_opt_ref, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
        
        xopt_ref = array([5, 3.1])
        assert_array_almost_equal(
            xopt_ref, opt_disc.optimization_result.x_opt, decimal=4, err_msg="Wrong objective value")
        
        
#         dspace_out = exec_eng.dm.get_value(f'{self.ns}.{self.sc_name}.'+'design_space_out')

#         self.assertAlmostEqual(
#             sellar_obj_opt, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
#         exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
#         assert_array_almost_equal(
#             exp_x, opt_disc.optimization_result.x_opt, decimal=4,
#             err_msg="Wrong optimal x solution")

    def _test_02_mixed_optim_outer_approximation_nonlinear_pb_feasible_NLP(self):
        print("\n Test 2 : Mixed optim case with monolevel Outer Approximation on linear problem with feasible NLPs")
        
        #-- set up exec engine with the process
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)
        exec_eng.factory.set_builders_to_coupling_builder(builder)
        exec_eng.configure()

        #-- set up design space
        dspace_dict = {'variable': ['x1', 'x2'],
                       'value': [[2], [3.]],
                       'lower_bnd': [[0], [0.]],
                       'upper_bnd': [[999], [999.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True]],
                       'variable_type' : [DesignSpace.INTEGER, DesignSpace.FLOAT]}
        dspace = pd.DataFrame(dspace_dict)

        #-- configure
        opt_dict = self._get_basic_solver_options(dspace)
        exec_eng.load_study_from_input_dict(opt_dict)
        
        #-- execution
        res = exec_eng.execute()

        #-- retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            self.study_name + '.' + self.sc_name)[0]
        
        # check optimal x, y and objective value
        obj_opt_ref = -30.5
        self.assertAlmostEqual(
            obj_opt_ref, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
        
        xopt_ref = array([5, 3.1])
        assert_array_almost_equal(
            xopt_ref, opt_disc.optimization_result.x_opt, decimal=4, err_msg="Wrong objective value")

if '__main__' == __name__:
    cls = TestMixedOptimAlgorithms()
    cls.setUp()
    cls.test_01_mixed_optim_outer_approximation_linear_pb_feasible_NLP()
