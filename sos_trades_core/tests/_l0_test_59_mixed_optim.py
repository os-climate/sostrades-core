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
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sos_trades_core.sos_processes.test.test_sellar_opt.usecase import Study as study_sellar_opt
from sos_trades_core.sos_processes.test.test_Griewank_opt.usecase import Study as study_griewank
from sos_trades_core.sos_processes.test.test_sellar_opt_idf.usecase import Study as study_sellar_idf
import os
from gemseo.core.mdo_scenario import MDOScenario
from copy import deepcopy


class TestMixedOptimAlgorithms(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'mixed_opt'
        self.ns = f'{self.study_name}'
        self.sc_name = "MixedOptScenario"
        self.c_name = "MixedOptCoupling"

        dspace_dict = {'variable': ['x', 'y'],
                       'value': [1., 1.],
                       'lower_bnd': [0., 1.],
                       'upper_bnd': [3., 2]
                       }

        self.dspace = pd.DataFrame(dspace_dict)
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_mixedopt_coupling'

    def test_01_mixed_optim_outer_approximation(self):
        print("\n Test 1 : Mixed optim case with monolevel Outer Approximation")
        
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        builder = factory.get_builder_from_process(repo=self.repo,
                                                   mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(builder)

        exec_eng.configure()

        #-- set up design space
        dspace = pd.DataFrame(self.dspace)

        #-- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 200
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "OuterApproximation"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = ['constr']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-6,
                                                                    "ineq_tolerance": 1e-6,
                                                                    "normalize_design_space": False}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Mixed Opt inputs
        values_dict = {}
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.x'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.y'] = 1.
        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        res = exec_eng.execute()

        # retrieve discipline to check the result...
        opt_disc = exec_eng.dm.get_disciplines_with_name(
            self.study_name + '.' + self.sc_name)[0]

        # check optimal x, y and objective value
#         self.assertAlmostEqual(
#             sellar_obj_opt, opt_disc.optimization_result.f_opt, places=4, msg="Wrong objective value")
#         exp_x = array([8.3109e-15, 1.9776e+00, 3.2586e-13])
#         assert_array_almost_equal(
#             exp_x, opt_disc.optimization_result.x_opt, decimal=4,
#             err_msg="Wrong optimal x solution")


if '__main__' == __name__:
    cls = TestMixedOptimAlgorithms()
    cls.setUp()
    cls.test_16_test_post_run()
