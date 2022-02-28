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
import unittest
import numpy as np
import pandas as pd
from os.path import join, dirname
from pandas import DataFrame, read_csv

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest


class TestDesignVar(unittest.TestCase):
    """
    DesignVar test class
    """

    def setUp(self):
        self.study_name = 'optim'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarOptimScenario"
        self.c_name = "SellarCoupling"

        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [1., [5., 2.], 1., 1.],
                       'lower_bnd': [0., [-10., 0.], -100., -100.],
                       'upper_bnd': [10., [10., 10.], 100., 100.],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}

        self.dspace = pd.DataFrame(dspace_dict)
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt'

        self.output_descriptor = {}

        self.output_descriptor['x'] = {'out_name': 'forest_investment', 'type': 'dataframe',
                                                       'key': 'forest_investment', 'namespace_in': 'ns_witness',
                                                       'namespace_out': 'ns_witness'}

        self.output_descriptor['z'] = {'out_name': 'deforestation_surface', 'type': 'dataframe',
                                                        'key': 'deforested_surface', 'namespace_in': 'ns_witness',
                                                        'namespace_out': 'ns_witness'}

        self.output_descriptor['y_1'] = {'out_name': 'red_to_white_meat', 'type': 'array',
                                                       'namespace_in': 'ns_witness', 'namespace_out': 'ns_witness'}

        self.output_descriptor['y_2'] = {'out_name': 'meat_to_vegetables', 'type': 'array',
                                                        'namespace_in': 'ns_witness', 'namespace_out': 'ns_witness'}

    def test_01_check_execute(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        #-- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs
        disc_dict[f'{self.ns}.SellarOptimScenario.max_iter'] = 100
        disc_dict[f'{self.ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{self.ns}.SellarOptimScenario.design_space'] = self.dspace
        disc_dict[f'{self.ns}.SellarOptimScenario.formulation'] = 'MDF'
        disc_dict[f'{self.ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{self.ns}.SellarOptimScenario.ineq_constraints'] = [
            f'c_1', f'c_2']

        disc_dict[f'{self.ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-10,
                                                                    "ineq_tolerance": 2e-3,
                                                                    "normalize_design_space": False}
        exec_eng.dm.set_values_from_dict(disc_dict)

        # Sellar inputs
        values_dict = {}
        local_dv = 10.
        values_dict[f'{self.ns}.{self.sc_name}.x'] = np.array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = np.array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = np.array([1.])
        values_dict[f'{self.ns}.{self.sc_name}.z'] = np.array([1., 1.])
        exec_eng.dm.set_values_from_dict(values_dict)
        values_dict[f'{self.ns}.{self.sc_name}.Sellar_Problem.local_dv'] = local_dv

        exec_eng.dm.set_values_from_dict(values_dict)

        exec_eng.configure()

        exec_eng.execute()

        disc = exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.{self.c_name}')[0]



    def test_execute(self):

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        filterr = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filterr)
        # for graph in graph_list:
        #    graph.to_plotly().show()

    # def test_derivative(self):
    #     disc_techno = self.ee.root_process.sos_disciplines[0]
    #     #AbstractJacobianUnittest.DUMP_JACOBIAN = True
    #     output_names = [f'{self.name}.invest_mix',
    #                     f'{self.name}.deforestation_surface',
    #                     f'{self.name}.forest_investment',
    #                     f'{self.name}.red_to_white_meat',
    #                     f'{self.name}.meat_to_vegetables']
    #
    #     self.check_jacobian(location=dirname(__file__), filename=f'jacobian_design_var_bspline_invest_distrib_full.pkl', discipline=disc_techno, step=1e-15, inputs=self.input_names,
    #                         outputs=output_names, derr_approx='complex_step')
