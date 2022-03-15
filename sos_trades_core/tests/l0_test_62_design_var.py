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
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc


AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
OBJECTIVE = FunctionManager.OBJECTIVE


class TestDesignVar(AbstractJacobianUnittest):
    """
    DesignVar test class
    """
    AbstractJacobianUnittest.DUMP_JACOBIAN = False

    def analytic_grad_entry(self):
        return [self.test_derivative
                ]

    def setUp(self):

        self.study_name = 'Test'
        self.ns = f'{self.study_name}'
        self.sc_name = "SellarOptimScenario"
        self.c_name = "SellarCoupling"

        dspace_dict = {'variable': ['x_in', 'z_in'],
                       'value': [[1., 1., 3., 2.], [5., 2., 2., 1., 1., 1.]],
                       'lower_bnd': [[0., 0., 0., 0.], [-10., 0., -10., -10., -10., -10.]],
                       'upper_bnd': [[10., 10., 10., 10.], [10., 10., 10., 10., 10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        self.dspace = pd.DataFrame(dspace_dict)

        self.design_var_descriptor = {'x_in': {'out_name': 'x',
                                           'type': 'array',
                                           'out_type': 'dataframe',
                                           'key': 'value',
                                           'index': np.arange(0, 4, 1),
                                           'index_name': 'test',
                                           'namespace_in': 'ns_OptimSellar',
                                           'namespace_out': 'ns_OptimSellar'
                                               },
                                  'z_in': {'out_name': 'z',
                                           'type': 'array',
                                           'out_type': 'array',
                                           'index': np.arange(0, 10, 1),
                                           'index_name': 'index',
                                           'namespace_in': 'ns_OptimSellar',
                                           'namespace_out': 'ns_OptimSellar'
                                           }
                                      }
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_w_design_var'


        self.ee = ExecutionEngine(self.study_name)
        factory = self.ee.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(opt_builder)
        self.ee.configure()

        # -- set up disciplines in Scenario
        values_dict = {}

        # design var
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.DesignVar.design_var_descriptor'] = self.design_var_descriptor

        # Optim inputs
        values_dict[f'{self.ns}.{self.sc_name}.max_iter'] = 1
        values_dict[f'{self.ns}.{self.sc_name}.algo'] = "SLSQP"
        values_dict[f'{self.ns}.{self.sc_name}.design_space'] = self.dspace
        values_dict[f'{self.ns}.{self.sc_name}.formulation'] = 'DisciplinaryOpt'
        values_dict[f'{self.ns}.{self.sc_name}.objective_name'] = 'obj'
        values_dict[f'{self.ns}.{self.sc_name}.ineq_constraints'] = [f'c_1', f'c_2']
        values_dict[f'{self.ns}.{self.sc_name}.algo_options'] = {"ftol_rel": 1e-10,
                                                                 "ineq_tolerance": 2e-3,
                                                                 "normalize_design_space": False}

        # Sellar inputs
        local_dv = 10.
        values_dict[f'{self.ns}.{self.sc_name}.x_in'] = np.array([1., 1., 3., 2.])
        values_dict[f'{self.ns}.{self.sc_name}.y_1'] = 5.
        values_dict[f'{self.ns}.{self.sc_name}.y_2'] = 1.
        values_dict[f'{self.ns}.{self.sc_name}.z_in'] = np.array([5., 2., 2., 1., 1., 1.])
        values_dict[f'{self.ns}.{self.sc_name}.{self.c_name}.Sellar_Problem.local_dv'] = local_dv

        # function manager
        func_df = pd.DataFrame(
            columns=['variable', 'ftype', 'weight', AGGR_TYPE])
        func_df['variable'] = ['c_1', 'c_2', 'obj']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT, OBJECTIVE]
        func_df['weight'] = [200, 0.000001, 0.1]
        func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM]
        func_mng_name = 'FunctionManager'

        prefix = f'{self.study_name}.{self.sc_name}.{self.c_name}.{func_mng_name}.'
        values_dict[prefix +
                    FunctionManagerDisc.FUNC_DF] = func_df

        # load and run
        self.ee.load_study_from_input_dict(values_dict)

    def test_01_check_execute(self):
        print("\n Test 1 : check configure and treeview")
        self.ee.configure()
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(f'{self.ns}.{self.sc_name}.{self.c_name}.DesignVar')[0]

        # checks output type is well created for dataframes (most commonly used)
        df = disc.get_sosdisc_outputs('x')
        assert isinstance(df, pd.DataFrame)
        assert all(df.columns == [self.design_var_descriptor['x_in']['index_name'], self.design_var_descriptor['x_in']['key']])

        filters = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filters)
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_derivative(self):
        disc = self.ee.dm.get_disciplines_with_name(f'{self.ns}.{self.sc_name}.{self.c_name}.DesignVar')[0]

        input_names = [f'{self.ns}.{self.sc_name}.x_in',
                       f'{self.ns}.{self.sc_name}.z_in',
                       ]

        output_names = [f'{self.ns}.{self.sc_name}.x',
                        f'{self.ns}.{self.sc_name}.z',
                        ]

        self.check_jacobian(location=dirname(__file__), filename=f'jacobian_design_var_bspline.pkl', discipline=disc, step=1e-15, inputs=input_names,
                            outputs=output_names, derr_approx='complex_step')


if '__main__' == __name__:
    cls = TestDesignVar()
    cls.setUp()
    cls.test_01_check_execute()
    cls.test_derivative()
