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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import AbstractJacobianUnittest
from sostrades_core.execution_engine.design_var.design_var_disc import DesignVarDiscipline


class TestDesignVar(AbstractJacobianUnittest):
    """
    DesignVar unitary test class
    """
    AbstractJacobianUnittest.DUMP_JACOBIAN = False

    def analytic_grad_entry(self):
        return [self.test_derivative
                ]

    def setUp(self):
        self.study_name = 'Test'
        self.ns = f'{self.study_name}'

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
                                               'namespace_in': 'ns_public',
                                               'namespace_out': 'ns_public'
                                               },
                                      'z_in': {'out_name': 'z',
                                               'type': 'array',
                                               'out_type': 'array',
                                               'index': np.arange(0, 10, 1),
                                               'index_name': 'index',
                                               'namespace_in': 'ns_public',
                                               'namespace_out': 'ns_public'
                                               }
                                      }

        self.ee = ExecutionEngine(self.study_name)
        factory = self.ee.factory
        design_var_path = 'sostrades_core.execution_engine.design_var.design_var_disc.DesignVarDiscipline'
        design_var_builder = factory.get_builder_from_module('DesignVar', design_var_path)
        self.ee.ns_manager.add_ns_def({'ns_public': self.ns,
                                       'ns_optim': self.ns})
        self.ee.factory.set_builders_to_coupling_builder(design_var_builder)
        self.ee.configure()

        # -- set up disciplines in Scenario
        values_dict = {}

        # design var
        values_dict[
            f'{self.ns}.DesignVar.design_var_descriptor'] = self.design_var_descriptor
        values_dict[
            f'{self.ns}.design_space'] = self.dspace
        values_dict[f'{self.ns}.x_in'] = np.array([1., 1., 3., 2.])
        values_dict[f'{self.ns}.z_in'] = np.array([5., 2., 2., 1., 1., 1.])
        self.values_dict = values_dict

    def test_01_check_execute_default_dataframe_fill(self):
        '''

        Test the class with the default method one column per key

        '''
        # load and run
        self.ee.load_study_from_input_dict(self.values_dict)
        self.ee.configure()
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(f'{self.ns}.DesignVar')[0]

        # checks output type is well created for dataframes (most commonly used)
        df = disc.get_sosdisc_outputs('x')
        assert isinstance(df, pd.DataFrame)
        assert all(
            df.columns == [self.design_var_descriptor['x_in']['index_name'], self.design_var_descriptor['x_in']['key']])
        assert (df['value'].values == self.values_dict[f'{self.ns}.x_in']).all()

    def test_02_check_execute_default_dataframe_fill(self):
        '''

        Test the class with the method 'one column for key, one for value'

        '''
        self.design_var_descriptor = {'x_in': {'out_name': 'x',
                                               'type': 'array',
                                               'out_type': 'dataframe',
                                               'key': 'value',
                                               'index': np.arange(0, 4, 1),
                                               'namespace_in': 'ns_public',
                                               'namespace_out': 'ns_public',
                                               DesignVarDiscipline.DATAFRAME_FILL:
                                                   DesignVarDiscipline.DATAFRAME_FILL_POSSIBLE_VALUES[1],
                                               DesignVarDiscipline.COLUMNS_NAMES: ['name', 'sharevalue']
                                               },
                                      'z_in': {'out_name': 'z',
                                               'type': 'array',
                                               'out_type': 'array',
                                               'index': np.arange(0, 10, 1),
                                               'index_name': 'index',
                                               'namespace_in': 'ns_public',
                                               'namespace_out': 'ns_public'
                                               }
                                      }
        self.values_dict[
            f'{self.ns}.DesignVar.design_var_descriptor'] = self.design_var_descriptor

        self.ee.load_study_from_input_dict(self.values_dict)
        self.ee.configure()
        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(f'{self.ns}.DesignVar')[0]

        # checks output type is well created for dataframes (most commonly used)
        df = disc.get_sosdisc_outputs('x')
        assert isinstance(df, pd.DataFrame)
        column_names = self.design_var_descriptor['x_in'][DesignVarDiscipline.COLUMNS_NAMES]
        assert all(
            df.columns == column_names)
        assert (df[column_names[0]].values == self.design_var_descriptor['x_in']['key']).all()
        assert (df[column_names[1]].values == self.values_dict[f'{self.ns}.x_in']).all()


if '__main__' == __name__:
    cls = TestDesignVar()
    cls.setUp()
    cls.test_01_check_execute()
