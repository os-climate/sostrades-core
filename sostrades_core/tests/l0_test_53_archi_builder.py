'''
Copyright 2022 Remy SAS

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
import pandas as pd
from logging import Handler

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.sos_processes.test.test_architecture_standard.usecase_simple_architecture import Study
from tempfile import gettempdir


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestArchiBuilder(unittest.TestCase):
    """
    Architecture Builder test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

    def test_01_get_builder_from_class_name(self):

        builder = self.exec_eng.factory.get_builder_from_class_name('SumVB', 'SumValueBlockDiscipline', [
            'sostrades_core.sos_wrapping'])

        disc = builder.build()

        self.assertEqual(
            'sostrades_core.sos_wrapping.sum_valueblock_discipline', disc.get_module())

    def test_02_build_architecture_standard(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Tomato'],
             'Current': ['Remy', 'Tomato', 'Opex', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # Check builders and activation_dict
        disc_archi = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.namespace}.Business')[0]
        self.assertEqual(list(disc_archi.activated_builders.keys()), [
                         'Business.Remy', 'Business.Tomato', 'Business.Remy.Opex', 'Business.Tomato.CAPEX'])
        self.assertDictEqual(disc_archi.activation_dict, {'Business': {
                             'Business.Remy.Opex': 'Remy', 'Business.Tomato.CAPEX': 'Tomato'}})

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Tomato'],
             'Opex': [True, False],
             'CAPEX': [False, True]})

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Tomato'],
             'Opex': [True, False],
             'CAPEX': [False, False]})
        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t|_ Tomato', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        disc_Remy = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Remy')[0]

        # No children because the scatter has no children
        self.assertListEqual([disc.sos_name for disc in disc_Remy.config_dependency_disciplines], [
                             'Remy.Opex'])

        # desactiation of all builders
        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Tomato'],
             'Opex': [False, False],
             'CAPEX': [False, False]})
        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t|_ Tomato', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_03_build_architecture_standard_without_activation(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Tomato'],
             'Current': ['Remy', 'Tomato', 'Opex', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # Check builders and activation_dict
        disc_archi = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.namespace}.Business')[0]
        self.assertEqual(list(disc_archi.activated_builders.keys()), [
                         'Business.Remy', 'Business.Tomato', 'Business.Remy.Opex', 'Business.Tomato.CAPEX'])
        self.assertDictEqual(disc_archi.activation_dict, {})

        activation_df = pd.DataFrame(
            {'Remy': [True],
             'Tomato': [True],
             'Opex': [True],
             'CAPEX': [True]})

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        # CAPEX desactivation
        activation_df = pd.DataFrame(
            {'Remy': [True],
             'Tomato': [True],
             'Opex': [True],
             'CAPEX': [False]})

        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t|_ Tomato', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        disc_Remy = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Remy')[0]

        # No children because the scatter has no children
        self.assertListEqual([disc.sos_name for disc in disc_Remy.config_dependency_disciplines], [
                             'Remy.Opex'])

        # desactiation of all builders
        activation_df = pd.DataFrame(
            {'Remy': [False],
             'Tomato': [False],
             'Opex': [False],
             'CAPEX': [False]})

        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_04_check_architecture_df(self):

        # mydict = {'input_name': 'product_list',
        #           'input_type': 'string_list',
        #           'input_ns': 'ns_business',
        #           'output_name': 'product_name',
        #           'scatter_ns': 'ns_ac'}
        # self.exec_eng.scattermap_manager.add_build_map('product_list', mydict)
        #
        # self.exec_eng.ns_manager.add_ns_def({'ns_business': self.study_name})

        vb_builder_name = 'ArchiBuilder'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['ScatterArchi'],
             'Current': ['ChildrenScatter'],
             'Type': ['ValueBlockDiscipline'],
             'Action': ['standard'],
             'Activation': [False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['ArchiBuilder', 'ArchiBuilder', 'Standard1', 'Standard2', 'Standard2'],
             'Current': ['Standard1', 'Standard2', 'Scatter', 'SubArchi', 'ScatterArchi'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'product_list', 'ValueBlockDiscipline'), ('standard'),
                        ('scatter_architecture', 'product_list', 'SumValueBlockDiscipline', sub_architecture_df)],
             'Activation': [False, False, False, False, False], })

        print('architecture df: \n', architecture_df)

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Standard1',
                       '\t\t\t|_ Scatter',
                       '\t\t\t\t|_ driver',
                       '\t\t|_ Standard2',
                       '\t\t\t|_ SubArchi',
                       '\t\t\t|_ ScatterArchi',
                       '\t\t\t\t|_ driver', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Standard1',
                       '\t\t\t|_ Scatter',
                       '\t\t|_ Standard2',
                       '\t\t\t|_ SubArchi',
                       '\t\t\t|_ ScatterArchi', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame({'product_list': [None], 'ChildrenScatter': True, 'Standard1': [True], 'Standard2': [
                                     True], 'Scatter': [True], 'SubArchi': [True], 'ScatterArchi': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.ArchiBuilder.activation_df').to_dict())

        activation_df = pd.DataFrame({'product_list': [None], 'Standard1': [True], 'Standard2': [
                                     True], 'Scatter': [False], 'SubArchi': [False], 'ScatterArchi': [True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.ArchiBuilder.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Standard1',
                       '\t\t|_ Standard2',
                       '\t\t\t|_ ScatterArchi', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.ArchiBuilder.activation_df').to_dict())

    def test_05_build_architecture_scatter(self):

        # mydict = {'input_name': 'product_list',
        #           'input_type': 'string_list',
        #           'input_ns': 'ns_public',
        #           'output_name': 'AC_name',
        #           'scatter_ns': 'ns_ac'}
        # self.exec_eng.scattermap_manager.add_build_map('product_list', mydict)
        #
        # self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['CAPEX'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'product_list', 'ValueBlockDiscipline')],
             'Activation': [False]})

        print('architecture df: \n', architecture_df)

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'product_list': [None],
             'CAPEX': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict())

        activation_df = pd.DataFrame(
            {'product_list': ['AC1', 'AC2', 'AC3'],
             'CAPEX': [True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ CAPEX',
                       '\t\t\t|_ AC1',
                       '\t\t\t|_ AC2',
                       '\t\t\t|_ AC3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'product_list': ['AC1', 'AC2', 'AC3'],
             'CAPEX': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ CAPEX',
                       '\t\t\t|_ AC1',
                       '\t\t\t|_ AC2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_06_build_architecture_scatter_with_multiple_configure(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Tomato'],
             'Current': ['Remy', 'Tomato', 'Opex', 'CAPEX', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'product_list', 'ValueBlockDiscipline'), ('scatter', 'product_list', 'ValueBlockDiscipline'), ('scatter', 'product_list', 'ValueBlockDiscipline')],
             'Activation': [True, True, False, False, False], })

        print('architecture df: \n', architecture_df)

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t\t|_ CAPEX',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # test configure and cleaning

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Remy', 'Tomato'],
             'product_list': ['A320', 'A321', 'B737'],
             'CAPEX': [True, True, True],
             'Opex': [True, False, False]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t\t\t|_ A320',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ A320',
                       '\t\t\t\t|_ A321',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ B737', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['A320', 'A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.Opex.driver.scenario_df')['scenario_name'].values.tolist(), ['A320'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Tomato.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['B737'])

        # test configure and cleaning

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Remy', 'Tomato'],
             'product_list': ['A320', 'A321', 'B737'],
             'CAPEX': [False, True, True],
             'Opex': [True, True, False]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Opex',
                       '\t\t\t\t|_ A320',
                       '\t\t\t\t|_ A321',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ A321',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ B737']
        exp_tv_str = '\n'.join(exp_tv_list)
        self.exec_eng.display_treeview_nodes()

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.Opex.driver.scenario_df')['scenario_name'].values.tolist(), ['A320', 'A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Tomato.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['B737'])

    def test_07_architecture_multi_level(self):
        # FIXME: driver node is not being hidden ?
        self.repo_business = 'business_case.sos_wrapping.'
        vb_builder_name = 'Business'

        component_sales_architecture_df = pd.DataFrame(
            {'Parent': ['CAPEX', 'CAPEX'],
             'Current': ['Airframe', 'Propulsion'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': ['standard', 'standard'],
             'Activation': [False, False]})

        component_fhs_architecture_df = pd.DataFrame(
            {'Parent': ['FHS', 'FHS'],
             'Current': ['Airframe', 'Propulsion'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': ['standard', 'standard'],
             'Activation': [False, False]})

        component_Delivery_architecture_df = pd.DataFrame(
            {'Parent': ['Delivery', 'Delivery'],
             'Current': ['Airframe', 'Propulsion'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': ['standard', 'standard'],
             'Activation': [False, False]})

        Opex_architecture_df = pd.DataFrame(
            {'Parent': ['Opex', 'Opex'],
             'Current': ['FHS', 'Delivery'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('scatter_architecture', 'product_list', 'SumValueBlockDiscipline', component_fhs_architecture_df), ('scatter_architecture', 'product_list', 'SumValueBlockDiscipline', component_Delivery_architecture_df)],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Tomato'],
             'Current': ['Remy', 'Tomato', 'CAPEX', 'Opex', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter_architecture', 'product_list', 'SumValueBlockDiscipline', component_sales_architecture_df), ('architecture', Opex_architecture_df), ('scatter', 'product_list', 'ValueBlockDiscipline')],
             'Activation': [True, True, False, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})
        activation_df = pd.DataFrame({'Business': ['Remy', 'Remy', 'Tomato', 'Tomato'],
                                      'product_list': ['AC1', 'AC2', 'AC3', 'AC4'],
                                      'CAPEX': [True, True, True, True],
                                      'Opex': [True, True, False, False],
                                      'Airframe': [True, True, False, False],
                                      'Propulsion': [True, True, False, False],
                                      'FHS': [True, True, False, False],
                                      'Delivery': [True, True, False, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ AC1',
                       '\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ AC2',
                       '\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Propulsion',
                       '\t\t\t|_ Opex',
                       '\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Delivery',
                       '\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t\t|_ AC3',
                       '\t\t\t\t|_ AC4', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['AC1', 'AC2'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Tomato.CAPEX.driver.scenario_df')['scenario_name'].values.tolist(), ['AC3', 'AC4'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.Opex.FHS.driver.scenario_df')['scenario_name'].values.tolist(), ['AC1', 'AC2'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.Opex.Delivery.driver.scenario_df')['scenario_name'].values.tolist(), ['AC1', 'AC2'])

    def test_08_process_simple_architecture_execution(self):

        repo = 'sostrades_core.sos_processes.test'
        builder = self.exec_eng.factory.get_builder_from_process(
            repo, 'test_architecture_standard')

        my_handler = UnitTestHandler()
        self.exec_eng.logger.addHandler(my_handler)

        self.exec_eng.factory.set_builders_to_coupling_builder(builder)
        self.exec_eng.load_study_from_input_dict({})

        usecase_simple_architecture = Study()
        usecase_simple_architecture.study_name = self.study_name
        values_dict_list = usecase_simple_architecture.setup_usecase()
        values_dict = {}
        for values_list in values_dict_list:
            values_dict.update(values_list)

        self.exec_eng.load_study_from_input_dict(values_dict)

        activ_df = pd.DataFrame({'Business': ['Remy', 'Tomato', 'Zucchini'],
                                 'CAPEX': [True, True, True],
                                 'OPEX': [True, True, True],
                                 'Manhour': [True, False, True]})
        values_dict['MyCase.Business.activation_df'] = activ_df

        self.exec_eng.load_study_from_input_dict(values_dict)

        msg_log_error = 'Invalid Value Block Activation Configuration: [\'Zucchini\'] in column Business not in *possible values* [\'Remy\', \'Tomato\']'

        self.assertTrue(msg_log_error in my_handler.msg_list)

        msg_log_error = 'Invalid Value Block Activation Configuration: value block OPEX not available for [\'Tomato\']'
        self.assertTrue(msg_log_error in my_handler.msg_list)

        activ_df = pd.DataFrame({'Business': ['Remy',   'Tomato'],
                                 'CAPEX': [True, True],
                                 'OPEX': [True, False],
                                 'Manhour': [True, False]})

        self.assertTrue(activ_df.equals(
            self.exec_eng.dm.get_value('MyCase.Business.activation_df')))

        activ_df = pd.DataFrame({'Business': ['Remy', 'Tomato'],
                                 'CAPEX': [True, True],
                                 'OPEX': [True, True],
                                 'Manhour': [True, False]})
        values_dict['MyCase.Business.activation_df'] = activ_df
        self.exec_eng.load_study_from_input_dict(values_dict)

        self.assertTrue(activ_df.equals(
            self.exec_eng.dm.get_value('MyCase.Business.activation_df')))

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ Business',
                       '\t\t|_ Remy',
                       '\t\t\t|_ CAPEX',
                       '\t\t\t|_ OPEX',
                       '\t\t\t\t|_ Manhour',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ CAPEX']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        for disc in self.exec_eng.factory.proxy_disciplines:
            self.assertEqual(disc.status, 'DONE')

    def test_09_build_scatter_architecture_at_architecture_node(self):
        # FIXME: driver node is not being hidden ?
        vb_builder_name = 'Business'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['Cooking'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': ['standard'],
             'Activation': [False], })

        architecture_df = pd.DataFrame(
            {'Parent': [None],
             'Current': ['Business'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter_architecture', 'Actor_list', 'ValueBlockDiscipline', sub_architecture_df)],
             'Activation': [False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.exec_eng.load_study_from_input_dict({})
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'Actor_list': ['Remy', 'Tomato', 'Zucchini'],
             'Business': [True, True, False],
             'Cooking': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t\t|_ Cooking',
                       '\t\t|_ Tomato',
                       '\t\t\t|_ Cooking']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_10_build_scatter_at_architecture_node(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': [None],
             'Current': ['Business'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'Actor_list', 'ValueBlockDiscipline')],
             'Activation': [False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.exec_eng.load_study_from_input_dict({})
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'Actor_list': ['Remy', 'Tomato', 'Zucchini'],
             'Business': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Remy',
                       '\t\t|_ Tomato']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_11_build_standard_value_block_at_architecture_node(self):
        # TODO: not actually testing this functionality, rewrite test when ArchiBuilder as tool (see code commented out)
        vb_builder_name = 'ArchiBuilder' # 'Opex'

        architecture_df = pd.DataFrame(
            {'Parent': [None, 'Opex', 'Opex', 'Opex', 'Opex'],
             'Current': ['Opex', 'Delivery', 'Manhour', 'Cooking', 'Energy'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.exec_eng.load_study_from_input_dict({})
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       f'\t\t|_ Opex',
                       f'\t\t\t|_ Delivery',
                       f'\t\t\t|_ Manhour',
                       f'\t\t\t|_ Cooking',
                       f'\t\t\t|_ Energy']
        # exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
        #                f'|_ {self.namespace}',
        #                f'\t|_ {vb_builder_name}',
        #                f'\t\t|_ Delivery',
        #                f'\t\t|_ Manhour',
        #                f'\t\t|_ Cooking',
        #                f'\t\t|_ Energy']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestArchiBuilder()
    cls.setUp()
    cls.test_08_process_simple_architecture_execution()
