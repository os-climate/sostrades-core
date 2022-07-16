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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import pandas as pd
from logging import Handler

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_processes.test.test_architecture.usecase_simple_architecture import Study
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
            'sos_trades_core.sos_wrapping'])

        disc = builder.build()

        self.assertEqual(
            'sos_trades_core.sos_wrapping.sum_valueblock_discipline', disc.__module__)

    def test_02_build_architecture_standard(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales'],
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
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # Check builders and activation_dict
        disc_archi = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.namespace}.Business')[0]
        self.assertEqual(list(disc_archi.activated_builders.keys()), [
                         'Business.Airbus', 'Business.Boeing', 'Business.Airbus.Services', 'Business.Boeing.AC_Sales'])
        self.assertDictEqual(disc_archi.activation_dict, {'Business': {
                             'Business.Airbus.Services': 'Airbus', 'Business.Boeing.AC_Sales': 'Boeing'}})

        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Boeing'],
             'Services': [True, False],
             'AC_Sales': [False, True]})

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Boeing'],
             'Services': [True, True],
             'AC_Sales': [False, False]})
        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t|_ Boeing', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        disc_airbus = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus')[0]

        # No children because the scatter has no children
        self.assertListEqual([disc.sos_name for disc in disc_airbus.children_list], [
                             'Airbus.Services'])

        # desactiation of all builders
        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Boeing'],
             'Services': [False, False],
             'AC_Sales': [False, False]})
        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t|_ Boeing', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_03_build_architecture_standard_without_activation(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales'],
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
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # Check builders and activation_dict
        disc_archi = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.namespace}.Business')[0]
        self.assertEqual(list(disc_archi.activated_builders.keys()), [
                         'Business.Airbus', 'Business.Boeing', 'Business.Airbus.Services', 'Business.Boeing.AC_Sales'])
        self.assertDictEqual(disc_archi.activation_dict, {})

        activation_df = pd.DataFrame(
            {'Airbus': [True],
             'Boeing': [True],
             'Services': [True],
             'AC_Sales': [True]})

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        # AC_Sales desactivation
        activation_df = pd.DataFrame(
            {'Airbus': [True],
             'Boeing': [True],
             'Services': [True],
             'AC_Sales': [False]})

        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t|_ Boeing', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict(), activation_df.to_dict())

        disc_airbus = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus')[0]

        # No children because the scatter has no children
        self.assertListEqual([disc.sos_name for disc in disc_airbus.children_list], [
                             'Airbus.Services'])

        # desactiation of all builders
        activation_df = pd.DataFrame(
            {'Airbus': [False],
             'Boeing': [False],
             'Services': [False],
             'AC_Sales': [False]})

        dict_values = {'MyCase.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_04_check_architecture_df(self):

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_business',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_business': self.study_name})

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
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('standard'), ('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', sub_architecture_df)],
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
                       '\t\t|_ Standard2',
                       '\t\t\t|_ SubArchi',
                       '\t\t\t|_ ScatterArchi', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame({'AC_list': [None], 'ChildrenScatter': True, 'Standard1': [True], 'Standard2': [
                                     True], 'Scatter': [True], 'SubArchi': [True], 'ScatterArchi': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.ArchiBuilder.activation_df').to_dict())

        activation_df = pd.DataFrame({'AC_list': [None], 'Standard1': [True], 'Standard2': [
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

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['AC_Sales'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', 'ValueBlockDiscipline')],
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
                       '\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'AC_list': [None],
             'AC_Sales': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict())

        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC2', 'AC3'],
             'AC_Sales': [True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        self.exec_eng.display_treeview_nodes(display_variables=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ AC_Sales',
                       '\t\t\t|_ AC1',
                       '\t\t\t|_ AC2',
                       '\t\t\t|_ AC3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC2', 'AC3'],
             'AC_Sales': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ AC_Sales',
                       '\t\t\t|_ AC1',
                       '\t\t\t|_ AC2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_06_build_architecture_scatter_with_multiple_configure(self):

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('scatter', 'AC_list', 'ValueBlockDiscipline')],
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
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t|_ AC_Sales',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # test configure and cleaning

        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Airbus', 'Boeing'],
             'AC_list': ['A320', 'A321', 'B737'],
             'AC_Sales': [True, True, True],
             'Services': [True, False, False]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ A320',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ A320',
                       '\t\t\t\t|_ A321',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ B737', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.AC_list'), ['A320', 'A321', 'B737'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.AC_Sales.AC_list'), ['A320', 'A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.AC_list'), ['A320'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.AC_Sales.AC_list'), ['B737'])

        # test configure and cleaning

        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Airbus', 'Boeing'],
             'AC_list': ['A320', 'A321', 'B737'],
             'AC_Sales': [False, True, True],
             'Services': [True, True, False]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ A320',
                       '\t\t\t\t|_ A321',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ A321',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ B737']
        exp_tv_str = '\n'.join(exp_tv_list)
        self.exec_eng.display_treeview_nodes()

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.AC_list'), ['A320', 'A321', 'B737'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.AC_Sales.AC_list'), ['A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.AC_list'), ['A320', 'A321'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.AC_Sales.AC_list'), ['B737'])

    def test_07_architecture_multi_level(self):

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.repo_business = 'business_case.sos_wrapping.'
        vb_builder_name = 'Business'

        component_sales_architecture_df = pd.DataFrame(
            {'Parent': ['AC_Sales', 'AC_Sales'],
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

        component_oss_architecture_df = pd.DataFrame(
            {'Parent': ['OSS', 'OSS'],
             'Current': ['Airframe', 'Propulsion'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': ['standard', 'standard'],
             'Activation': [False, False]})

        services_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services'],
             'Current': ['FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', component_fhs_architecture_df), ('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', component_oss_architecture_df)],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'AC_Sales', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', component_sales_architecture_df), ('architecture', services_architecture_df), ('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [True, True, False, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.study_name})

        self.exec_eng.configure()

        activation_df = pd.DataFrame({'Business': ['Airbus', 'Airbus', 'Boeing', 'Boeing'],
                                      'AC_list': ['AC1', 'AC2', 'AC3', 'AC4'],
                                      'AC_Sales': [True, True, True, True],
                                      'Services': [True, True, False, False],
                                      'Airframe': [True, True, False, False],
                                      'Propulsion': [True, True, False, False],
                                      'FHS': [True, True, False, False],
                                      'OSS': [True, True, False, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ AC1',
                       '\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ AC2',
                       '\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Propulsion',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ OSS',
                       '\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ Propulsion',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ AC3',
                       '\t\t\t\t|_ AC4', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(list(self.exec_eng.dm.get_value(
            'MyCase.AC_list')), ['AC1', 'AC2', 'AC3', 'AC4'])
        self.assertListEqual(list(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.AC_Sales.AC_list')), ['AC1', 'AC2'])
        self.assertListEqual(list(self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.AC_Sales.AC_list')), ['AC3', 'AC4'])
        self.assertListEqual(list(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.FHS.AC_list')), ['AC1', 'AC2'])
        self.assertListEqual(list(self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.OSS.AC_list')), ['AC1', 'AC2'])

    def test_08_process_simple_architecture(self):

        repo = 'sos_trades_core.sos_processes.test'
        builder = self.exec_eng.factory.get_builder_from_process(
            repo, 'test_architecture')

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

        # check type in usecase
        self.assertListEqual(values_dict['MyCase.Business.activation_df']['AC_list'].apply(
            type).values.tolist(), [str, str, int])

        self.exec_eng.load_study_from_input_dict(values_dict)

        # check type after configure
        self.assertListEqual(self.exec_eng.dm.get_value('MyCase.Business.activation_df')[
                             'AC_list'].apply(type).values.tolist(), [str, str, str])

        activ_df = pd.DataFrame({'Business': ['Airbus', 'Airbus', 'Airbus', 'Boeing', 'Embraer'],
                                 'AC_list': ['A320', 'A321', 'A380',  737, 'E170'],
                                 'AC_Sales': [True, True, True, True, True],
                                 'Services': [True, True, True, True, False],
                                 'FHS': [True, False, True, False, False]})
        values_dict['MyCase.Business.activation_df'] = activ_df

        self.exec_eng.load_study_from_input_dict(values_dict)

        msg_log_error = 'Invalid Value Block Activation Configuration: [\'Embraer\'] in column Business not in *possible values* [\'Airbus\', \'Boeing\']'
        self.assertTrue(msg_log_error in my_handler.msg_list)
        msg_log_error = 'Invalid Value Block Activation Configuration: value block Services not available for [\'Boeing\']'
        self.assertTrue(msg_log_error in my_handler.msg_list)

        activ_df = pd.DataFrame({'Business': ['Airbus', 'Airbus', 'Airbus', 'Boeing'],
                                 'AC_list': ['A320', 'A321', 'A380',  '737'],
                                 'AC_Sales': [True, True, True, True],
                                 'Services': [True, True, True, False],
                                 'FHS': [True, False, True, False]})

        self.assertTrue(activ_df.equals(
            self.exec_eng.dm.get_value('MyCase.Business.activation_df')))

        activ_df = pd.DataFrame({'Business': ['Airbus', 'Airbus', 'Boeing', 'Boeing'],
                                 'AC_list': ['A320', 'A321', 737, 787],
                                 'AC_Sales': [True, True, True, True],
                                 'Services': [True, True, False, False],
                                 'FHS': [True, False, False, False]})
        values_dict['MyCase.Business.activation_df'] = activ_df
        self.exec_eng.load_study_from_input_dict(values_dict)

        self.assertTrue(activ_df.equals(
            self.exec_eng.dm.get_value('MyCase.Business.activation_df')))

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ Business',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ A320',
                       '\t\t\t\t|_ A321',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ A320',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ 737',
                       '\t\t\t\t|_ 787']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        activ_df = pd.DataFrame({'Business': ['Airbus', 'Airbus', 'Boeing'],
                                 'AC_list': ['A320', 'A380', 787],
                                 'AC_Sales': [True, True, True],
                                 'Services': [True, True, False],
                                 'FHS': [True, False, False]})
        values_dict['MyCase.Business.activation_df'] = activ_df
        self.exec_eng.load_study_from_input_dict(values_dict)

        for disc in self.exec_eng.factory.sos_disciplines:
            self.assertEqual(disc.status, 'CONFIGURE')

        self.exec_eng.execute()

        for disc in self.exec_eng.factory.sos_disciplines:
            self.assertEqual(disc.status, 'DONE')

    def test_09_build_scatter_architecture_at_architecture_node(self):

        vb_builder_name = 'Business'

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        actor_map_dict = {'input_name': 'Actor_list',

                          'input_ns': 'ns_public',
                          'output_name': 'Actor_name',
                          'scatter_ns': 'ns_actor'}
        self.exec_eng.smaps_manager.add_build_map('Actor_list', actor_map_dict)

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['Flight Hour'],
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
            {'Actor_list': ['Airbus', 'Boeing', 'Embraer'],
             'Business': [True, True, False],
             'Flight Hour': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Flight Hour',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ Flight Hour']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_10_build_scatter_at_architecture_node(self):

        vb_builder_name = 'Business'

        actor_map_dict = {'input_name': 'Actor_list',

                          'input_ns': 'ns_public',
                          'output_name': 'Actor_name',
                          'scatter_ns': 'ns_actor'}
        self.exec_eng.smaps_manager.add_build_map('Actor_list', actor_map_dict)

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
            {'Actor_list': ['Airbus', 'Boeing', 'Embraer'],
             'Business': [True, True, False]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t|_ Boeing']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_11_build_standard_value_block_at_architecture_node(self):

        vb_builder_name = 'Services'

        architecture_df = pd.DataFrame(
            {'Parent': [None, 'Services', 'Services', 'Services', 'Services'],
             'Current': ['Services', 'OSS', 'FHS', 'Pool', 'TSP'],
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
                       f'\t\t|_ OSS',
                       f'\t\t|_ FHS',
                       f'\t\t|_ Pool',
                       f'\t\t|_ TSP']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_12_architecture_without_archi_name_in_architecture_df(self):

        # add namespaces definition
        self.exec_eng.ns_manager.add_ns_def({'ns_public': self.exec_eng.study_name,
                                             'ns_ac': f'{self.exec_eng.study_name}.Business',
                                             'ns_subsystem': f'{self.exec_eng.study_name}.Business'})

        # actor, subsystem and AC_list scatter maps dict
        subsystem_map = {'input_name': 'subsystem_list',

                         'input_ns': 'ns_subsystem',
                         'output_name': 'subsystem',
                         'scatter_ns': 'ns_subsystem_scatter'}

        ac_list_map = {'input_name': 'AC_list',

                       'input_ns': 'ns_ac',
                       'output_name': 'AC_name',
                       'scatter_ns': 'ns_ac_scatter',
                       'ns_to_update': ['ns_subsystem']}

        # add actor, subsystem and AC_list maps
        self.exec_eng.smaps_manager.add_build_map(
            'subsystem_list_map', subsystem_map)
        self.exec_eng.smaps_manager.add_build_map(
            'AC_list_map', ac_list_map)

        architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Services', 'Services', None],
             'Current': ['OSS', 'FHS', 'Pool', 'TSP', 'Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystem_list', 'ValueBlockDiscipline'))],
             'Activation': [False, False, False, False, False]})

        builder_architecture = self.exec_eng.factory.create_architecture_builder(
            'Business', architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_architecture)
        self.exec_eng.load_study_from_input_dict({})
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Services',
                       f'\t\t\t|_ OSS',
                       f'\t\t\t|_ FHS',
                       f'\t\t\t|_ Pool',
                       f'\t\t\t|_ TSP',
                       f'\t\t|_ Sales']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                      'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                      'OSS': [True, True, True, True],
                                      'FHS': [True, True, True, True],
                                      'Pool': [True, True, True, True],
                                      'TSP': [True, True, True, True],
                                      'Sales': [True, True, True, True]})

        dict_values = {}
        dict_values[f'{self.study_name}.Business.activation_df'] = activation_df

        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Services',
                       f'\t\t\t|_ OSS',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ FHS',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Pool',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ TSP',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t|_ Sales',
                       f'\t\t\t|_ NSA-300',
                       f'\t\t\t\t|_ Airframe',
                       f'\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ NSA-400',
                       f'\t\t\t\t|_ Airframe',
                       f'\t\t\t\t|_ Propulsion']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                      'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                      'OSS': [False, False, True, True],
                                      'FHS': [True, False, True, False],
                                      'Pool': [False, False, False, False],
                                      'TSP': [True, True, False, False],
                                      'Sales': [True, False, False, True]})

        dict_values = {}
        dict_values[f'{self.study_name}.Business.activation_df'] = activation_df

        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Services',
                       f'\t\t\t|_ OSS',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ FHS',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t|_ TSP',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t|_ Sales',
                       f'\t\t\t|_ NSA-300',
                       f'\t\t\t\t|_ Airframe',
                       f'\t\t\t|_ NSA-400',
                       f'\t\t\t\t|_ Propulsion']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                      'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                      'OSS': [False, False, False, False],
                                      'FHS': [False, False, False, False],
                                      'Pool': [False, False, False, False],
                                      'TSP': [False, False, False, False],
                                      'Sales': [False, False, False, False]})

        dict_values = {}
        dict_values[f'{self.study_name}.Business.activation_df'] = activation_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestArchiBuilder()
    cls.setUp()
    cls.test_12_architecture_without_archi_name_in_architecture_df()
