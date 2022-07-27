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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir


class TestVBArchiBuilder(unittest.TestCase):
    """
    Class to test Value Block build in architecture builder
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

    def test_01_configure_data_io_for_vb_discipline(self):

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_business',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        vb_type_list = ['ValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus',  'Boeing', 'Boeing', ],
             'Current': ['Airbus', 'Boeing', 'AC_Sales',  'AC_Sales', 'Services', ],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [True, True, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        activation_df = pd.DataFrame(
            {'Business': ['Airbus',  'Boeing'],
             'AC_Sales': [True, True],
             'Services': [False, True]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()

        # Airbus is a simple value block
        disc_airbus = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus')[0]
        self.maxDiff = None
        data_in_th_list = ['linearization_mode',
                           'cache_type', 'cache_file_path', 'debug_mode', 'AC_Sales.output']
        self.assertListEqual(
            list(disc_airbus._data_in.keys()), data_in_th_list)

        data_out_th_list = ['output_gather']
        self.assertListEqual(
            list(disc_airbus._data_out.keys()), data_out_th_list)

        # Boeing is a sumvalueblock
        disc_boeing = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Boeing')[0]
        self.maxDiff = None
        data_in_th_list = ['linearization_mode', 'cache_type',
                           'cache_file_path', 'debug_mode', 'AC_Sales.output', 'Services.output']
        self.assertListEqual(
            list(disc_boeing._data_in.keys()), data_in_th_list)
        data_out_th_list = ['output_gather', 'output']
        self.assertListEqual(
            list(disc_boeing._data_out.keys()), data_out_th_list)

    def test_02_configure_data_io_for_multiple_vb_discipline(self):

        vb_type_list = ['ValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'AC_Sales', 'AC_Sales', 'Boeing', 'Services', 'Services'],
             'Current': ['Airbus', 'Boeing', 'AC_Sales',  'sale1', 'sale2', 'Services', 'Services1', 'Services2'],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [True, True, False, False, False, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        activation_df = pd.DataFrame(
            {'Business': ['Airbus',  'Boeing'],
             'AC_Sales': [True, True],
             'Services': [False, True]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()

        # Airbus is a simple value block
        disc_airbus = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus')[0]
        self.maxDiff = None
        data_in_th_list = ['linearization_mode',
                           'cache_type', 'cache_file_path', 'debug_mode', 'AC_Sales.output_gather']
        self.assertListEqual(
            list(disc_airbus._data_in.keys()), data_in_th_list)

        data_out_th_list = ['output_gather']
        self.assertListEqual(
            list(disc_airbus._data_out.keys()), data_out_th_list)

        # Boeing is a sumvalueblock
        disc_boeing = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Boeing')[0]
        self.maxDiff = None
        data_in_th_list = ['linearization_mode', 'cache_type',
                           'cache_file_path',  'debug_mode', 'Services.output_gather', 'Services.output']
        self.assertListEqual(
            list(disc_boeing._data_in.keys()), data_in_th_list)
        data_out_th_list = ['output_gather', 'output']
        self.assertListEqual(
            list(disc_boeing._data_out.keys()), data_out_th_list)

    def test_03_run_sum_vb_disciplines(self):

        mydict = {'input_name': 'AC_list',

                  'input_ns': 'ns_business',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        vb_type_list = ['SumValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Airbus', 'Boeing', 'Services'],
             'Current': ['Airbus', 'Boeing', 'AC_Sales', 'Services', 'AC_Sales', 'FHS'],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [True, True, False, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        values_dict = {}

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()
        output_fhs = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.FHS.output')

        output_sales = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.AC_Sales.output')

        output_services_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.output_gather')

        output_services = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Services.output')
        output_airbus_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.output_gather')

        self.assertDictEqual(output_services_gather, {'FHS': output_fhs})
        self.assertDictEqual(output_services,  output_fhs)

        self.assertDictEqual(output_airbus_gather, {'AC_Sales': output_sales,
                                                    'Services.FHS': output_fhs,
                                                    'Services': output_services})


if '__main__' == __name__:
    unittest.main()
