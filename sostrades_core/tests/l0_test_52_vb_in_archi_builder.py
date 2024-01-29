'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/25-2023/11/03 Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
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
        vb_type_list = ['ValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Tomato', 'Tomato', ],
             'Current': ['Remy', 'Tomato', 'CAPEX', 'CAPEX', 'OPEX', ],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [True, True, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Tomato'],
             'CAPEX': [True, True],
             'OPEX': [False, True]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()

        # Remy is a simple value block
        disc_Remy = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Remy')[0]
        self.maxDiff = None
        data_in_th_list = set(['linearization_mode',
                               'cache_type', 'cache_file_path', 'debug_mode', ('output', 'CAPEX')])
        self.assertTrue(
            set(disc_Remy.get_data_in().keys()) == data_in_th_list)

        data_out_th_list = set(['output_gather'])
        self.assertTrue(
            set(disc_Remy.get_data_out().keys()) == data_out_th_list)

        # Tomato is a sumvalueblock
        disc_Tomato = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Tomato')[0]
        self.maxDiff = None
        data_in_th_list = set(['linearization_mode', 'cache_type',
                               'cache_file_path', 'debug_mode', ('output', 'CAPEX'), ('output', 'OPEX')])
        self.assertTrue(
            set(disc_Tomato.get_data_in().keys()) == data_in_th_list)
        data_out_th_list = set(['output_gather', 'output'])
        self.assertTrue(
            set(disc_Tomato.get_data_out().keys()) == data_out_th_list)

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
            {'Parent': ['Business', 'Business', 'Remy', 'CAPEX', 'CAPEX', 'Tomato', 'OPEX', 'OPEX'],
             'Current': ['Remy', 'Tomato', 'CAPEX', 'CAPEX1', 'CAPEX2', 'OPEX', 'OPEX1', 'OPEX2'],
             'Type': vb_type_list,
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard'), ('standard'),
                        ('standard'), ('standard')],
             'Activation': [True, True, False, False, False, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.configure()

        activation_df = pd.DataFrame(
            {'Business': ['Remy', 'Tomato'],
             'CAPEX': [True, True],
             'OPEX': [False, True]})

        values_dict = {
            f'{self.study_name}.Business.activation_df': activation_df}

        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()

        # Remy is a simple value block
        disc_Remy = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Remy')[0]
        self.maxDiff = None
        data_in_th_list = set(['linearization_mode',
                               'cache_type', 'cache_file_path', 'debug_mode', ('output_gather', 'CAPEX')])
        self.assertTrue(
            set(disc_Remy.get_data_in().keys()) == data_in_th_list)

        data_out_th_list = set(['output_gather'])
        self.assertTrue(
            set(disc_Remy.get_data_out().keys()) == data_out_th_list)

        # Tomato is a sumvalueblock
        disc_Tomato = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Tomato')[0]
        self.maxDiff = None
        data_in_th_list = set(['linearization_mode', 'cache_type',
                               'cache_file_path', 'debug_mode', ('output_gather', 'OPEX'), ('output', 'OPEX')])
        self.assertTrue(
            set(disc_Tomato.get_data_in().keys()) == data_in_th_list)
        data_out_th_list = set(['output_gather', 'output'])
        self.assertTrue(
            set(disc_Tomato.get_data_out().keys()) == data_out_th_list)

    def test_03_run_sum_vb_disciplines(self):
        vb_type_list = ['SumValueBlockDiscipline',
                        'ValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'SumValueBlockDiscipline',
                        'FakeValueBlockDiscipline',
                        'FakeValueBlockDiscipline']
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Tomato', 'OPEX'],
             'Current': ['Remy', 'Tomato', 'CAPEX', 'OPEX', 'CAPEX', 'Manhour'],
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
        output_Manhour = self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.OPEX.Manhour.output')

        output_sales = self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.CAPEX.output')

        output_OPEX_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.OPEX.output_gather')

        output_OPEX = self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.OPEX.output')
        output_Remy_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Remy.output_gather')

        self.assertDictEqual(output_OPEX_gather, {'Manhour': output_Manhour})
        self.assertDictEqual(output_OPEX, output_Manhour)

        self.assertDictEqual(output_Remy_gather, {'CAPEX': output_sales,
                                                  'Manhour': output_Manhour,
                                                  'OPEX': output_OPEX})


if '__main__' == __name__:
    cls = TestVBArchiBuilder()
    cls.setUp()
    cls.test_01_configure_data_io_for_vb_discipline()
