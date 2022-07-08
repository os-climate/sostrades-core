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
from tempfile import gettempdir

import pandas as pd

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.execution_engine.sos_discipline_scatter import SoSDisciplineScatter
from sos_trades_core.sos_wrapping.valueblock_discipline import ValueBlockDiscipline


class TestAdvancedArchiBuilder(unittest.TestCase):
    """
    Class to test multi level process built from architecture builder, scatter of architecture and cleaning
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

    def test_01_build_sub_architecture(self):
        vb_builder_name = 'Business'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('architecture', sub_architecture_df), ('standard')],
             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t|_ FHS',
                       '\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t|_ OSS',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_02_build_two_sub_architectures(self):
        vb_builder_name = 'Business'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        sub_architecture_component_df = pd.DataFrame(
            {'Parent': ['AC_Sales', 'AC_Sales', 'Airframe', 'Airframe'],
             'Current': ['Propulsion', 'Airframe', 'Wing', 'VTP'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })
        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing', 'Airbus'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('architecture', sub_architecture_df),
                        ('architecture', sub_architecture_component_df),
                        ('architecture', sub_architecture_component_df)],
             'Activation': [True, True, False, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t|_ FHS',
                       '\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t|_ OSS',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Wing',
                       '\t\t\t\t\t|_ VTP',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       '\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t|_ Wing',
                       '\t\t\t\t\t|_ VTP', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_03_build_sub_sub_architecture(self):
        vb_builder_name = 'Business'

        sub_sub_architecture_df = pd.DataFrame(
            {'Parent': ['OSS', 'OSS', 'OSS level 1', 'OSS level 1'],
             'Current': ['OSS level 1', 'OSS level 2', 'OSS level 11', 'OSS level 12'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('architecture', sub_sub_architecture_df)],
             'Activation': [False, False, False, False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('architecture', sub_architecture_df), ('standard')],
             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        self.exec_eng.load_study_from_input_dict({})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t|_ FHS',
                       '\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t|_ OSS',
                       '\t\t\t\t\t\t|_ OSS level 1',
                       '\t\t\t\t\t\t\t|_ OSS level 11',
                       '\t\t\t\t\t\t\t|_ OSS level 12',
                       '\t\t\t\t\t\t|_ OSS level 2',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_04_build_scatter_architecture(self):
        vb_builder_name = 'Business'

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Airbus', 'Boeing'],
             'Current': ['Airbus', 'Boeing', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'),
                        ('scatter_architecture', 'AC_list',
                         'SumValueBlockDiscipline', sub_architecture_df),
                        ('standard')],

             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        self.exec_eng.configure()
        activation_df = pd.DataFrame(
            {'Business': ['Airbus', 'Airbus', 'Airbus'],
             'AC_list': ['AC1', 'AC2', 'AC3'],
             'Services': [True, True, True]})

        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Services',
                       '\t\t\t\t|_ AC1',
                       '\t\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ AC2',
                       '\t\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ AC3',
                       '\t\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t|_ Maintenance',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t|_ Boeing',
                       '\t\t\t|_ AC_Sales',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertEqual(self.exec_eng.dm.get_disciplines_with_name('MyCase.Business.Airbus.Services')[
                             1].scatter_builders.cls.__name__, 'SumValueBlockDiscipline')
        self.assertEqual(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus.Services.AC1')[0].__class__.__name__, 'SumValueBlockDiscipline')
        self.assertEqual(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus.Services.AC2')[0].__class__.__name__, 'SumValueBlockDiscipline')
        self.assertEqual(self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.Airbus.Services.AC3')[0].__class__.__name__, 'SumValueBlockDiscipline')

    def test_05_build_scatter_of_scatter(self):
        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        mydict = {'input_name': 'component_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'component_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('component_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['AC_Sales'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', ('scatter', 'component_list', 'ValueBlockDiscipline'))],
             'Activation': [False]})

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
             'component_list': [None],
             'AC_Sales': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict())

        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC1', 'AC2', 'AC3'],
             'component_list': ['Propulsion', 'Airframe', 'Airframe', 'Airframe'],
             'AC_Sales': [True, True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        self.exec_eng.display_treeview_nodes(display_variables=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ AC_Sales',
                       '\t\t\t|_ AC1',
                       '\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC2',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC3',
                       '\t\t\t\t|_ Airframe',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        ac1_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales.AC1')

        acsales_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales')

        self.assertListEqual([child.get_disc_full_name()
                              for child in ac1_disciplines[0].children_list],
                             ['MyCase.Business.AC_Sales.AC1.Propulsion',
                              'MyCase.Business.AC_Sales.AC1.Airframe'])
        # We have sumdisciplines but also scatter disciplines (which do not
        # have outputs)
        self.assertListEqual([child.get_disc_full_name()
                              for child in acsales_disciplines[0].children_list], ['MyCase.Business.AC_Sales.AC1',
                                                                                   'MyCase.Business.AC_Sales.AC2',
                                                                                   'MyCase.Business.AC_Sales.AC3',
                                                                                   'MyCase.Business.AC_Sales.AC1',
                                                                                   'MyCase.Business.AC_Sales.AC2',
                                                                                   'MyCase.Business.AC_Sales.AC3'])

        acpropu_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales.AC1.Propulsion')

        self.assertListEqual([child.get_disc_full_name()
                              for child in acpropu_disciplines[0].children_list], [])

    def test_06_build_scatter_of_scatter_with_option(self):
        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        mydict = {'input_name': 'component_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'component_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('component_list', mydict)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['AC_Sales'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', ('scatter', 'component_list', 'ValueBlockDiscipline'),
                         'FakeValueBlockDiscipline')],
             'Activation': [False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

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
             'component_list': [None],
             'AC_Sales': [True]})

        self.assertDictEqual(activation_df.to_dict(), self.exec_eng.dm.get_value(
            'MyCase.Business.activation_df').to_dict())

        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC1', 'AC2', 'AC3'],
             'component_list': ['Propulsion', 'Airframe', 'Airframe', 'Airframe'],
             'AC_Sales': [True, True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        self.exec_eng.display_treeview_nodes(display_variables=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ AC_Sales',
                       '\t\t\t|_ AC1',
                       '\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC2',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC3',
                       '\t\t\t\t|_ Airframe',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        ac1_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales.AC1')

        # check if discipline from first scatter taking in account
        assert len(ac1_disciplines) == 2
        assert isinstance(ac1_disciplines[0], ValueBlockDiscipline)
        assert isinstance(ac1_disciplines[1], SoSDisciplineScatter)

        acsales_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales')

        self.assertListEqual([child.get_disc_full_name()
                              for child in ac1_disciplines[0].children_list],
                             ['MyCase.Business.AC_Sales.AC1.Propulsion',
                              'MyCase.Business.AC_Sales.AC1.Airframe'])
        # We have sumdisciplines but also scatter disciplines (which do not
        # have outputs)
        self.assertListEqual([child.get_disc_full_name()
                              for child in acsales_disciplines[0].children_list], ['MyCase.Business.AC_Sales.AC1',
                                                                                   'MyCase.Business.AC_Sales.AC2',
                                                                                   'MyCase.Business.AC_Sales.AC3',
                                                                                   'MyCase.Business.AC_Sales.AC1',
                                                                                   'MyCase.Business.AC_Sales.AC2',
                                                                                   'MyCase.Business.AC_Sales.AC3'])

        acpropu_disciplines = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.Business.AC_Sales.AC1.Propulsion')

        self.assertListEqual([child.get_disc_full_name()
                              for child in acpropu_disciplines[0].children_list], [])

        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC1', 'AC2', 'AC3', 'AC4'],
             'component_list': ['Propulsion', 'Airframe', 'Airframe', 'Airframe', 'Airframe'],
             'AC_Sales': [True, True, True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ AC_Sales',
                       '\t\t\t|_ AC1',
                       '\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC2',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC3',
                       '\t\t\t\t|_ Airframe',
                       '\t\t\t|_ AC4',
                       '\t\t\t\t|_ Airframe',
                       ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_07_build_scatter_under_sub_architecture(self):
        vb_builder_name = 'Business'

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Airbus'],
             'Current': ['Flight Hour'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['Airbus'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('architecture', sub_architecture_df)],
             'Activation': [False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        # Double configure without ac_list can mess with archi builder
        self.exec_eng.configure()
        activation_df = pd.DataFrame(
            {'AC_list': ['AC1', 'AC2', 'AC3'],
             'Flight Hour': [True, True, True],
             'Airbus': [True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Airbus',
                       '\t\t\t|_ Flight Hour',
                       '\t\t\t\t|_ AC1',
                       '\t\t\t\t|_ AC2',
                       '\t\t\t\t|_ AC3', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()
        print(self.exec_eng.display_treeview_nodes())

    def _test_08_build_scatter_under_scatter_architecture(self):
        '''
        Build scatter under scatter_architecture is not possible yet
        We have to manage properly the namespaces to have this capability
        '''
        vb_builder_name = 'Business'

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_public',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        actor_map_dict = {'input_name': 'Actor_list',
                          'input_type': 'string_list',
                          'input_ns': 'ns_public',
                          'output_name': 'Actor_name',
                          'scatter_ns': 'ns_actor'}
        self.exec_eng.smaps_manager.add_build_map('Actor_list', actor_map_dict)

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Business'],
             'Current': ['Flight Hour'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [False], })

        architecture_df = pd.DataFrame(
            {'Parent': [None],
             'Current': ['Business'],
             'Type': ['SumValueBlockDiscipline'],
             'Action': [
                 ('scatter_architecture', 'Actor_list', 'SumBusinessActorValueBlockDiscipline', sub_architecture_df)],
             'Activation': [False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        self.exec_eng.load_study_from_input_dict({})

        activation_df = pd.DataFrame(
            {'Actor_list': ['Airbus', 'Airbus', 'Boeing'],
             'Business': [True, True, True],
             'AC_list': ['AC1', 'AC2', 'AC3'],
             'Flight Hour': [True, True, True]})
        self.exec_eng.load_study_from_input_dict(
            {'MyCase.Business.activation_df': activation_df})

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ {vb_builder_name}',
                       '\t\t|_ Actors',
                       '\t\t\t|_ Airbus',
                       '\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t|_ AC2',
                       '\t\t\t|_ Boeing',
                       '\t\t\t\t|_ Flight Hour',
                       '\t\t\t\t\t|_ AC3', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_09_build_scatter_of_architecture(self):
        """
        This test aims at proving  the ability to build a scatter of architecture.
        we build a simple architecture under a scatter of actors
        The test also proves the ability of the scatter to clean the architecture afterward

        """
        mydict = {'input_name': 'actors_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_actors',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('actors_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_actors', 'MyCase')

        architecture_name = 'Services'
        architecture_df = pd.DataFrame(
            {'Parent': ['Flight Hour', 'Maintenance', 'Services', 'Services'],
             'Current': ['FHS', 'OSS', 'Flight Hour', 'Maintenance'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, True, True], })

        builder = self.factory.create_architecture_builder(
            architecture_name, architecture_df)

        scatter = self.exec_eng.factory.create_scatter_builder('Business',
                                                               'actors_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        self.exec_eng.configure()

        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Boeing']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("with Boeing and Airbus as actors")
        print(self.exec_eng.display_treeview_nodes())
        initial_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                             f'|_ {self.namespace}',
                             f'\t|_ Business',
                             '\t\t|_ Airbus',
                             '\t\t\t|_ Services',
                             '\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t|_ FHS',
                             '\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t|_ OSS',
                             '\t\t|_ Boeing',
                             '\t\t\t|_ Services',
                             '\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t|_ FHS',
                             '\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t|_ OSS'
                             ]
        initial_tree_view = '\n'.join(initial_tree_view)
        self.assertEqual(initial_tree_view,
                         self.exec_eng.display_treeview_nodes())

        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Embraer']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Boeing and inserting Embraer")
        print(self.exec_eng.display_treeview_nodes())
        second_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                            f'|_ {self.namespace}',
                            f'\t|_ Business',
                            '\t\t|_ Airbus',
                            '\t\t\t|_ Services',
                            '\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t|_ FHS',
                            '\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t|_ OSS',
                            '\t\t|_ Embraer',
                            '\t\t\t|_ Services',
                            '\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t|_ FHS',
                            '\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t|_ OSS'
                            ]
        second_tree_view = '\n'.join(second_tree_view)
        self.assertEqual(second_tree_view,
                         self.exec_eng.display_treeview_nodes())

        dict_values = {self.study_name +
                       '.actors_list': ['Comac', 'ATR']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Embraer and Airbus and inserting Comac and ATR")
        print(self.exec_eng.display_treeview_nodes())
        third_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                           f'|_ {self.namespace}',
                           f'\t|_ Business',
                           '\t\t|_ Comac',
                           '\t\t\t|_ Services',
                           '\t\t\t\t|_ Flight Hour',
                           '\t\t\t\t\t|_ FHS',
                           '\t\t\t\t|_ Maintenance',
                           '\t\t\t\t\t|_ OSS',
                           '\t\t|_ ATR',
                           '\t\t\t|_ Services',
                           '\t\t\t\t|_ Flight Hour',
                           '\t\t\t\t\t|_ FHS',
                           '\t\t\t\t|_ Maintenance',
                           '\t\t\t\t\t|_ OSS'
                           ]
        third_tree_view = '\n'.join(third_tree_view)
        self.assertEqual(
            third_tree_view, self.exec_eng.display_treeview_nodes())

    def test_10_build_scatter_of_architecture_with_sub_architecture(self):
        """
                This test aims at proving  the ability to build a scatter of architecture and sub architecture.
                we build a complex sub architecture of architecture under a scatter of actors
                The test also proves the ability of the scatter to clean the architecture of sub architecture afterward

        """
        mydict = {'input_name': 'actors_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_actors',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('actors_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_actors', 'MyCase')

        vb_builder_name = 'Business'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, False, False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'First_Node', 'Second_Node'],
             'Current': ['First_Node', 'Second_Node', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('architecture', sub_architecture_df), ('standard')],
             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scatter = self.exec_eng.factory.create_scatter_builder('Scatter',
                                                               'actors_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Boeing']}
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.load_study_from_input_dict(dict_values)
        print("with Boeing and Airbus as actors")
        initial_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                             f'|_ {self.namespace}',
                             '\t|_ Scatter',
                             '\t\t|_ Airbus',
                             '\t\t\t|_ Business',
                             '\t\t\t\t|_ First_Node',
                             '\t\t\t\t\t|_ Services',
                             '\t\t\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t\t\t|_ FHS',
                             '\t\t\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t\t\t|_ OSS',
                             '\t\t\t\t|_ Second_Node',
                             '\t\t\t\t\t|_ AC_Sales',
                             '\t\t|_ Boeing',
                             '\t\t\t|_ Business',
                             '\t\t\t\t|_ First_Node',
                             '\t\t\t\t\t|_ Services',
                             '\t\t\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t\t\t|_ FHS',
                             '\t\t\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t\t\t|_ OSS',
                             '\t\t\t\t|_ Second_Node',
                             '\t\t\t\t\t|_ AC_Sales',
                             ]
        initial_tree_view = '\n'.join(initial_tree_view)
        self.assertEqual(initial_tree_view,
                         self.exec_eng.display_treeview_nodes())
        print(self.exec_eng.display_treeview_nodes())
        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Embraer']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Boeing and inserting Embraer")
        second_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                            f'|_ {self.namespace}',
                            '\t|_ Scatter',
                            '\t\t|_ Airbus',
                            '\t\t\t|_ Business',
                            '\t\t\t\t|_ First_Node',
                            '\t\t\t\t\t|_ Services',
                            '\t\t\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t\t\t|_ FHS',
                            '\t\t\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t\t\t|_ OSS',
                            '\t\t\t\t|_ Second_Node',
                            '\t\t\t\t\t|_ AC_Sales',
                            '\t\t|_ Embraer',
                            '\t\t\t|_ Business',
                            '\t\t\t\t|_ First_Node',
                            '\t\t\t\t\t|_ Services',
                            '\t\t\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t\t\t|_ FHS',
                            '\t\t\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t\t\t|_ OSS',
                            '\t\t\t\t|_ Second_Node',
                            '\t\t\t\t\t|_ AC_Sales',
                            ]
        second_tree_view = '\n'.join(second_tree_view)
        self.assertEqual(second_tree_view,
                         self.exec_eng.display_treeview_nodes())
        print(self.exec_eng.display_treeview_nodes())

    def test_11_build_scatter_of_architecture_with_sub_architecture_from_process(self):
        """
                This test aims at proving  the ability to build a scatter of architecture and sub architecture.
                we build a complex sub architecture of architecture under a scatter of actors
                The test also proves the ability of the scatter to clean the architecture of sub architecture afterward

        """
        builder_process = self.exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_scatter_architecture')

        self.exec_eng.factory.set_builders_to_coupling_builder(builder_process)

        self.exec_eng.configure()

        activ_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                 'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                 'OSS': [True, True, True, True],
                                 'FHS': [True, True, True, True],
                                 'Pool': [True, True, True, True],
                                 'TSP': [True, True, True, True],
                                 'Sales': [True, True, True, True]})

        dict_values = {
            f'{self.study_name}.Business.actors_list': ['Airbus', 'Boeing'],
            f'{self.study_name}.Business.Airbus.activation_df': activ_df,
            f'{self.study_name}.Business.Boeing.activation_df': activ_df}

        self.exec_eng.load_study_from_input_dict(dict_values)

        print("with Boeing and Airbus as actors")
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Airbus',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Sales',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t|_ Boeing',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Sales',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        dict_values = {self.study_name +
                       '.Business.actors_list': ['Airbus', 'Embraer']}

        self.exec_eng.load_study_from_input_dict(dict_values)

        print("Deleting Boeing and inserting Embraer")
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Airbus',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Sales',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t|_ Embraer',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t|_ Sales']
        exp_tv_str = '\n'.join(exp_tv_list)
        print(self.exec_eng.display_treeview_nodes())

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        dict_values = {
            f'{self.study_name}.Business.Embraer.activation_df': activ_df}

        self.exec_eng.load_study_from_input_dict(dict_values)

        print("Deleting Boeing and inserting Embraer")
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ Business',
                       f'\t\t|_ Airbus',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Sales',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t|_ Embraer',
                       f'\t\t\t|_ Services',
                       f'\t\t\t\t|_ OSS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ FHS',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ Pool',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ TSP',
                       f'\t\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t|_ Sales',
                       f'\t\t\t\t|_ NSA-300',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion',
                       f'\t\t\t\t|_ NSA-400',
                       f'\t\t\t\t\t|_ Airframe',
                       f'\t\t\t\t\t|_ Propulsion']
        exp_tv_str = '\n'.join(exp_tv_list)
        print(self.exec_eng.display_treeview_nodes())
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_12_build_scatter_of_sub_architecture_and_scatter(self):
        """
                This test aims at proving  the ability to build a scatter of architecture and sub architecture.
                we build a complex sub architecture of architecture under a scatter of actors
                The test also proves the ability of the scatter to clean the architecture of sub architecture afterward

        """
        mydict = {'input_name': 'actors_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_actors',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        dict_sub = {'input_name': 'aircrafts_list',
                    'input_type': 'string_list',
                    'input_ns': 'ns_aircrafts',
                    'output_name': 'ac_name',
                    'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('actors_list', mydict)
        self.exec_eng.smaps_manager.add_build_map('aircrafts_list', dict_sub)
        self.exec_eng.ns_manager.add_ns('ns_actors', 'MyCase')
        self.exec_eng.ns_manager.add_ns('ns_aircrafts', 'MyCase')

        vb_builder_name = 'Business'

        sub_architecture_df = pd.DataFrame(
            {'Parent': ['Services', 'Services', 'Flight Hour', 'Maintenance'],
             'Current': ['Flight Hour', 'Maintenance', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'aircrafts_list', 'ValueBlockDiscipline'),
                        ('standard')],
             'Activation': [False, False, False, False], })

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'First_Node', 'Second_Node'],
             'Current': ['First_Node', 'Second_Node', 'Services', 'AC_Sales'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'ValueBlockDiscipline',
                      'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('architecture', sub_architecture_df), ('standard')],
             'Activation': [True, True, False, False], })

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scatter = self.exec_eng.factory.create_scatter_builder('Scatter',
                                                               'actors_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})
        self.exec_eng.configure()
        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Boeing']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        dict_values = {self.study_name +
                       '.aircrafts_list': ['AC1', 'AC2', 'AC3']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        print("with Boeing and Airbus as actors")
        initial_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                             f'|_ {self.namespace}',
                             '\t|_ Scatter',
                             '\t\t|_ Airbus',
                             '\t\t\t|_ Business',
                             '\t\t\t\t|_ First_Node',
                             '\t\t\t\t\t|_ Services',
                             '\t\t\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t\t\t|_ FHS',
                             '\t\t\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t\t\t|_ OSS',
                             '\t\t\t\t|_ Second_Node',
                             '\t\t\t\t\t|_ AC_Sales',
                             '\t\t|_ Boeing',
                             '\t\t\t|_ Business',
                             '\t\t\t\t|_ First_Node',
                             '\t\t\t\t\t|_ Services',
                             '\t\t\t\t\t\t|_ Flight Hour',
                             '\t\t\t\t\t\t\t|_ FHS',
                             '\t\t\t\t\t\t|_ Maintenance',
                             '\t\t\t\t\t\t\t|_ OSS',
                             '\t\t\t\t|_ Second_Node',
                             '\t\t\t\t\t|_ AC_Sales',
                             ]
        initial_tree_view = '\n'.join(initial_tree_view)
        # self.assertEqual(initial_tree_view,self.exec_eng.display_treeview_nodes())
        print(self.exec_eng.display_treeview_nodes())
        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Embraer']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Boeing and inserting Embraer")
        second_tree_view = [f'Nodes representation for Treeview {self.namespace}',
                            f'|_ {self.namespace}',
                            '\t|_ Scatter',
                            '\t\t|_ Airbus',
                            '\t\t\t|_ Business',
                            '\t\t\t\t|_ First_Node',
                            '\t\t\t\t\t|_ Services',
                            '\t\t\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t\t\t|_ FHS',
                            '\t\t\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t\t\t|_ OSS',
                            '\t\t\t\t|_ Second_Node',
                            '\t\t\t\t\t|_ AC_Sales',
                            '\t\t|_ Embraer',
                            '\t\t\t|_ Business',
                            '\t\t\t\t|_ First_Node',
                            '\t\t\t\t\t|_ Services',
                            '\t\t\t\t\t\t|_ Flight Hour',
                            '\t\t\t\t\t\t\t|_ FHS',
                            '\t\t\t\t\t\t|_ Maintenance',
                            '\t\t\t\t\t\t\t|_ OSS',
                            '\t\t\t\t|_ Second_Node',
                            '\t\t\t\t\t|_ AC_Sales',
                            ]
        second_tree_view = '\n'.join(second_tree_view)
        # self.assertEqual(second_tree_view, self.exec_eng.display_treeview_nodes())
        print(self.exec_eng.display_treeview_nodes())

    def test_13_build_scatter_of_architecture_without_father_and_with_sum_value_block(self):
        """
        This test aims at proving  the ability to build a value block at the actor node of a  scatter of architecture.
        For that purpose, we introduce a root_node in current nodes of architecture_df
        """
        mydict = {'input_name': 'actors_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_actors',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.exec_eng.smaps_manager.add_build_map('actors_list', mydict)
        self.exec_eng.ns_manager.add_ns('ns_actors', 'MyCase')

        architecture_name = 'Business'
        architecture_df = pd.DataFrame(
            {'Parent': ['Flight Hour', 'Maintenance', None, None, None],
             'Current': ['FHS', 'OSS', 'Flight Hour', 'Maintenance', '@root_node@'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'FakeValueBlockDiscipline',
                      'FakeValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('standard'), ('standard'), ('standard')],
             'Activation': [False, False, True, True, False], })

        builder = self.factory.create_architecture_builder(
            architecture_name, architecture_df)
        scatter = self.exec_eng.factory.create_scatter_builder('Business',
                                                               'actors_list', builder)

        self.exec_eng.factory.set_builders_to_coupling_builder(scatter)

        self.exec_eng.ns_manager.add_ns_def({'ns_vbdict': self.study_name,
                                             'ns_public': self.study_name,
                                             'ns_segment_services': self.study_name,
                                             'ns_services': self.study_name,
                                             'ns_services_ac': self.study_name,
                                             'ns_seg': self.study_name,
                                             'ns_ac': self.study_name,
                                             'ns_coc': self.study_name,
                                             'ns_data_ac': self.study_name,
                                             'ns_business_ac': self.study_name,
                                             'ns_rc': self.study_name,
                                             'ns_market': self.study_name,
                                             'ns_market_in': self.study_name,
                                             'ns_business': f'{self.study_name}.Business',
                                             'ns_Airbus': f'{self.study_name}.Business.Airbus',
                                             'ns_Boeing': f'{self.study_name}.Business.Boeing'})

        self.exec_eng.configure()

        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Boeing']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("with Boeing and Airbus as actors")
        print(self.exec_eng.display_treeview_nodes())

        tree_view = [f'Nodes representation for Treeview {self.namespace}',
                     f'|_ {self.namespace}',
                     f'\t|_ Business',
                     '\t\t|_ Airbus',
                     '\t\t\t|_ Flight Hour',
                     '\t\t\t\t|_ FHS',
                     '\t\t\t|_ Maintenance',
                     '\t\t\t\t|_ OSS',
                     '\t\t|_ Boeing',
                     '\t\t\t|_ Flight Hour',
                     '\t\t\t\t|_ FHS',
                     '\t\t\t|_ Maintenance',
                     '\t\t\t\t|_ OSS']
        tree_view = '\n'.join(tree_view)
        self.assertEqual(tree_view,
                         self.exec_eng.display_treeview_nodes())

        self.exec_eng.execute()

        output_flight_hour_airbus = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Flight Hour.output')
        output_flight_hour_boeing = self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.Flight Hour.output')

        output_maintenance_airbus = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Maintenance.output')

        output_maintenance_boeing = self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.Maintenance.output')

        output_airbus_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.output_gather')
        output_boeing_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Boeing.output_gather')

        self.assertDictEqual(output_airbus_gather, {'Flight Hour': output_flight_hour_airbus,
                                                    'Maintenance': output_maintenance_airbus})

        self.assertDictEqual(output_boeing_gather, {'Flight Hour': output_flight_hour_boeing,
                                                    'Maintenance': output_maintenance_boeing})

        dict_values = {self.study_name +
                       '.actors_list': ['Airbus', 'Embraer']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Boeing and inserting Embraer")
        tree_view = [f'Nodes representation for Treeview {self.namespace}',
                     f'|_ {self.namespace}',
                     f'\t|_ Business',
                     '\t\t|_ Airbus',
                     '\t\t\t|_ Flight Hour',
                     '\t\t\t\t|_ FHS',
                     '\t\t\t|_ Maintenance',
                     '\t\t\t\t|_ OSS',
                     '\t\t|_ Embraer',
                     '\t\t\t|_ Flight Hour',
                     '\t\t\t\t|_ FHS',
                     '\t\t\t|_ Maintenance',
                     '\t\t\t\t|_ OSS']
        tree_view = '\n'.join(tree_view)
        self.assertEqual(tree_view,
                         self.exec_eng.display_treeview_nodes())

        self.exec_eng.execute()

        output_flight_hour_airbus = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Flight Hour.output')
        output_flight_hour_embraer = self.exec_eng.dm.get_value(
            'MyCase.Business.Embraer.Flight Hour.output')

        output_maintenance_airbus = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.Maintenance.output')

        output_maintenance_embraer = self.exec_eng.dm.get_value(
            'MyCase.Business.Embraer.Maintenance.output')

        output_airbus_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Airbus.output_gather')
        output_embraer_gather = self.exec_eng.dm.get_value(
            'MyCase.Business.Embraer.output_gather')

        self.assertDictEqual(output_airbus_gather, {'Flight Hour': output_flight_hour_airbus,
                                                    'Maintenance': output_maintenance_airbus})

        self.assertDictEqual(output_embraer_gather, {'Flight Hour': output_flight_hour_embraer,
                                                     'Maintenance': output_maintenance_embraer})

    def test_14_build_scatter_of_sub_architecture_with_process_with_root_node(self):
        """
        In this test, we reproduce the APDS process  with a root_node.
        Since the children_dict is not yet well configured, we just test that
        we are able to reproduce the tree view node and that Core sum value Blocks
        are built at the actor node.

        """

        builder_process = self.exec_eng.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_scatter_architecture_with_root')
        self.exec_eng.factory.set_builders_to_coupling_builder(builder_process)
        self.exec_eng.load_study_from_input_dict({})

        activ_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                 'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                 'OSS': [True, True, True, True],
                                 'FHS': [True, True, True, True],
                                 'Pool': [True, True, True, True],
                                 'TSP': [True, True, True, True],
                                 'Sales': [True, True, True, True]
                                 })

        dict_values = {
            f'{self.study_name}.Business.actors_list': ['Airbus', 'Boeing'],
            f'{self.study_name}.Business.Airbus.activation_df': activ_df,
            f'{self.study_name}.Business.Boeing.activation_df': activ_df}

        self.exec_eng.load_study_from_input_dict(dict_values)

        print("with Boeing and Airbus as actors")
        tree_view = [f'Nodes representation for Treeview {self.namespace}',
                     f'|_ {self.namespace}',
                     f'\t|_ Business',
                     f'\t\t|_ Airbus',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t|_ Sales',
                     f'\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t|_ Boeing',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t|_ Sales',
                     f'\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion']
        tree_view = '\n'.join(tree_view)
        self.assertTrue(
            isinstance(self.exec_eng.dm.get_disciplines_with_name('MyCase.Business.Airbus')[1], ValueBlockDiscipline))
        self.assertTrue(
            isinstance(self.exec_eng.dm.get_disciplines_with_name('MyCase.Business.Boeing')[1], ValueBlockDiscipline))
        assert tree_view == self.exec_eng.display_treeview_nodes()

        dict_values = {self.study_name +
                       '.Business.actors_list': ['Airbus', 'Embraer'],
                       f'{self.study_name}.Business.Airbus.activation_df': activ_df,
                       f'{self.study_name}.Business.Embraer.activation_df': activ_df
                       }
        self.exec_eng.load_study_from_input_dict(dict_values)
        print("Deleting Boeing and inserting Embraer")
        tree_view = [f'Nodes representation for Treeview {self.namespace}',
                     f'|_ {self.namespace}',
                     f'\t|_ Business',
                     f'\t\t|_ Airbus',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t|_ Sales',
                     f'\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t|_ Embraer',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t|_ Sales',
                     f'\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion']
        tree_view = '\n'.join(tree_view)
        assert tree_view == self.exec_eng.display_treeview_nodes()
        self.assertTrue(
            isinstance(self.exec_eng.dm.get_disciplines_with_name('MyCase.Business.Embraer')[1], ValueBlockDiscipline))

        print(" deactivating sales in embraer")

        embraer_activ_df = pd.DataFrame({'AC_list': ['NSA-300', 'NSA-300', 'NSA-400', 'NSA-400'],
                                         'subsystem_list': ['Airframe', 'Propulsion', 'Airframe', 'Propulsion'],
                                         'OSS': [True, True, True, True],
                                         'FHS': [True, True, True, True],
                                         'Pool': [True, True, True, True],
                                         'TSP': [True, True, True, True],
                                         'Sales': [False, False, False, False]
                                         })
        dict_values = {
            f'{self.study_name}.Business.Embraer.activation_df': embraer_activ_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        print(self.exec_eng.display_treeview_nodes())

        tree_view = [f'Nodes representation for Treeview {self.namespace}',
                     f'|_ {self.namespace}',
                     f'\t|_ Business',
                     f'\t\t|_ Airbus',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t|_ Sales',
                     f'\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t|_ Propulsion',
                     f'\t\t|_ Embraer',
                     f'\t\t\t|_ Services',
                     f'\t\t\t\t|_ OSS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ FHS',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ Pool',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t|_ TSP',
                     f'\t\t\t\t\t|_ NSA-300',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion',
                     f'\t\t\t\t\t|_ NSA-400',
                     f'\t\t\t\t\t\t|_ Airframe',
                     f'\t\t\t\t\t\t|_ Propulsion']
        tree_view = '\n'.join(tree_view)
        assert tree_view == self.exec_eng.display_treeview_nodes()

        # The value Block at node Embraer doesn't have children anymore
        assert (self.exec_eng.dm.get_disciplines_with_name('MyCase.Business.Embraer')[1]).children_list == []
