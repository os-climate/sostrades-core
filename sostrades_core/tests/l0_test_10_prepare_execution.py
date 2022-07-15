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
from sostrades_core.execution_engine.sos_mda_chain import SoSMDAChain
from gemseo.core.discipline import MDODiscipline
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
import time
import _thread

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestPrepareExecution(unittest.TestCase):
    """
    SoSCoupling status test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.repo = 'sostrades_core.sos_processes.test'

    def test_01_prepare_execution(self):

        namespace = 'MyCase'
        ee = ExecutionEngine(namespace)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        ee.configure()
        # check treeview structure
        exp_tv_list = ['Nodes representation for Treeview MyCase',
                       '|_ MyCase',
                       '\t|_ Disc1',
                       '\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == ee.display_treeview_nodes()
        # -- setup inputs
        values_dict = {}
        values_dict[f'{namespace}.Disc2.constant'] = -10.
        values_dict[f'{namespace}.Disc2.power'] = -10.
        values_dict[f'{namespace}.Disc1.a'] = 10.
        values_dict[f'{namespace}.Disc1.b'] = 20.
        values_dict[f'{namespace}.Disc1.indicator'] = 10.
        values_dict[f'{namespace}.x'] = 3.

        ee.load_study_from_input_dict(values_dict)
        
        ee.prepare_execution()
        
        self.assertTrue(isinstance(ee.root_process.mdo_discipline, SoSMDAChain))
        
        for proxy_disc in ee.root_process.proxy_disciplines:
            self.assertTrue(isinstance(proxy_disc.mdo_discipline, MDODiscipline))
            self.assertIn(proxy_disc.mdo_discipline, ee.root_process.sub_mdo_disciplines)
            self.assertTrue(proxy_disc.mdo_discipline.proxy_discipline is proxy_disc)
            self.assertListEqual(ee.root_process.proxy_disciplines[0].get_input_data_names(), ee.root_process.proxy_disciplines[0].mdo_discipline.input_grammar.get_data_names())
            self.assertListEqual(ee.root_process.proxy_disciplines[0].get_output_data_names(), ee.root_process.proxy_disciplines[0].mdo_discipline.output_grammar.get_data_names())
