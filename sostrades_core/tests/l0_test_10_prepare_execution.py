'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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

from gemseo.mda.mda_chain import MDAChain

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_discipline import SoSDiscipline

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''


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

        self.assertTrue(isinstance(ee.root_process.discipline_wrapp.discipline, MDAChain))

        for proxy_disc in ee.root_process.proxy_disciplines:
            self.assertTrue(isinstance(proxy_disc.discipline_wrapp.discipline, SoSDiscipline))
            self.assertIn(proxy_disc.discipline_wrapp.discipline,
                          ee.root_process.discipline_wrapp.discipline.disciplines)
            self.assertIn(proxy_disc, ee.root_process.proxy_disciplines)
            self.assertListEqual(proxy_disc.get_input_data_names(numerical_inputs=False),
                                 list(proxy_disc.discipline_wrapp.discipline.input_grammar.names))
            self.assertListEqual(proxy_disc.get_output_data_names(),
                                 list(proxy_disc.discipline_wrapp.discipline.output_grammar.names))
