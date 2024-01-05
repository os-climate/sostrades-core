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
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_mdo_discipline import SoSMDODiscipline
from gemseo.mda.mda_chain import MDAChain

from numpy import array, ones


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

        self.assertTrue(isinstance(ee.root_process.mdo_discipline_wrapp.mdo_discipline, MDAChain))

        for proxy_disc in ee.root_process.proxy_disciplines:
            self.assertTrue(isinstance(proxy_disc.mdo_discipline_wrapp.mdo_discipline, SoSMDODiscipline))
            self.assertIn(proxy_disc.mdo_discipline_wrapp.mdo_discipline,
                          ee.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines)
            self.assertIn(proxy_disc, ee.root_process.proxy_disciplines)
            self.assertListEqual(proxy_disc.get_input_data_names(),
                                 proxy_disc.mdo_discipline_wrapp.mdo_discipline.input_grammar.names)
            self.assertListEqual(proxy_disc.get_output_data_names(),
                                 proxy_disc.mdo_discipline_wrapp.mdo_discipline.output_grammar.names)

    # def test_02_init_execution(self):
    #     '''
    #     test of models initialization in wrapps
    #     '''
    #     disc_name = "KnapsackProblem"
    #     # Knapsack characteristics.
    #     values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
    #     weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
    #     capacity_weight = 269.
    #     n_items = len(values)
    #
    #     values_dict = {
    #         f"{self.name}.{disc_name}.x": ones(n_items, dtype=int),
    #         f"{self.name}.{disc_name}.items_value": values,
    #         f"{self.name}.{disc_name}.items_weight": weights,
    #         f"{self.name}.{disc_name}.capacity_weight": capacity_weight,
    #     }
    #
    #     # Set-up execution engine.
    #     exec_eng = ExecutionEngine(self.name)
    #     mod_id = "sostrades_core.sos_wrapping.test_discs.knapsack.KnapsackProblem"
    #     builder = exec_eng.factory.get_builder_from_module("KnapsackProblem", mod_id)
    #     exec_eng.factory.set_builders_to_coupling_builder(builder)
    #     exec_eng.configure()
    #
    #     # Set-up study.
    #     exec_eng.load_study_from_input_dict(values_dict)
    #     exec_eng.configure()
    #
    #     # Execute.
    #     exec_eng.execute()