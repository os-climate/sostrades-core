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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- Generate test 1 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Disc3 Very Simple Multi Scenario Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        # scatter build map
        ac_map = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_scenario',
                  'ns_to_update': ['ns_data_ac']}

        self.ee.smaps_manager.add_build_map('name_list', ac_map)

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_disc3', 'ns_barrierr', 'ns_out_disc3']}

        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)

        # shared namespace
        self.ee.ns_manager.add_ns('ns_barrierr', self.ee.study_name)
        self.ee.ns_manager.add_ns(
            'ns_scatter_scenario', f'{self.ee.study_name}.outer_ms')
        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{self.ee.study_name}.outer_ms.Disc3')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{self.ee.study_name}.outer_ms')
        self.ee.ns_manager.add_ns(
            'ns_data_ac', self.ee.study_name)
        # ns_eval_outer = self.ee.ns_manager.add_ns(
        #     'ns_eval', f'{self.ee.study_name}.outer_ms')
        # ns_eval_inner = self.ee.ns_manager.add_ns(
        #     'ns_eval', f'{self.ee.study_name}.outer_ms.subprocess.inner_ms')

        # get builder for disc1
        builder_disc1 = self.ee.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                                 mod_id='test_disc1_scenario')
        # create an inner ms driver for disc1 only in builder_list
        builder_list = self.ee.factory.create_scatter_driver_with_tool(
            'inner_ms', builder_disc1, 'name_list')
        # get builder for disc3 and append to builder_list
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc3_scenario.Disc3'
        disc3_builder = self.ee.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)
        # builder_list[0].associate_namespaces(ns_eval_inner)

        # create an outer ms driver
        multi_scenarios = self.ee.factory.create_scatter_driver_with_tool(
            'outer_ms', builder_list, 'scenario_list')
        # multi_scenarios[0].associate_namespaces(ns_eval_outer)
        return multi_scenarios

        # ns_lower_doe_eval = self.ee.ns_manager.add_ns('ns_doe_eval', f'{self.ee.study_name}.DoEEvalUpper.DoEEvalLower')
        # doe_eval_builder_lower = self.ee.factory.create_evaluator_builder(
        #     'DoEEvalLower', 'doe_eval', builder_list_2)
        # doe_eval_builder_lower.associate_namespaces(ns_lower_doe_eval)
