'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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
# -- Generate test 1 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Multi Instance Nested (DriverEvaluator)',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        # get builder for disc1
        builder_disc1 = self.ee.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                                 mod_id='test_disc1_scenario')
        
        builder_list = self.ee.factory.create_multi_instance_driver('inner_ms', builder_disc1)
        
        # get builder for disc3 and append to builder_list
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc3_scenario.Disc3'
        disc3_builder = self.ee.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)
        # shared namespace
        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{self.ee.study_name}')

        # self.ee.scattermap_manager.add_build_map('outer_map'
        #                                          , {'ns_to_update': ['ns_driver']})
        # create an outer ms driver
        multi_scenarios = self.ee.factory.create_multi_instance_driver('outer_ms', builder_list)
        return multi_scenarios

        # ns_lower_doe_eval = self.ee.ns_manager.add_ns('ns_doe_eval', f'{self.ee.study_name}.DoEEvalUpper.DoEEvalLower')
        # doe_eval_builder_lower = self.ee.factory.create_evaluator_builder(
        #     'DoEEvalLower', 'doe_eval', builder_list_2)
        # doe_eval_builder_lower.associate_namespaces(ns_lower_doe_eval)
