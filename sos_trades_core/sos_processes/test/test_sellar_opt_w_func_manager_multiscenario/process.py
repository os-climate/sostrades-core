# pylint: skip-file
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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar

        # builders_list = builders.get()
        # opt_builder = self.ee.factory.create_optim_builder(
        #     'SellarOptimScenario', builders.get())

        builder_list = self.ee.factory.get_builder_from_process(repo='sos_trades_core.sos_processes.test',
                                                                mod_id='test_sellar_opt_w_func_manager')

        # builders_list.append(opt_builder)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_functions', 'ns_barrierrr', 'ns_optim', 'ns_OptimSellar']}

        self.ee.smaps_manager.add_build_map('scenario_list', scenario_map)
        #repo_ = 'sos_trades_core.sos_processes.test'

        # builder_list = self.ee.factory.get_builder_from_process(repo=repo_,
        #                                                  mod_id='test_sellar_opt_w_func_manager')
        scatter_scenario_name = 'multi_scenarios'
        self.ee.ns_manager.add_ns('ns_barrierrr', self.ee.study_name)
        for ns in self.ee.ns_manager.ns_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, scatter_scenario_name, after_name=self.ee.study_name)
            if ns.name not in ['ns_functions', 'ns_barrierrr', 'ns_public', 'ns_optim', 'ns_OptimSellar']:
                self.ee.ns_manager.update_namespace_with_extra_ns(
                    ns, after_name=scatter_scenario_name)
        # Add new namespaces needed for the scatter multiscenario
        ns_dict = {
            'ns_scatter_scenario': f'{self.ee.study_name}.{scatter_scenario_name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)
        self.ee.ns_manager.add_ns_def(ns_dict)
        gather_node_name = 'Post-processing'

        multi_scenarios = self.ee.factory.create_multi_scenario_builder(
            scatter_scenario_name, 'scenario_list', [builder_list], autogather=False, gather_node=gather_node_name, business_post_proc=False)

        return multi_scenarios
