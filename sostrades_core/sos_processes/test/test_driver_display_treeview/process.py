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
#-- Generate test 2 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Display Treeview Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        my_scatter_dict = {'input_name': 'scenario_list',
                           'input_type': 'string_list',
                           'input_ns': 'ns_eval',
                           'scatter_ns': 'ns_ac', }  # or object ScatterMapBuild
        # >> introduce ScatterMap
        if 'scenario_list' not in self.ee.smaps_manager.build_maps_dict:
            self.ee.smaps_manager.add_build_map(
                'scenario_list', my_scatter_dict)

        # instantiate factory by getting builder from process
        cls_list = self.ee.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                            mod_id='test_disc1_disc2_coupling')

        disc1_builder = cls_list[0]
        disc2_builder = cls_list[1]
        self.ee.ns_manager.add_ns(
            'ns_eval', f'{self.ee.study_name}.multi_scenarios', display_value=f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_ac', f'{self.ee.study_name}.Disc1', display_value=f'{self.ee.study_name}.Disc1')

        self.ee.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.ee.study_name}.Disc1')
        self.ee.ns_manager.add_display_ns_to_builder(
            disc2_builder, f'{self.ee.study_name}.Disc2')

        multi_scenarios = self.ee.factory.create_driver(
            'multi_scenarios', cls_list, map_name='scenario_list', hide_coupling_in_driver=True)
        self.ee.ns_manager.add_display_ns_to_builder(
            multi_scenarios[0], f'{self.ee.study_name}')
        return multi_scenarios
