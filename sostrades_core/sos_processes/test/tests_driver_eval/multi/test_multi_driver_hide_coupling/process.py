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
# -- Generate test 2 process
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Multi Instance Driver Hide Coupling In Driver Option',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        # instantiate factory by getting builder from process
        cls_list = self.ee.factory.get_builder_from_process(repo='sostrades_core.sos_processes.test',
                                                            mod_id='test_disc1_disc2_coupling')

        disc1_builder = cls_list[0]
        disc2_builder = cls_list[1]
        self.ee.ns_manager.add_ns(
            'ns_ac', f'{self.ee.study_name}.Disc1', display_value=f'{self.ee.study_name}.Disc1')

        self.ee.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.ee.study_name}.Disc1')
        self.ee.ns_manager.add_display_ns_to_builder(
            disc2_builder, f'{self.ee.study_name}.Disc2')

        multi_scenarios = self.ee.factory.create_multi_instance_driver(
            'multi_scenarios', cls_list)
        self.ee.ns_manager.add_display_ns_to_builder(
            multi_scenarios[0], f'{self.ee.study_name}')
        return multi_scenarios
