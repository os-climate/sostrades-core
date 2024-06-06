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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- Generate test 2 process

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test (Disc1_scenario, Disc2_scenario) Coupling Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        # 1. instantiate builder of Disc1
        mod_path = 'sostrades_core.sos_wrapping.test_discs.disc1_scenario.Disc1'
        disc_name = 'Disc1'
        disc1_builder = self.ee.factory.get_builder_from_module(
            disc_name, mod_path)
        self.ee.ns_manager.add_ns(
            'ns_ac', f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_data_ac', f'{self.ee.study_name}')
        # 2. instantiate builder of Disc3
        mod_path = 'sostrades_core.sos_wrapping.test_discs.disc3_scenario.Disc3'
        disc_name = 'Disc3'
        disc3_builder = self.ee.factory.get_builder_from_module(
            disc_name, mod_path)
        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{self.ee.study_name}.Disc3')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{self.ee.study_name}')
        # 3. create builder list
        builder_list = [disc1_builder, disc3_builder]

        return builder_list
