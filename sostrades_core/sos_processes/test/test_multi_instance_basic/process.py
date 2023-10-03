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
        'label': 'Test Multi Instance Basic (DriverEvaluator)',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        # simple 2-disc process NOT USING nested scatters

        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_ac', f'{self.ee.study_name}')
        self.ee.ns_manager.add_ns(
            'ns_data_ac', f'{self.ee.study_name}')

        # instantiate factory # get instantiator from Discipline class
        mod_list1 = 'sostrades_core.sos_wrapping.test_discs.disc1_scenario.Disc1'
        disc1_builder = self.ee.factory.get_builder_from_module(
            'Disc1', mod_list1)

        mod_list3 = 'sostrades_core.sos_wrapping.test_discs.disc3_scenario.Disc3'
        disc3_builder = self.ee.factory.get_builder_from_module(
            'Disc3', mod_list3)

        builder_list = [disc1_builder, disc3_builder]

        # multi scenario driver builder
        multi_scenarios = self.ee.factory.create_multi_instance_driver(
            'multi_scenarios', builder_list)

        return multi_scenarios
