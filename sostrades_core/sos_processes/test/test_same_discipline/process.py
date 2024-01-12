'''
Copyright 2022 Airbus SAS
Modifications on 2024/01/11-2024/01/11 Copyright 2024 Capgemini
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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Process: Same Discipline ',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        ns_7 = self.ee.ns_manager.add_ns(
            'ns_protected', f'{self.ee.study_name}.Disc7')
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7_wo_df.Disc7'
        disc7_builder = self.ee.factory.get_builder_from_module(
            'Disc7', mod_list)
        disc7_builder.associate_namespaces(ns_7)

        # for associated namespaces, to add a value to the existing namespace, remove namespace cleaning
        ns_72 = self.ee.ns_manager.add_ns(
            'ns_protected', f'{self.ee.study_name}.Disc72', clean_existing=False)
        disc7_builder2 = self.ee.factory.get_builder_from_module(
            'Disc72', mod_list)
        disc7_builder2.associate_namespaces(ns_72)

        return [disc7_builder, disc7_builder2]
