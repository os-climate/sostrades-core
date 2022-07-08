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
Generate a doe scenario
"""
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder

class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_processes.test.test_DiscAllTypes_doe',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.disc_all_types.'
        mods_dict = {'DiscAllTypes': disc_dir + 'DiscAllTypes'}
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_test': self.ee.study_name + '.DiscAllTypesDoeScenario'})
        doe_builder = self.ee.factory.create_doe_builder('DiscAllTypesDoeScenario', builder_list)
           
        return doe_builder
