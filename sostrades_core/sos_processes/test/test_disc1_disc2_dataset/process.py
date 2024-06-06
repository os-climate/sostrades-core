'''
Copyright 2023 Capgemini

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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Dataset with 2 simple Disciplines',
        'description': 'Test process with 2 disciplines for dataset dev',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mods_dict = {
            'Disc1': 'sostrades_core.sos_wrapping.test_discs.disc1_disc2_dataset.Disc1',
            'Disc2': 'sostrades_core.sos_wrapping.test_discs.disc1_disc2_dataset.Disc2'}
        
        builder_list = self.create_builder_list(mods_dict)

        self.ee.ns_manager.add_ns_def(ns_info = {'ns_a': self.ee.study_name, 
                                                 'ns_xy_disc1': self.ee.study_name + '.Disc1VirtualNode',
                                                 'ns_xy_disc2': self.ee.study_name + '.Disc2VirtualNode'})

        return builder_list
