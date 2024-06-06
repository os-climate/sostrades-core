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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Process: Same Subprocess inside a process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict = {'proc1.Disc2': disc_dir + 'disc2.Disc2',
                     'proc1.Disc1': disc_dir + 'disc1.Disc1'}
        proc_builder = BaseProcessBuilder(self.ee)
        builder_list = proc_builder.create_builder_list(
            mods_dict, ns_dict={'ns_ac': f'{self.ee.study_name}.proc1'}, associate_namespace=True)

        mods_dict = {'proc2.Disc2': disc_dir + 'disc2.Disc2',
                     'proc2.Disc1': disc_dir + 'disc1.Disc1'}
        builder_list2 = proc_builder.create_builder_list(
            mods_dict, ns_dict={'ns_ac': f'{self.ee.study_name}.proc2'}, associate_namespace=True)

        builder_list.extend(builder_list2)

        return builder_list
