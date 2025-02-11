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
# -- Generate test 2 process

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Disc2 Coupling Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict = {'Disc1': disc_dir + 'disc1.Disc1',
                     'Disc2': disc_dir + 'disc2.Disc2', }
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_ac': self.ee.study_name})

        lower_coupling_builder = self.ee.factory.create_builder_coupling("LowerCoupling")
        lower_coupling_builder.set_builder_info('cls_builder', builder_list)

        upper_coupling_builder = self.ee.factory.create_builder_coupling("UpperCoupling")
        upper_coupling_builder.set_builder_info('cls_builder', [lower_coupling_builder])

        return upper_coupling_builder
