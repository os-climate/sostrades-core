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
        'label': 'Core test Disc1 Disc2 Mono Instance Eval with sample generator Option',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict = {'Disc2': disc_dir + 'disc2.Disc2',
                     'Disc1': disc_dir + 'disc1_doe_eval.Disc1'}
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_ac': self.ee.study_name})

        doe_eval_builder = self.ee.factory.create_mono_instance_driver('Eval', builder_list)

        return doe_eval_builder
