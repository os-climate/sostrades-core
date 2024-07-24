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
# -- Generate test 1 process
import os

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        mods_dict = {
            'Disc1': 'sostrades_core.sos_wrapping.test_discs.disc1_two_ns.Disc1'}
        builder_list = self.create_builder_list(mods_dict)
        db1_infos = {'database_label': 'db1', 'database_query': {'id': 'Disc1'}}
        database_infos = {'shared_ns': {f'ns_a__{self.ee.study_name}': db1_infos}, 'local_ns': {f'{self.ee.study_name}.Disc1': db1_infos}}
        self.ee.ns_manager.add_ns_def(ns_info={'ns_a': self.ee.study_name, 'ns_b': self.ee.study_name})
        db_conf_path = os.environ.get('SOS_TRADES_MONGODB_CONFIGURATION', None)

        self.ee.ns_manager.set_database_conf_path(db_conf_path)
        self.ee.ns_manager.set_db_infos_to_ns(database_infos)
        return builder_list
