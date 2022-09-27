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
# -- Generate test 2 process

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_processes.test.test_disc1_disc2_doe_eval',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict_2 = {'Disc1': disc_dir + 'disc1.Disc1'}
        builder_list_2 = self.create_builder_list(mods_dict_2, ns_dict={'ns_ac': self.ee.study_name,
                                                                      'ns_doe_eval': f'{self.ee.study_name}.DoEEval1.DoEEval2'})


        doe_eval_builder_2 = self.ee.factory.create_evaluator_builder(
            'DoEEval2', 'doe_eval', builder_list_2)

        # self.ee.ns_manager.update_all_shared_namespaces_by_name(
        #     'Upper', 'ns_doe_eval')
        self.ee.ns_manager.add_ns('ns_doe_eval',f'{self.ee.study_name}.DoEEval1')

        doe_eval_builder = self.ee.factory.create_evaluator_builder(
            'DoEEval1', 'doe_eval', doe_eval_builder_2)

        return doe_eval_builder

