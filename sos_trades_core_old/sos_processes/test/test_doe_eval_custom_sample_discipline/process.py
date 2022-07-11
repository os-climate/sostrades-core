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
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Process of a discipline that generates a custom sample used in a custom DoE_Eval',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        disc_name_samples_gene = 'Combvec'
        mod_path_samples_gene = 'sostrades_core.sos_wrapping.test_discs.combvec.Combvec'
        sc_name = 'DoE_Eval'
        disc_name = 'Sumstat'
        mod_path = 'sostrades_core.sos_wrapping.test_discs.sum_stat.Sumstat'
        disc_builder_samples_gene = self.ee.factory.get_builder_from_module(
            disc_name_samples_gene, mod_path_samples_gene)
        self.ee.ns_manager.add_ns(
            'ns_doe_eval', f'{self.ee.study_name}.{sc_name}')
        disc_builder = self.ee.factory.get_builder_from_module(
            disc_name, mod_path)
        builder_list = [disc_builder]
        self.ee.ns_manager.add_ns('ns_sum_stat', f'{self.ee.study_name}')
        doe_eval_builder = self.ee.factory.create_evaluator_builder(
            f'{sc_name}', 'doe_eval', builder_list)
        doe_eval_builder_from_combvec = [
            disc_builder_samples_gene, doe_eval_builder]
        return doe_eval_builder_from_combvec
