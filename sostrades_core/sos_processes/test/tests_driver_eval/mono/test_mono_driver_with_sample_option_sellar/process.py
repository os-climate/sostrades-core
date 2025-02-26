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
        'label': 'Core test sellar Mono Instance Eval with sample generator Option',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''Default initialisation test'''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar.'
        mods_dict = {
            'Sellar_Problem': disc_dir + 'SellarProblem',
            'Sellar_2': disc_dir + 'Sellar2',
            'Sellar_1': disc_dir + 'Sellar1',
        }
        builder_list_sellar = self.create_builder_list(
            mods_dict,
            ns_dict={
                'ns_OptimSellar': self.ee.study_name,
            },
        )

        doe_eval_builder = self.ee.factory.create_mono_instance_driver(
            'Eval', builder_list_sellar
        )

        return doe_eval_builder
