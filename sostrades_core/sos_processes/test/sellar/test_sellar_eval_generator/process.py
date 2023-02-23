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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    # ontology information
    _ontology_data = {
        'label': 'Core Test Sellar Eval Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        if 1 == 1:  # It is as it should be: the else part should be removed
            # Select the nested subprocess
            repo = 'sostrades_core.sos_processes.test.sellar'
            sub_proc = 'test_sellar_eval_simple'
            eval_driver = self.ee.factory.get_builder_from_process(
                repo=repo, mod_id=sub_proc)
        else:
            # Select the nested subprocess
            repo = 'sostrades_core.sos_processes.test.sellar'
            sub_proc = 'test_sellar_list'
            coupling_builder = self.ee.factory.get_builder_from_process(
                repo=repo, mod_id=sub_proc)

            # driver builder
            flatten_subprocess = False
            if flatten_subprocess:
                eval_driver = self.ee.factory.create_driver(
                    'Eval', coupling_builder, flatten_subprocess=flatten_subprocess)
            else:
                eval_driver = self.ee.factory.create_driver(
                    'Eval', coupling_builder, flatten_subprocess=flatten_subprocess)

            # shift nested subprocess namespaces
            # no need to shift

            # driver namespaces
            self.ee.ns_manager.add_ns(
                'ns_sampling', f'{self.ee.study_name}.Eval')
            self.ee.ns_manager.add_ns('ns_eval', f'{self.ee.study_name}.Eval')

        # sample generator builder
        mod_generator = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        generator_builder = self.ee.factory.get_builder_from_module(
            'SampleGenerator', mod_generator)

        return eval_driver + [generator_builder]
