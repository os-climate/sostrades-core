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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Test Disc1 Disc3 Multi Instance Driver With Sample Generator',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        repo = 'sostrades_core.sos_processes.test.tests_driver_eval.multi'
        sub_proc = 'test_multi_driver_subprocess_1_3'
        eval_driver = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_proc)

        # add sample generator builder
        mod_generator = 'sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper.SampleGeneratorWrapper'
        generator_builder = self.ee.factory.get_builder_from_module(
            'Sample_Generator', mod_generator)

        return eval_driver + [generator_builder]