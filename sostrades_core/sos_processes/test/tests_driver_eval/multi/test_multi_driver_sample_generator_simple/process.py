'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2024/05/16 Copyright 2023 Capgemini

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
        'label': 'Test Multi Instance With Sample Generator (DriverEvaluator)',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        # simple 2-disc process NOT USING nested scatters coupled with a SampleGenerator

        # multi scenario driver builder
        repo_name = "sostrades_core.sos_processes.test.tests_driver_eval.multi"
        proc_name = "test_multi_driver_simple"
        multi_scenarios = self.ee.factory.get_builder_from_process(repo=repo_name,
                                                                   mod_id=proc_name)

        return multi_scenarios
