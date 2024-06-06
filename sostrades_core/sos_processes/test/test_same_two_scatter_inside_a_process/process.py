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
from copy import deepcopy

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""

class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Process: two same scatter inside a process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        repo_name = 'sostrades_core.sos_processes.test.tests_driver_eval.multi'
        proc_name = 'test_multi_driver'
        builder_list1 = self.ee.factory.get_builder_from_process(
            repo_name, proc_name)
        builder_list2 = self.ee.factory.get_builder_from_process(
            repo_name, proc_name)
        ns_list_standard = deepcopy(self.ee.ns_manager.ns_list)
        # ns_scatter1 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
        #     'Scatter1', after_name=exec_eng.study_name)
        for builder in builder_list1:
            builder.set_disc_name(f'Scatter1.{builder.sos_name}')
            # builder.associate_namespaces(ns_scatter1)
        # ns_scatter2 = exec_eng.ns_manager.update_namespace_list_with_extra_ns(
        #     'Scatter2', after_name=exec_eng.study_name, namespace_list=ns_list_standard)
        for builder in builder_list2:
            builder.set_disc_name(f'Scatter2.{builder.sos_name}')
            # builder.associate_namespaces(ns_scatter2)
        builder_list1.extend(builder_list2)

        return builder_list1
