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
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Generate an optimization scenario
"""


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Sellar Opt Discopt Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        default initialisation test
        '''
        # add coupling Sellar
        mda_builder = self.ee.factory.get_builder_from_process(
            'sostrades_core.sos_processes.test', 'test_sellar_coupling')

        ns_dict = {'ns_OptimSellar': self.ee.study_name + '.SellarOptimScenario.SellarCoupling'}
        self.ee.ns_manager.add_ns_def(ns_dict)
        opt_builder = self.ee.factory.create_optim_builder('SellarOptimScenario', [mda_builder])

        return opt_builder
