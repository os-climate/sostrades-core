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
        'label': 'Core Test Sellar Coupling Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''Default initialisation test'''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar.'
        mods_dict = {'Sellar_Problem': disc_dir + 'SellarProblem',
                     'Sellar_2': disc_dir + 'Sellar2',
                     'Sellar_1': disc_dir + 'Sellar1'}

        builder__list = self.create_builder_list(
            mods_dict, ns_dict={'ns_OptimSellar': self.ee.study_name + '.SellarCoupling'})
        coupling_builder = self.ee.factory.create_builder_coupling(
            "SellarCoupling")
        coupling_builder.set_builder_info('cls_builder', builder__list)

        return coupling_builder
