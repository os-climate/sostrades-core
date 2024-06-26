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
        'label': 'Core Test Sellar Opt with Func Manager Faulty',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        '''
        default initialisation test
        '''
        # add disciplines Sellar
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sellar.'

        mod_func = 'sostrades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
    
        mods_dict = {'Sellar_Problem': disc_dir + 'SellarProblem',
                     'Sellar_1': disc_dir + 'Sellar1',
                     'Sellar_3': disc_dir + 'Sellar3',
                     'FunctionManager': mod_func}
    
        ns_dict = {'ns_functions': self.ee.study_name + '.' + 'SellarOptimScenario.SellarCoupling',
                   'ns_optim': self.ee.study_name + '.' + 'SellarOptimScenario.SellarCoupling',
                   'ns_OptimSellar': self.ee.study_name + '.SellarOptimScenario.SellarCoupling'}
        builder_list = self.create_builder_list(mods_dict, ns_dict=ns_dict)
        coupling_builder = self.ee.factory.create_builder_coupling("SellarCoupling")
        coupling_builder.set_builder_info('cls_builder', builder_list)
        #coupling_builder.set_builder_info('with_data_io', True)
        opt_builder = self.ee.factory.create_optim_builder(
            'SellarOptimScenario', [coupling_builder])
    
        return opt_builder