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
Generate an optimization scenario
"""
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder

class ProcessBuilder(BaseProcessBuilder):
    def get_builders(self):
        '''
        default initialisation test
        '''
        disc_dir = 'sos_trades_core.sos_wrapping.test_discs.disc_mixed_opt.'
        mods_dict = {'DiscMixedOpt': disc_dir + 'DiscMixedOptNonLinearFeasible'}
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_mixed_optim': self.ee.study_name + '.MixedOptimScenario'})
        
        coupling_builder = self.ee.factory.create_builder_coupling("MixedCoupling")
        coupling_builder.set_builder_info('cls_builder', builder_list)
        coupling_builder.set_builder_info('with_data_io', True)
        opt_builder = self.ee.factory.create_optim_builder('MixedOptimScenario', [coupling_builder])
           
        return opt_builder
