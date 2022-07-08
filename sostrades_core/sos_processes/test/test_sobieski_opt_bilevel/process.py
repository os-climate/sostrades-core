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
                
        ns_dict = {'ns_OptimSobieski': self.ee.study_name+ '.SobOptimScenario'}
        self.ee.ns_manager.add_ns('ns_OptimSobieski', self.ee.study_name + '.SobOptimScenario')
        # add disciplines sos_Sobieski
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.sos_Sobieski.'
        #create sc_struct
        struct_builder= self.ee.factory.get_builder_from_module('struct', disc_dir + 'SobieskiStructure')
        sc_struct = self.ee.factory.create_optim_builder('sc_struct', [struct_builder])
        #create sc_aero
        aero_builder= self.ee.factory.get_builder_from_module('aero', disc_dir + 'SobieskiAerodynamics')
        sc_aero = self.ee.factory.create_optim_builder('sc_aero', [aero_builder])
        #create sc_prop
        prop_builder= self.ee.factory.get_builder_from_module('prop', disc_dir + 'SobieskiPropulsion')
        sc_prop = self.ee.factory.create_optim_builder('sc_prop', [prop_builder])
        
        #create mission 
        mission_builder = self.ee.factory.get_builder_from_module('mission', disc_dir + 'SobieskiMission')
        builder_list=[sc_struct,sc_aero,sc_prop,mission_builder]
        #create bilevel optim
        opt_builder = self.ee.factory.create_optim_builder('SobOptimScenario', builder_list)
           
        return opt_builder
