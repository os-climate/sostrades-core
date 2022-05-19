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
        # add disciplines Sellar
        mod_path0 = 'sos_trades_core.sos_wrapping.test_discs.sos_Sobieski.SobieskiMission'
        disc_name = 'SobieskiMission'
        disc0_builder = self.ee.factory.get_builder_from_module(disc_name, mod_path0)
        
        mod_path1 = 'sos_trades_core.sos_wrapping.test_discs.sos_Sobieski.SobieskiStructure'
        disc_name = 'SobieskiStructure'
        disc1_builder = self.ee.factory.get_builder_from_module(disc_name, mod_path1)
        
        mod_path2 = 'sos_trades_core.sos_wrapping.test_discs.sos_Sobieski.SobieskiAerodynamics'
        disc_name = 'SobieskiAerodynamics'
        disc2_builder = self.ee.factory.get_builder_from_module(disc_name, mod_path2)
        
        mod_path3 = 'sos_trades_core.sos_wrapping.test_discs.sos_Sobieski.SobieskiPropulsion'
        disc_name = 'SobieskiPropulsion'
        disc3_builder = self.ee.factory.get_builder_from_module(disc_name, mod_path3)
        
        ns_dict = {'ns_OptimSobieski': self.ee.study_name}
        self.ee.ns_manager.add_ns_def(ns_dict)
        
        disc_list=[disc0_builder,disc1_builder,disc2_builder,disc3_builder]
        coupling_builder = self.ee.factory.create_builder_coupling("SobieskyCoupling")
        coupling_builder.set_builder_info('cls_builder', disc_list)
        coupling_builder.set_builder_info('with_data_io', True)
           
        return coupling_builder
