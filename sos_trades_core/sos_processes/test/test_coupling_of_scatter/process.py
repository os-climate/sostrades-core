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
#-- Generate test 2 process
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder

class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Coupling Of Scatter Process',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        my_namespace = ('ns_barrierr', self.ee.study_name)
    
        my_scatter_dict = {'input_name': 'name_list',

                           'input_ns': my_namespace[0],
                           'output_name': 'ac_name',
                           'scatter_ns': 'ns_ac',
                           'gather_ns': 'ns_barrierr'}  # or object ScatterMapBuild
        # >> introduce ScatterMap
        self.ee.smaps_manager.add_build_map('name_list', my_scatter_dict)
        self.ee.ns_manager.add_ns_def(dict((my_namespace, )))
    
        # instantiate factory by getting builder from process
        cls_list = self.ee.factory.get_builder_from_process(repo='sos_trades_core.sos_processes.test',
                                                       mod_id='test_disc1_disc2_coupling')
        scatter_builder_list = self.ee.factory.create_multi_scatter_builder_from_list('name_list',
                                                                                 cls_list,
                                                                                 autogather=True)
    
        return scatter_builder_list
