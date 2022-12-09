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
        'label': 'Core Test (Scatter-Disc1,Disc3) from proc',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        # 1. Define builder list from sub_proc
        repo = 'sostrades_core.sos_processes.test'
        sub_proc = 'test_disc1_disc3_coupling'
        builder_list = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_proc)

        # 2. scatter build map for scatter of Disc1
        map_name = 'name_list'
        input_ns = 'ns_scatter_scenario'
        gather_ns = 'ns_scenario'
        output_name = 'ac_name'
        ac_map = {'input_name': map_name,
                  'input_ns': input_ns,
                  'output_name': output_name,
                  'scatter_ns': 'ns_ac',
                  'gather_ns': gather_ns,
                  'ns_to_update': ['ns_data_ac']}
        if map_name not in self.ee.scattermap_manager.build_maps_dict:
            self.ee.scattermap_manager.add_build_map(map_name, ac_map)

        # shared namespace for scatter
        self.ee.ns_manager.add_ns(
            input_ns, f'{self.ee.study_name}')
        # shared namespace for gather
        self.ee.ns_manager.add_ns(
            gather_ns, f'{self.ee.study_name}')

        # 3. instantiate scatter for Disc1
        scatter_list = self.ee.factory.create_multi_scatter_builder_from_list(
            map_name, builder_list=[builder_list[0]], autogather=False)

        # 4. add Disc3 builder to scatter
        scatter_list.append(builder_list[1])

        return scatter_list
