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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

class ScatterMap:
    '''
    Specification: ScatterMap class allows to define parameters to build Scatter and Gather disciplines
    '''
    INPUT_NAME = 'input_name'
    INPUT_TYPE = 'input_type'
    INPUT_NS = 'input_ns'
    OUTPUT_NAME = 'output_name'
    OUTPUT_TYPE = 'output_type'
    OUTPUT_NS = 'output_ns'
    SCATTER_VAR_NAME = 'scatter_var_name'
    SCATTER_COLUMN_NAME = 'scatter_column_name'
    SCATTER_NS = 'scatter_ns'
    GATHER_NS = 'gather_ns'
    GATHER_NS_IN = 'gather_ns_in'
    GATHER_NS_OUT = 'gather_ns_out'
    NS_TO_UPDATE = 'ns_to_update'
    NS_TO_UPDATE_WITH_ACTOR = 'ns_to_update_with_actor'
    ASSOCIATED_INPUTS = 'associated_inputs'
    DEFAULT = 'default'

    def __init__(self, ee, name, s_map):
        '''
        Class to describe a scatter map and manage several instances of the same scatter map
            :params: name 
            :type: string

            :params: s_map 
            :type: dict 

            if s_map is a scatter build map:
            :keys: input_name, input_type, input_ns, output_name, scatter_ns, gather_ns (optional,=input_ns by default), ns_to_update (optional)
                    gather_ns_in (optional,=gather_ns by default),gather_ns_out (optional,=gather_ns by default)

            if s_map is a scatter data map:
            :keys: input_name, input_type, input_ns, output_name, output_type, scatter_var_name, scatter_column_name

            scatter_column_name is only necessary when the input_type is a dataframe
        '''
        self.name = name
        self.map = s_map
        self.ee = ee
        self.dependency_disc_list = []  # list of dependency disciplines
        self.builder = None  # builder associated to the scatter_map

    def update_map(self, s_map):
        '''
        Mechanism to update value
        '''
        self.map = s_map

    def get_map(self):
        '''
        Get the map in the ScatterMap
        '''
        return self.map

    def configure_map(self, builder):

        self.builder = builder

    def modify_scatter_ns(self, builder_name, scatter_name, local_namespace):
        '''
        Modify disc scatter_ns value and add it to ns_manager 
        '''

        if not isinstance(self.builder, list):
            base_namespace = local_namespace.replace(
                f'.{builder_name}', '')
        else:
            base_namespace = local_namespace

        self.ee.ns_manager.add_ns(
            self.get_scatter_ns(), f'{base_namespace}.{scatter_name}')

    def update_ns(self, old_ns, name, after_name):
        '''
        Create namespaces for ns_to_update and add to shared_ns_dict
        '''
        for ns_name in old_ns:
            updated_value = self.ee.ns_manager.update_ns_value_with_extra_ns(
                old_ns[ns_name].get_value(), name, after_name=after_name)
            self.ee.ns_manager.add_ns(
                ns_name, updated_value)

    def get_input_name(self):
        '''
        Get the input_name in the map
        '''
        return self.map[self.INPUT_NAME]

    def get_input_type(self):
        '''
        Get the input_type in the map
        '''
        return self.map[self.INPUT_TYPE]

    def get_input_ns(self):
        '''
        Get the input_ns in the map of output_ns
        '''
        if self.INPUT_NS in self.map:
            return self.map[self.INPUT_NS]
        else:
            return self.map[self.OUTPUT_NS]

    def get_output_name(self):
        '''
        Get the output_name in the map
        '''
        return self.map[self.OUTPUT_NAME]

    def get_output_type(self):
        '''
        Get the output_type in the map
        '''
        return self.map[self.OUTPUT_TYPE]

    def get_output_ns(self):
        '''
        Get the output_ns in the map
        '''
        if self.OUTPUT_NS in self.map:
            return self.map[self.OUTPUT_NS]
        else:
            return self.map[self.INPUT_NS]

    def get_scatter_var_name(self):
        '''
        Get the scatter_var_name in the map (input_name of the scatter build map associated)
        '''
        return self.map[self.SCATTER_VAR_NAME]

    def get_scatter_column_name(self):
        '''
        Get the scatter_column_name in the map (input_name of the scatter build map associated). Only necessary when input_type is a dataframe
        '''
        if self.SCATTER_COLUMN_NAME in self.map:
            return self.map[self.SCATTER_COLUMN_NAME]
        else:
            return []

    def get_scatter_ns(self):
        '''
        Get the scatter namespace name in the map (namespace associated to the map)
        '''
        if self.SCATTER_NS in self.map:
            return self.map[self.SCATTER_NS]
        else:
            return None

    def get_gather_ns(self):
        '''
        Get the gather namespace name if gather_ns in map
        '''
        if self.GATHER_NS in self.map:
            return self.map[self.GATHER_NS]
        else:
            return self.map[self.INPUT_NS]

    def get_gather_ns_in(self):
        '''
        Get the gather input namespace name if gather_ns_in in map else return ns_gather
        '''
        if self.GATHER_NS_IN in self.map:
            return self.map[self.GATHER_NS_IN]
        else:
            return self.get_gather_ns()

    def get_gather_ns_out(self):
        '''
        Get the gather output namespace name if gather_ns in map else return ns_gather
        '''
        if self.GATHER_NS_OUT in self.map:
            return self.map[self.GATHER_NS_OUT]
        else:
            return self.get_gather_ns()

    def get_ns_to_update(self):
        '''
        Get dependent namespaces if ns_to_update in map
        '''
        if self.NS_TO_UPDATE in self.map:
            return self.map[self.NS_TO_UPDATE]
        else:
            return []

    def get_ns_to_update_with_actor(self):
        '''
        Get dependent namespaces if ns_to_update in map
        '''
        if self.NS_TO_UPDATE_WITH_ACTOR in self.map:
            return self.map[self.NS_TO_UPDATE_WITH_ACTOR]
        else:
            return []

    def get_dependency_disc_list(self):
        '''
        Get the list of disciplines dependent on this scatterMap
        '''
        return self.dependency_disc_list

    def get_associated_inputs(self):
        '''
        Get the associated_inputs on this scatterMap
        '''
        if self.ASSOCIATED_INPUTS in self.map:
            return self.map[self.ASSOCIATED_INPUTS]
        else:
            return []

    def get_default_input(self):
        '''
        Get default input value on this scatterMap
        '''
        if self.DEFAULT in self.map:
            return self.map[self.DEFAULT]
        else:
            return None

    def add_dependency(self, disc):
        '''
        Add scatter_map disciplinary dependency
        '''
        if disc not in self.dependency_disc_list:
            self.dependency_disc_list.append(disc)

    def remove_dependency(self, disc):
        '''
        Remove disciplinary dependency
        '''
        if disc in list(self.dependency_disc_list):
            self.dependency_disc_list.remove(disc)
