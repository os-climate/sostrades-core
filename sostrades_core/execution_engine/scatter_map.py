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

    NS_TO_UPDATE = 'ns_to_update'
    NS_NOT_TO_UPDATE = 'ns_not_to_update'
    SCATTER_LIST_TUPLE = 'scatter_list'
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
        self.check_map(s_map)
        self.__map = s_map
        self.ee = ee
        self.dependency_disc_list = []  # list of dependency disciplines
        self.builder = None  # builder associated to the scatter_map

    def check_map(self, map_dict):
        '''
        Check if the map is valid
        '''

        if self.NS_TO_UPDATE in map_dict and self.NS_NOT_TO_UPDATE in map_dict:
            raise Exception(
                f'The scatter map {self.name} can not have both {self.NS_TO_UPDATE} and {self.NS_NOT_TO_UPDATE} keys')
        if self.NS_TO_UPDATE in map_dict and not isinstance(map_dict[self.NS_TO_UPDATE], list) and any(
                isinstance(val, str) for val in map_dict[self.NS_TO_UPDATE]):
            raise Exception(
                f'The {self.NS_TO_UPDATE} key in scatter map {self.name} must be a string list')
        if self.NS_NOT_TO_UPDATE in map_dict and not isinstance(map_dict[self.NS_NOT_TO_UPDATE], list) and any(
                isinstance(val, str) for val in map_dict[self.NS_NOT_TO_UPDATE]):
            raise Exception(
                f'The {self.NS_NOT_TO_UPDATE} key in scatter map {self.name} must be a string list')
        if self.SCATTER_LIST_TUPLE in map_dict and not isinstance(map_dict[self.SCATTER_LIST_TUPLE], tuple) and any(
                isinstance(val, str) for val in map_dict[self.SCATTER_LIST_TUPLE]):
            raise Exception(
                f'The {self.SCATTER_LIST_TUPLE} key in scatter map {self.name} must be a tuple composed with (scatter_list name,scatter_list namespace)')
    def update_map(self, s_map):
        '''
        Mechanism to update value
        '''
        self.check_map(s_map)
        self.__map = s_map

    def get_map(self):
        '''
        Get the map in the ScatterMap
        '''
        return self.__map

    def configure_map(self, builder):

        self.builder = builder

    def get_ns_to_update(self):
        '''
        Get dependent namespaces if ns_to_update in map
        '''
        if self.NS_TO_UPDATE in self.__map:
            return self.__map[self.NS_TO_UPDATE]
        else:
            return []

    def get_ns_not_to_update(self):
        '''
        Get dependent namespaces if ns_to_update in map
        '''
        if self.NS_NOT_TO_UPDATE in self.__map:
            return self.__map[self.NS_NOT_TO_UPDATE]
        else:
            return []

    def is_ns_to_update_or_not(self):
        '''
        Returns True if ns_to_update is in scatter_map
        Returns False if ns_to_update is in scatter_map
        Returns None if none of them are in scatter_map
        '''
        if self.NS_TO_UPDATE in self.__map:
            return True
        elif self.NS_NOT_TO_UPDATE in self.__map:
            return False
        else:
            return None

    def get_scatter_list_name_and_namespace(self):
        '''
        Get scatter list name and namespaces for scenario_df propagation
        '''
        if self.SCATTER_LIST_TUPLE in self.__map:
            return self.__map[self.SCATTER_LIST_TUPLE]
        else:
            return None

    def get_dependency_disc_list(self):
        '''
        Get the list of disciplines dependent on this scatterMap
        '''
        return self.dependency_disc_list

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
