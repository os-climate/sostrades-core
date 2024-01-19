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
from sostrades_core.execution_engine.scatter_map import ScatterMap


class ScatterMapsManagerException(Exception):
    pass


class ScatterMapsManager:
    '''
    Specification: ScatterMapsManager allows to manager scatter maps for scatter disciplines
    '''
    INPUT_NAME = 'input_name'
    INPUT_TYPE = 'input_type'
    OUTPUT_NAME = 'output_name'
    OUTPUT_TYPE = 'output_type'
    SCATTER_COLUMN_NAME = 'scatter_column_name'
    SCATTER_VAR_NAME = 'scatter_var_name'

    def __init__(self, name, ee):
        '''
        CLass to manage scatter maps
        '''
        self.name = name
        self.ee = ee
        self.build_maps_dict = {}
        self.data_maps_dict = {}
        self.reuse_existing_data_maps = False

    def get_build_map(self, map_name):
        '''
        Get map from build_maps_dict
        '''
        smap = self.build_maps_dict[map_name]

        return smap

    def get_data_map(self, map_name):
        '''
        Get map from data_maps_dict
        '''
        smap = self.data_maps_dict[map_name]

        return smap

    def associate_disc_to_build_map(self, disc):
        '''
        Associate a discipline to its own build map
        '''
        smap = self.get_build_map(disc.map_name)
        smap.add_dependency(disc.disc_id)

    def add_build_map(self, map_name, map_dict):
        '''
        Instantiate build map and add to build_maps_dict
        '''
        if map_name in self.build_maps_dict:
            # If the map does exist with same definition than do nothing
            if self.build_maps_dict[map_name] == map_dict:
                pass
            else:
                raise ScatterMapsManagerException(
                    f'Map {map_name} already defined with another definition')
        else:
            s_map = ScatterMap(self.ee, map_name, map_dict)
            self.build_maps_dict.update({map_name: s_map})

    def add_data_map(self, map_name, map_dict):
        '''
        Instantiate data map and add to data_maps_dict
        '''
        if map_name in self.data_maps_dict:
            # check if parameter reuse existing map is True
            if self.reuse_existing_data_maps:
                # if parameter is set to true, then we return the existing map.
                # It is probably because it is used in a multi scenario of multi_scenario process
                s_map = self.data_maps_dict[map_name]
            else:
                raise ScatterMapsManagerException(
                    f'Map {map_name} already defined')
        else:
            map_dict = self.check_map_parameters(map_dict)
            s_map = ScatterMap(self.ee, map_name, map_dict)
            self.data_maps_dict.update({map_name: s_map})

        return s_map

    def remove_build_map(self, map_name):
        '''
        Remove build map from build_maps_dict
        '''
        if map_name in self.build_maps_dict:
            del self.build_maps_dict[map_name]
        else:
            raise ScatterMapsManagerException(
                f'Trying to remove not existing in build map {map_name}')

    def remove_data_map(self, map_name):
        '''
        Remove data map from data_maps_dict
        '''
        if map_name in self.data_maps_dict:
            del self.data_maps_dict[map_name]
        else:
            raise ScatterMapsManagerException(
                f'Trying to remove not existing in data map {map_name}')

    def check_map_parameters(self, map):
        '''
        Check type and length of map parameters and put lists in input_name, input_type, output_name, output_type 
        '''
        # check type
        if type(map[self.SCATTER_VAR_NAME]) != str:
            raise ScatterMapsManagerException(
                f'{self.SCATTER_VAR_NAME}: {map[self.SCATTER_VAR_NAME]} must be of type string in map: {map}')

        parameters_list = [map[self.INPUT_NAME], map[self.INPUT_TYPE],
                           map[self.OUTPUT_NAME], map[self.OUTPUT_TYPE]]
        if self.SCATTER_COLUMN_NAME in map:
            parameters_list.append(map[self.SCATTER_COLUMN_NAME])

        if not all(type(element) == type(parameters_list[0]) for element in parameters_list):
            raise ScatterMapsManagerException(
                f'{str(parameters_list).strip("[]")} must have same type (list or string) in map: {map}')

        else:
            if any(isinstance(element, list) for element in parameters_list):
                # check length
                if not all(len(element) == len(parameters_list[0]) for element in parameters_list):
                    raise ScatterMapsManagerException(
                        f'{str(parameters_list).strip("[]")} must have same length in map: {map}')

            else:
                # put lists in input_name, input_type, output_name, output_type
                map[self.INPUT_NAME] = [map[self.INPUT_NAME]]
                map[self.INPUT_TYPE] = [map[self.INPUT_TYPE]]
                map[self.OUTPUT_NAME] = [map[self.OUTPUT_NAME]]
                map[self.OUTPUT_TYPE] = [map[self.OUTPUT_TYPE]]
                if self.SCATTER_COLUMN_NAME in map:
                    map[self.SCATTER_COLUMN_NAME] = [
                        map[self.SCATTER_COLUMN_NAME]]

            return map

    def is_input_name_in_build_maps_dict(self, input_name_list):
        '''
        Return True if name matches with an input_name in buil_maps_dict
        '''
        for build_map in self.build_maps_dict.values():
            for input_name in input_name_list:
                if build_map.get_input_name() == input_name:
                    return True
        return False

    def get_input_ns_from_build_map(self, input_name):
        '''
        Get input ns of build map with input_name
        '''
        for build_map in self.build_maps_dict.values():
            if build_map.get_input_name() == input_name:
                return build_map.get_input_ns()

    def get_input_type_from_build_map(self, input_name):
        '''
        Get input type of build map with input_name
        '''
        for build_map in self.build_maps_dict.values():
            if build_map.get_input_name() == input_name:
                return build_map.get_input_type()

    def get_build_map_with_input_name(self, input_name):
        '''
        Get build map with input_name
        '''
        for build_map in self.build_maps_dict.values():
            if build_map.get_input_name() == input_name:
                return build_map
