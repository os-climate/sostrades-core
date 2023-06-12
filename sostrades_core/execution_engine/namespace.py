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


class Namespace:
    '''
    Specification: Namespace class describes name, value and dependencies of namespace object
    '''
    NS_NAME_SEPARATOR = '__'

    def __init__(self, name, value, display_value=None, database_infos=None):
        '''
        Class to describe a namespace and manage several instance of the same namespace
        '''
        self.name = name
        self.value = value
        self.__display_value = display_value
        self.dependency_disc_list = []  # list of dependency disciplines
        self.database_infos = database_infos 

    def to_dict(self):
        ''' Method that serialize as dict a Namespace object '''
        return self.__dict__

    def update_value(self, val):
        '''
        Mechanism to update value
        '''
        self.value = val

    def get_value(self):
        '''
        Get the value in the Namespace
        '''
        return self.value

    def get_display_value(self):
        '''
        Get the display value in the Namespace if NOne return value
        '''
        if self.__display_value is None:
            return self.value
        else:
            return self.__display_value

    def get_display_value_if_exists(self):
        '''
        Get the display value in the Namespace if None return NOne
        '''

        return self.__display_value
    def is_display_value(self):

        return self.__display_value is not None

    def set_display_value(self, val):
        '''
        Set the display value in the Namespace
        '''
        self.__display_value = val

    def get_ns_id(self):
        '''
        Get the namespace id used to store the namespace in the namespace_manager
        '''
        return f'{self.name}{self.NS_NAME_SEPARATOR}{self.value}'

    def get_dependency_disc_list(self):
        '''
        Get the list of disciplines which use the namespace
        '''
        return self.dependency_disc_list

    def add_dependency(self, disc_id):
        '''
        Add namespace disciplinary dependency
        '''
        if disc_id not in self.dependency_disc_list:
            self.dependency_disc_list.append(disc_id)

    def remove_dependency(self, disc_id):
        '''
        Remove disciplinary dependency
        '''
        if disc_id in self.dependency_disc_list:
            self.dependency_disc_list.remove(disc_id)

    def __eq__(self, other):

        same_name = self.name == other.name
        same_value = self.value == other.value
        return same_name and same_value
