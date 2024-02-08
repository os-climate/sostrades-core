'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/10-2023/11/03 Copyright 2023 Capgemini

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

import abc


class SosTool():
    '''
    Class that instantiate a tool to help a driver to build sub disciplines
    '''

    # ontology information
    _ontology_data = {
        'label': 'SoS Tool',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-indent fa-fw',
        'version': '',
    }

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''

        self.sos_name = sos_name

        self.associated_namespaces = None
        self.ee = ee
        self.driver = None
        self.sub_builders = cls_builder

    @property
    def sub_builders(self):

        return self.__builders

    @sub_builders.setter
    def sub_builders(self, sub_builders):
        """
        setter of sub_builders
        """
        if isinstance(sub_builders, list):
            self.__builders = sub_builders
        else:
            self.__builders = [sub_builders]

    def associate_tool_to_driver(self, driver, cls_builder=None, associated_namespaces=None):
        '''

        '''
        self.driver = driver

        if associated_namespaces is None:
            self.associated_namespaces = []
        else:
            self.associated_namespaces = associated_namespaces

        if cls_builder is not None:
            self.sub_builders = cls_builder

    @abc.abstractmethod
    def build(self):
        ''' 
        Configuration of the SoSscatter : 
        -First configure the scatter 
        -Get the list to scatter on and the associated namespace
        - Look if disciplines are already scatterred and compute the new list to scatter (only new ones)
        - Remove disciplines that are not in the scatter list
        - Scatter the instantiator cls and adapt namespaces depending if it is a list or a singleton
        '''

    @abc.abstractmethod
    def prepare_tool(self):
        '''
        Prepare tool function if some data of the driver are needed to configure the tool
        '''

    def get_dynamic_output_from_tool(self):
        pass

    def associate_namespaces_to_builder(self, builder, ns_list):
        '''
        Associate namespaces defined in the constructor + all namespaces which has been updated in update_namespaces
        '''
        if self.associated_namespaces != []:
            builder.add_namespace_list_in_associated_namespaces(
                self.associated_namespaces)
        builder.add_namespace_list_in_associated_namespaces(
            ns_list)

    def set_father_discipline(self):
        '''
        Set the current discipline to build the builder_list at father_executor of the driver level (which is the coupling above the driver
        '''

        self.ee.factory.current_discipline = self.driver.father_executor
