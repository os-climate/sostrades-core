'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
from sostrades_core.execution_engine.ns_manager import NamespaceManager
from sostrades_core.execution_engine.sos_builder import SoSBuilder


class ToolBuilder(SoSBuilder):
    '''
    Class that stores a class and associated attributes to be built afterwards
    '''
    NS_NAME_SEPARATOR = NamespaceManager.NS_NAME_SEPARATOR

    def __init__(self, tool_name, ee, cls):
        '''
        Constructor
        :param cls: class that will be instantiated by the builder
        :type cls: class
        '''
        SoSBuilder.__init__(self, tool_name, ee, cls)
        self.tool = None

    def instantiate(self):
        ''' Instantiates the class self.cls
        '''
        current_ns = self.ee.ns_manager.current_disc_ns

        # If we are in the builder of the high level coupling the current ns is None and
        # we have to check if the coupling has already been created
        # The future disc_name will be created without ns then
        if current_ns is None:
            future_new_ns_disc_name = self.sos_name
        else:
            future_new_ns_disc_name = f'{current_ns}.{self.sos_name}'

        if self.disc is None or future_new_ns_disc_name not in self.discipline_dict:
            self.create_tool(future_new_ns_disc_name)
        else:
            self.tool = self.discipline_dict[future_new_ns_disc_name]

        return self.tool

    def create_tool(self, future_new_ns_disc_name):

        self.tool = self.cls(**self.args)

        self.tool.father_builder = self
        self.discipline_dict[future_new_ns_disc_name] = self.tool
