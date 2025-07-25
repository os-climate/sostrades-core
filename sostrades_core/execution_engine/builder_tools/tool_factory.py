'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2025/02/26 Copyright 2025 Capgemini

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
from sostrades_core.execution_engine.builder_tools.tool_builder import ToolBuilder
from sostrades_core.tools.import_tool.import_tool import get_class_from_path, get_module_class_path


class ToolFactoryException(Exception):
    pass


class ToolFactory:
    """Specification: ToolFactory allows to manage tools used by driver to prepare build"""

    EE_PATH = 'sostrades_core.execution_engine'
    TOOL_FOLDER = f'{EE_PATH}.builder_tools'

    def __init__(self, execution_engine, sos_name):
        """
        Constructor

        :params: execution_engine (current execution engine instance)
        :type: ExecutionEngine

        :params: sos_name (discipline name)
        :type: string
        """
        self.__sos_name = sos_name
        self.__execution_engine = execution_engine

        self.__builder_tools = []

        self.__logger = self.__execution_engine.logger.getChild(self.__class__.__name__)

        self.__reset()

    def __reset(self):
        """Reinitialize members variables"""
        self.__builder_tools = []

    @property
    def sos_name(self):
        return self.__sos_name

    def add_tool(self, tool):
        """Add a tool to the list of factory tools"""
        self.__builder_tools.append(tool)

    def add_tool_list(self, tools):
        self.__builder_tools.extend(tools)

    def remove_tool(self, tool):
        """
        Remove one discipline from coupling
        :param disc: sos discipline to remove
        :type: SoSDiscipline Object
        """
        self.__builder_tools.remove(tool)

    @property
    def builder_tools(self):
        """
        Return all sostrades disciplines manage by the factory

        :returns: list of sostrades disciplines
        :type: SoSDisciplines[]
        """
        return self.__builder_tools

    @property
    def repository(self):
        """Return the repository used to create the process"""
        return self.__repository

    @repository.setter
    def repository(self, value):
        """Set the repository used to create the process"""
        self.__repository = value

    def get_builder_from_class_name(self, sos_name, mod_name, folder_list):
        """Get builder only using class name and retrievind the module path from the function get_module_class_path"""
        mod_path = get_module_class_path(mod_name, folder_list)

        if mod_path is None:
            raise ToolFactoryException(
                f'The builder {mod_name} has not been found in the folder list {folder_list}'
            )
        return self.get_builder_from_module(sos_name, mod_path)

    def get_builder_from_module(self, sos_name, mod_path):
        """Get a builder which is defined by the class in the mod_path"""
        cls = get_class_from_path(mod_path)
        builder = ToolBuilder(sos_name, self.__execution_engine, cls)
        return builder

    def create_tool_builder(
            self, tool_name, tool_type, sub_builders=None,
            map_name=None
    ):
        """Create a  tool builder"""
        tool_builder = self.get_builder_from_class_name(
            tool_name, tool_type, [self.TOOL_FOLDER])
        tool_builder.set_builder_info('map_name', map_name)
        tool_builder.set_builder_info('cls_builder', sub_builders)
        return tool_builder
