'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/07-2024/06/24 Copyright 2023 Capgemini

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
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

    from sostrades_core.execution_engine.execution_engine import ExecutionEngine

"""
Post processing manager allowing to coeespond namespace with post processing to execute
"""


class PostProcessingManager:
    """ Class the store couples namespace <=> list of post processing.
    Post processing are sually stored by disciplines, but it exist namespace that are not associated to a discipline
    So this class allow to store post processing for a namespace and not a discipline
    """

    FILTER_FUNCTION_NAME = 'post_processing_filters'
    POST_PROCESSING_FUNCTION_NAME = 'post_processings'

    def __init__(self, execution_engine: ExecutionEngine):
        """ Constructor

            :params: execution_engine, instance of execution engine that host the current PostProcessingManager instance
            :type: ExecutionEngine
        """

        # Associated execution exengine
        self.__execution_engine = execution_engine

        # List where all namespaces are gathered (of all disciplines)
        self.__namespace_postprocessing_dict = {}

        # Initialize logger
        self.__logger = execution_engine.logger.getChild(self.__class__.__name__)

    @property
    def namespace_post_processing(self):
        """ Return the dict of namespace <=> post processing association

        :return: Dictionary {namespace: PostProcessing[]}
        """

        return self.__namespace_postprocessing_dict

    def get_post_processing_functions_from_module(self, module_name: str):
        """
        Function to get the post processing functions from a module
        Since filter function is optional, returns None if not found
        For post_processing function, raises ValueError if not found

        :params: module_name, python module that contains functions to process post processing
                (see PostProcessingManager.FILTER_FUNCTION_NAME and PostProcessingManager.POST_PROCESSING_FUNCTION_NAME)
        :type: string (python module like)
        """
        filter_function = None
        try:
            filter_function = getattr(import_module(module_name), PostProcessingManager.FILTER_FUNCTION_NAME)
        except (ModuleNotFoundError, AttributeError, TypeError) as ex:
            self.__logger.exception(f'The following error occurs when trying to load post processing filter function for module f{module_name}.', exc_info=ex)
        try:
            return filter_function, getattr(import_module(module_name), PostProcessingManager.POST_PROCESSING_FUNCTION_NAME)
        except (ModuleNotFoundError, AttributeError, TypeError) as ex:
            self.__logger.exception(f'The following error occurs when trying to load post processing function for module f{module_name}.')
            raise ValueError(f'Unable to load post processing function in the module : {module_name}.{PostProcessingManager.POST_PROCESSING_FUNCTION_NAME}') from ex

    def add_post_processing_module_to_namespace(self, namespace_identifier: str, module_name: str):
        """ Method that add a couple of filter+post processing function to a dedicated namespace into
        the PostProcessingManager

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string

        :params: module_name, python module that contains functions to process post processing
                (see PostProcessingManager.FILTER_FUNCTION_NAME and PostProcessingManager.POST_PROCESSING_FUNCTION_NAME)
        :type: string (python module like)
        """

        # Try to resolve post processings functions keeping in mind that filter
        # function can be optional
        filters_function, post_processing_function = self.get_post_processing_functions_from_module(module_name=module_name)

        self.add_post_processing_functions_to_namespace(namespace_identifier, filters_function, post_processing_function)

    def remove_post_processing_module_to_namespace(self, namespace_identifier: str, module_name: str, missing_ok: bool = True):
        """ Method that removes a couple of filter+post processing function to a dedicated namespace into
        the PostProcessingManager

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string

        :params: module_name, python module that contains functions to process post processing
                (see PostProcessingManager.FILTER_FUNCTION_NAME and PostProcessingManager.POST_PROCESSING_FUNCTION_NAME)
        :type: string (python module like)

        :params: missing_ok, if false, raises an exception if post_processing is not found.
        :type: bool
        """
        # Try to resolve post processings functions keeping in mind that filter
        # function can be optional
        filters_function, post_processing_function = self.get_post_processing_functions_from_module(module_name=module_name)

        self.remove_post_processing_functions_to_namespace(namespace_identifier, filters_function, post_processing_function, missing_ok=missing_ok)

    def remove_post_processing_functions_to_namespace(self,
                                                   namespace_identifier: str,
                                                   filter_func: callable,
                                                   post_processing_func: callable,
                                                   missing_ok: bool = True):
        """ Method that removes a couple of filter+post processing function to a dedicated namespace into
        the PostProcessingManager

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string

        :params: filter_func, methods that generate filter for the associated post processing
        :type: (func)(ExecutionEngine, namespace): ChartFilter[]

        :params: post_processing_func, methods that generate post processing for the associated post processing
        :type: (func)(ExecutionEngine, namespace, ChartFilter[]): list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list

        :params: missing_ok, if false, raises an exception if post_processing is not found.
        :type: bool
        """

        # Initialize the namespace placeholder in dictionary if needed
        if namespace_identifier not in self.__namespace_postprocessing_dict:
            if missing_ok:
                return
            else:
                raise ValueError(f"Namespace {namespace_identifier} not found in namespace_postprocessing_dict.")

        # Update post processing list
        matchs = []
        for i, post_proc in enumerate(self.__namespace_postprocessing_dict[namespace_identifier]):
            if post_proc.matchs_functions(filter_func, post_processing_func):
                matchs.append(i)

        if len(matchs) == 0:
            if missing_ok:
                return
            else:
                raise ValueError(f"Post processing not found in namespace f{namespace_identifier}")
        elif len(matchs) > 1:
            raise ValueError(f"Multiple matching post processing to delete found in namespace f{namespace_identifier}")

        # Delete found post_proc
        del self.__namespace_postprocessing_dict[namespace_identifier][matchs[0]]

    def add_post_processing_functions_to_namespace(self,
                                                   namespace_identifier: str,
                                                   filter_func: callable,
                                                   post_processing_func: callable):
        """ Method that add a couple of filter+post processing function to a dedicated namespace into
        the PostProcessingManager

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string

        :params: filter_func, methods that generate filter for the associated post processing
        :type: (func)(ExecutionEngine, namespace): ChartFilter[]

        :params: post_processing_func, methods that generate post processing for the associated post processing
        :type: (func)(ExecutionEngine, namespace, ChartFilter[]): list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list
        """

        # Initialize the namespace placeholder in dictionary if needed
        if namespace_identifier not in self.__namespace_postprocessing_dict:
            self.__namespace_postprocessing_dict[namespace_identifier] = []

        # Set post processing that hold functions
        post_processing_object = PostProcessing(
            filter_func, post_processing_func, self.__logger)

        # Update inner dictionary
        self.__namespace_postprocessing_dict[namespace_identifier].append(
            post_processing_object)

    def remove_namespace(self, namespace_identifier: str):
        """ Method remove a namespace entry from the inner dictionary

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string
        """

        # Robustness and remove dictionary key
        if namespace_identifier in self.__namespace_postprocessing_dict:
            del self.__namespace_postprocessing_dict[namespace_identifier]

    def get_post_processing(self, namespace_identifier: str):  # -> list["PostProcessing"]:
        """ Method that retrieve post processing object using a given namespace

        :params: namespace_identifier, namespace that hold post processing given as arguments
        :type: string

        :returns: PostProcessing[]
        """

        results = []

        # Robustness and remove dictionary key
        if namespace_identifier in self.__namespace_postprocessing_dict:
            results = self.__namespace_postprocessing_dict[namespace_identifier]

        return results


class PostProcessing:
    """ This class is intended to store two functions pointer related to a post processing:
    - a filter generator function
    - a post processing generator function
    """

    def __init__(self, filter_func: callable, post_processing_func: callable, logger: logging.Logger):
        """ Constructor

            :params: filter_func, methods that generate filter for the associated post processing
            :type: (func)(ExecutionEngine, namespace): ChartFilter[]

            :params: post_processing_func, methods that generate post processing for the associated post processing
            :type: (func)(ExecutionEngine, namespace, ChartFilter[]): list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list

        """

        self.__filter_func = filter_func
        self.__post_processing_func = post_processing_func
        self.__logger = logger
        self.post_processing_error = ''

    def resolve_filters(self, execution_engine: ExecutionEngine, namespace: str):  # -> list[ChartFilter]:
        """ Method that execute filters stored function and return the results

        :params: execution_engine, instance of execution engine that allow to resolve post processing
        :type: ExecutionEngine

        :params: namespace, namespace that request the post processing
        :type: string

        :returns: ChartFilter[]
        """
        from sostrades_core.execution_engine.execution_engine import ExecutionEngine
        if not isinstance(execution_engine, ExecutionEngine):
            raise ValueError(
                f'"execution_engine" argument must be of type "ExecutionEngine" and not "{type(execution_engine)}"')

        filters = []

        try:
            if self.__filter_func is not None:
                filters = self.__filter_func(execution_engine, namespace)
        except:
            self.post_processing_error += f'An error occurs in the filter of the following post-processing namespace "{namespace}"'
            filters = []

        return filters

    def resolve_post_processings(self, execution_engine: ExecutionEngine, namespace: str,
                                 filters):  # list[ChartFilter])
        # -> list[Union[TwoAxesInstanciatedChart, InstanciatedPieChart, InstanciatedTable]]:
        """ Method that execute stored function and return the results

        :params: execution_engine, instance of execution engine that allow to resolve post processing
        :type: ExecutionEngine

        :params: namespace, namespace that request the post processing
        :type: string

        :params: filters, filters to apply to the post processing function
        :type: ChartFilter[]

        :returns: list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list
        """
        from sostrades_core.execution_engine.execution_engine import ExecutionEngine
        if not isinstance(execution_engine, ExecutionEngine):
            raise ValueError(
                f'"execution_engine" argument must be of type "ExecutionEngine" and not "{type(execution_engine)}"')

        post_processings = []

        try:
            if self.__post_processing_func is not None:
                post_processings = self.__post_processing_func(
                    execution_engine, namespace, filters)

        except:
            self.post_processing_error += f'\nERROR: {execution_engine.factory.process_identifier}.{execution_engine.study_name}, An error occurs in the following post-processing namespace "{namespace}"'
            self.__logger.exception(self.post_processing_error)
            post_processings = []

        return post_processings

    def matchs_functions(self, filter_func: callable, post_processing_func: callable):
        """
        Returns True if functions match
        """
        return filter_func == self.__filter_func and post_processing_func == self.__post_processing_func
