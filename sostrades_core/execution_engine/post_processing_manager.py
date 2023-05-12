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
import logging
from typing import Union
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.pie_charts.instanciated_pie_chart import InstanciatedPieChart
from sostrades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable


"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Post processing manager allowing to coeespond namespace with post processing to execute
"""

from importlib import import_module


class PostProcessingManager:
    """ Class the store couples namespace <=> list of post processing.
    Post processing are sually stored by disciplines, but it exist namespace that are not associated to a discipline
    So this class allow to store post processing for a namespace and not a discipline
    """

    FILTER_FUNCTION_NAME = 'post_processing_filters'
    POST_PROCESSING_FUNCTION_NAME = 'post_processings'

    def __init__(self, execution_engine: "ExecutionEngine"):
        """ Constructor

            :params: execution_engine, instance of execution engine that host the current PostProcessingManager instance
            :type: ExecutionEngine
        """

        # Associated execution exengine
        self.__execution_engine = execution_engine

        # List where all namespaces are gathered (of all disciplines)
        self.__namespace_postprocessing_dict = {}

        # Initialize logger
        self.__logger = self.__execution_engine.logger.getChild("PostProcessingManager")

    @property
    def namespace_post_processing(self):
        """ Return the dict of namespace <=> post processing association

        :return: Dictionary {namespace: PostProcessing[]}
        """

        return self.__namespace_postprocessing_dict

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
        filters_function = None
        post_processing_function = None

        try:
            filters_function = getattr(import_module(module_name),
                                       PostProcessingManager.FILTER_FUNCTION_NAME)
        except (ModuleNotFoundError, AttributeError, TypeError) as ex:
            self.__logger.exception(
                'The following error occurs when trying to load post processing filter function.')
            filters_function = None

        try:
            post_processing_function = getattr(import_module(module_name),
                                               PostProcessingManager.POST_PROCESSING_FUNCTION_NAME)
        except (ModuleNotFoundError, AttributeError, TypeError) as ex:
            self.__logger.exception(
                'The following error occurs when trying to load post processing function.')
            post_processing_function = None

        if post_processing_function is not None:
            self.add_post_processing_functions_to_namespace(
                namespace_identifier, filters_function, post_processing_function)
        else:
            raise ValueError(
                f'Unable to load post processing function in the module : {module_name}.{PostProcessingManager.POST_PROCESSING_FUNCTION_NAME}')

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

    def get_post_processing(self, namespace_identifier: str): #-> list["PostProcessing"]:
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

    def resolve_filters(self, execution_engine: "ExecutionEngine", namespace: str): #-> list[ChartFilter]:
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
            filters = []

        return filters

    def resolve_post_processings(self, execution_engine: "ExecutionEngine", namespace: str, filters): #list[ChartFilter])
        #-> list[Union[TwoAxesInstanciatedChart, InstanciatedPieChart, InstanciatedTable]]:
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
            self.__logger.exception(
                f'An error occurs in the following post-processing namespace "{namespace}"')
            post_processings = []

        return post_processings
