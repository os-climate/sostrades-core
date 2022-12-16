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

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Factory for post processing (2 axes chart, pie chart, table)
"""
import inspect
import importlib
from os.path import join, dirname, isfile

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.proxy_discipline_gather import ProxyDisciplineGather
from sostrades_core.execution_engine.data_manager import DataManager
from sostrades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart import \
    InstantiatedParetoFrontOptimalChart
from sostrades_core.tools.post_processing.post_processing_bundle import PostProcessingBundle


class PostProcessingFactory:
    """ Class that centralized extraction of post processsing information from discipline
    """

    MODULE_CONFIGURATION_FILE_GATHER = 'chart_post_processing_gather'
    MODULE_CHART_FILTER_METHOD = 'get_chart_filter_list'
    MODULE_CHART_GENERATION_METHOD = 'get_instanciated_charts'
    MODULE_TABLE_GENERATION_METHOD = 'get_instanciated_tables'
    NAMESPACED_POST_PROCESSING = 'namespaced_post_processing'
    NAMESPACED_POST_PROCESSING_NAME = 'Data'

    def get_all_post_processings(self, execution_engine, filters_only, as_json=True, for_test=False):
        """ Extract all post processing filters that are defined into the execution engine
            (using discipline and post processing manager)

            :params: execution_engine, execution engine that hold post processing data
            :type: ExecutionEngine

            :params: filters_only,
                     If True, only retrieve filters from the execution.
                     If False then get associated post processing
            :type: boolean

            :params: as_json: return a jsonified object
            :type: boolean

            :params: for_test: specify if it is for test purpose
            :type: boolean

            :returns: Dictionary {namespace: PostProcessingBundle[]}

        """

        logger = get_sos_logger(
            f'{execution_engine.logger.name}.PostProcessing')

        all_post_processings_bundle = {}

        # Manage disciplines
        for value in execution_engine.dm.disciplines_dict.values():

            discipline = value[DataManager.DISC_REF]
            discipline_full_name = discipline.get_disc_full_name()

            if discipline_full_name not in all_post_processings_bundle:
                all_post_processings_bundle[discipline_full_name] = []

            found_bundles = list(
                filter(lambda b: b.name == discipline.get_module(), all_post_processings_bundle[discipline_full_name]))

            current_bundle = None
            if len(found_bundles) == 0:
                current_bundle = PostProcessingBundle(
                    discipline.get_module(),discipline.get_disc_label(), [], [])
                all_post_processings_bundle[discipline_full_name].append(
                    current_bundle)
            else:
                current_bundle = found_bundles[0]

            # Extract filters
            filters = self.get_post_processing_filters_by_discipline(
                discipline)
            if filters and len(filters) > 0:
                current_bundle.filters.extend(filters)

            # If filters only is False then generate associated post processing
            if not filters_only:
                post_processings = self.get_post_processing_by_discipline(
                    discipline, filters, as_json, for_test=for_test)

                if post_processings and len(post_processings) > 0:
                    current_bundle.post_processings.extend(post_processings)

        # Manage filters from post processing manager (namespace filter)
        for namespace_name, post_processings in execution_engine.post_processing_manager.namespace_post_processing.items():
            # Key is the namespace name, wo we haev to find all of its
            # implement
            associated_namespaces = list(filter(
                lambda ns: ns.name == namespace_name, execution_engine.ns_manager.ns_list))

            # now we can generate filter for each of them
            for associated_namespace in associated_namespaces:

                if associated_namespace.value not in all_post_processings_bundle:
                    all_post_processings_bundle[associated_namespace.value] = [
                    ]

                found_bundles = list(
                    filter(lambda b: b.name == PostProcessingFactory.NAMESPACED_POST_PROCESSING,
                           all_post_processings_bundle[associated_namespace.value]))

                current_bundle = None
                if len(found_bundles) == 0:
                    current_bundle = PostProcessingBundle(
                        PostProcessingFactory.NAMESPACED_POST_PROCESSING_NAME, PostProcessingFactory.NAMESPACED_POST_PROCESSING_NAME, [], [])
                    all_post_processings_bundle[associated_namespace.value].append(
                        current_bundle)
                else:
                    current_bundle = found_bundles[0]

                for post_processing in post_processings:
                    filters = post_processing.resolve_filters(
                        execution_engine, associated_namespace.value)

                    if filters and len(filters) > 0:
                        current_bundle.filters.extend(filters)

                    # If filters only is False then generate associated post
                    # processing
                    if not filters_only:
                        generated_post_processings = post_processing.resolve_post_processings(
                            execution_engine, associated_namespace.value, filters)

                        if as_json:
                            generated_post_processings = self.__convert_post_processing_into_json(
                                generated_post_processings, logger=logger)

                        if generated_post_processings and len(generated_post_processings) > 0:
                            current_bundle.post_processings.extend(
                                generated_post_processings)

        key_to_delete = []
        # Remove disctionary key without any post processing bundle
        for key, value in all_post_processings_bundle.items():
            if len(value) == 0:
                key_to_delete.append(key)
            else:
                cleaned_list = list(
                    filter(lambda b: b.has_post_processings, value))

                if len(cleaned_list) == 0:
                    key_to_delete.append(key)
                else:
                    all_post_processings_bundle[key] = cleaned_list

        for key in key_to_delete:
            del all_post_processings_bundle[key]

        return all_post_processings_bundle

    def get_post_processing_filters_by_namespace(self, execution_engine, namespace):
        """ Method that retrieve post processing filter base on a namespace value
            (using discipline and post processing manager)

            :params: execution_engine, execution engine that hold post processing data
            :type: ExecutionEngine

            :params: namespace, namespace to use to retrieve filters
            :type: string

            :returns: ChartFilters[]
        """
        discipline_list = []

        try:
            # First check if namespace match disciplines
            discipline_list = execution_engine.dm.get_disciplines_with_name(
                namespace)

        except:
            discipline_list = []

        all_filters = []

        for discipline in discipline_list:

            all_filters.extend(
                self.get_post_processing_filters_by_discipline(discipline))

        # Then look into the post processing manager (namespace based)
        # Extract namespace object having 'namespace' argument as value
        associated_namespaces = list(filter(
            lambda ns: ns.value == namespace, execution_engine.ns_manager.ns_list))

        # For each of them check if they are reference in the post processing
        # manager
        for associated_namespace in associated_namespaces:

            post_processings = execution_engine.post_processing_manager.get_post_processing(
                associated_namespace.name)

            for post_processing in post_processings:
                all_filters.extend(post_processing.resolve_filters(
                    execution_engine, associated_namespace.value))

        return all_filters

    def get_post_processing_by_namespace(self, execution_engine, namespace, filters, as_json=True, for_test=False):
        """ Method that retrieve post processing filter base on a namespace value
            (using discipline and post processing manager)

            :params: execution_engine, execution engine that hold post processing data
            :type: ExecutionEngine

            :params: namespace, namespace to use to retrieve filters
            :type: string

            :params: filters: filter to apply to post processing generation
            :type: ChartFilter[]

            :params: as_json: return a jsonified object
            :type: boolean

            :params: for_test: specify if it is for test purpose
            :type: boolean

            :returns: Post-processing list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list
        """
        discipline_list = []

        try:
            # First check if namespace match disciplines
            discipline_list = execution_engine.dm.get_disciplines_with_name(
                namespace)

        except:
            discipline_list = []

        all_post_processings = []

        for discipline in discipline_list:

            all_post_processings.extend(
                self.get_post_processing_by_discipline(discipline, filters, as_json, for_test=for_test))

        # Then look into the post processing manager (namespace based)
        # Extract namespace object having 'namespace' argument as value
        associated_namespaces = list(filter(
            lambda ns: ns.value == namespace, execution_engine.ns_manager.ns_list))

        # For each of them check if they are reference in the post processing
        # manager
        for associated_namespace in associated_namespaces:

            post_processings = execution_engine.post_processing_manager.get_post_processing(
                associated_namespace.name)

            for post_processing in post_processings:
                results = post_processing.resolve_post_processings(
                    execution_engine, associated_namespace.value, filters)

                if results and len(results) > 0:
                    if as_json:
                        all_post_processings.extend(
                            self.__convert_post_processing_into_json(results))
                    else:
                        all_post_processings.extend(results)

        return all_post_processings

    def get_post_processing_filters_by_discipline(self, discipline, for_test=False):
        """ Retrieve post processing filters for a given discipline

        :params: discipline : discipline instance to query to get associated
        post processing filters
        :type: SoSDiscipline
        """

        result = []

        # Initialize logger for the discipline
        logger = get_sos_logger(
            f'{discipline.logger.name}.PostProcessing')

        #######################################################################
        # Load definition from module file

        if isinstance(discipline, ProxyDisciplineGather):
            # Use the associated discipline for gather disicplines

            try:

                if isinstance(discipline.builder.disc, ProxyDisciplineGather):
                    if isinstance(discipline.builder.disc.builder.disc, ProxyDisciplineGather):
                        if isinstance(discipline.builder.disc.builder.disc.builder.disc, ProxyDisciplineGather):
                            linked_builder_class = discipline.builder.disc.builder.disc.builder.disc.builder.cls
                        else:
                            linked_builder_class = discipline.builder.disc.builder.disc.builder.cls
                    else:
                        linked_builder_class = discipline.builder.disc.builder.cls
                else:
                    linked_builder_class = discipline.builder.cls

                module_configuration_file = PostProcessingFactory.MODULE_CONFIGURATION_FILE_GATHER

                post_processing_file = join(dirname(inspect.getfile(
                    linked_builder_class)), f'{module_configuration_file}.py')

                if isfile(post_processing_file):
                    disc_parent_module = '.'.join(
                        linked_builder_class.__module__.split('.')[:-1])

                    chart_module = f'{disc_parent_module}.{module_configuration_file}'

                    process_mod = getattr(
                        importlib.import_module(chart_module), PostProcessingFactory.MODULE_CHART_FILTER_METHOD)

                    for chart_name in process_mod(discipline):
                        result.append(chart_name)

            # Catch all the exception in case the module does not exist
            except Exception:
                logger.exception(
                    f'The following error occurs when trying to load post processing filters for {discipline.get_disc_full_name()} discipline. See exception details.')
        else:
            try:
                for post_processing_filter in discipline.get_chart_filter_list():
                    result.append(post_processing_filter)
            except Exception as e:
                discipline.logger.exception(
                    f'The following error occurs when trying to load post processing filters for {discipline.get_disc_full_name()} discipline. See exception details.')
                if for_test is True:
                    raise Exception(e)
        return result

    def get_post_processing_by_discipline(self, discipline, filters, as_json=True, for_test=False):
        """ Retrieve post processing for a given discipline

        :params: discipline: discipline instance to query to get associated post processing
        :type: SoSDiscipline

        :params: filters: filter to apply to post processing generation
        :type: ChartFilter[]

        :params: as_json: return a jesonified object
        :type: boolean

        :params: for_test: specify if it is for test purpose
        :type: boolean

        :returns: Post-processing list (TwoAxesInstanciatedChart/InstanciatedPieChart/InstanciatedTable) or json oject list
        """

        # Initialize logger for the discipline
        logger = get_sos_logger(
            f'{discipline.logger.name}.PostProcessing')

        post_processing_results = []

        ###############################################################
        # Load definition from module file

        if isinstance(discipline, ProxyDisciplineGather):
            # TODO: recursive check for lines below?
            # Use the associated discipline for gather disicplines
            if isinstance(discipline.builder.disc, ProxyDisciplineGather):
                if isinstance(discipline.builder.disc.builder.disc, ProxyDisciplineGather):
                    if isinstance(discipline.builder.disc.builder.disc.builder.disc, ProxyDisciplineGather):
                        linked_builder_class = discipline.builder.disc.builder.disc.builder.disc.builder.cls
                    else:
                        linked_builder_class = discipline.builder.disc.builder.disc.builder.cls
                else:
                    linked_builder_class = discipline.builder.disc.builder.cls
            else:
                linked_builder_class = discipline.builder.cls

            module_configuration_file = PostProcessingFactory.MODULE_CONFIGURATION_FILE_GATHER

            post_processing_file = join(dirname(inspect.getfile(
                linked_builder_class)), f'{module_configuration_file}.py')

            if isfile(post_processing_file):
                disc_parent_module = '.'.join(
                    linked_builder_class.__module__.split('.')[:-1])

                chart_module = f'{disc_parent_module}.{module_configuration_file}'

                process_mod = None
                try:
                    process_mod = getattr(
                        importlib.import_module(chart_module), PostProcessingFactory.MODULE_CHART_GENERATION_METHOD)

                except AttributeError:
                    # ignore error if the method does not exist (so there is no
                    # post processing charts)
                    pass

                if process_mod:
                    try:
                        for chart in process_mod(discipline, filters):
                            post_processing_results.append(chart)
                    except Exception as e:
                        logger.exception(
                            f'The following error occurs when trying to load chart post processing for {discipline.get_disc_full_name()} discipline. See exception details')
                        if for_test is True:
                            raise Exception(e)

                process_mod = None
                try:
                    process_mod = getattr(
                        importlib.import_module(chart_module), PostProcessingFactory.MODULE_TABLE_GENERATION_METHOD)

                except AttributeError:
                    # ignore error if the method does not exist (so there is no
                    # post processing tables)
                    pass

                if process_mod:
                    try:
                        for table in process_mod(discipline, filters):
                            post_processing_results.append(table)
                    except Exception as e:
                        logger.exception(
                            f'The following error occurs when trying to load table post processing for  {discipline.get_disc_full_name()} discipline. See exception details')
                        if for_test is True:
                            raise Exception(e)
        # Standard disciplines management
        else:

            try:
                for post_processing in discipline.get_post_processing_list(filters):
                    post_processing_results.append(post_processing)
            except Exception as e:
                logger.exception(
                    f'The following error occurs when trying to load post processing for {discipline.get_disc_full_name()} discipline. See exception details.')
                if for_test is True:
                    raise Exception(e)

        if as_json:
            json_objects = self.__convert_post_processing_into_json(
                post_processing_results, logger=logger)

            return json_objects

        return post_processing_results

    def __convert_post_processing_into_json(self, post_processings, logger=None):
        """ Manage to get plotly object into post processing object and convert it into
        json with the removing of the template section

        @param post_processings: post processing object to convert
        @type sostrades_core.tools.post_processing.*

        @param logger: logger instance
        @type Logging.logger

        @return json object list
        """

        json_objects = []

        for post_processing in post_processings:
            json_object = post_processing.to_plotly_dict(logger)

            if isinstance(post_processing, InstantiatedParetoFrontOptimalChart):
                json_object['is_pareto_trade_chart'] = True
            else:
                json_object['is_pareto_trade_chart'] = False

            if 'template' in json_object['layout']:
                json_object['layout'].pop('template', None)

            json_objects.append(json_object)

        return json_objects
