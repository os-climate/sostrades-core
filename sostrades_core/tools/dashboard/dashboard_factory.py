'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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
# -*- coding: utf-8 -*-

import json
import os
from importlib import import_module
from os.path import exists

from sostrades_core.tools.dashboard.dashboard import Dashboard, GraphData
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

DASHBOARD_FOLDER_NAME = "dashboard"
DASHBOARD_FILE_NAME = "dashboard.json"

def get_default_dashboard_in_process_repo(execution_engine)-> Dashboard:
    """
    Try to get a default dashboard from template located in process repository

    :params: execution_engine, execution engine that hold post processing data
    :type: ExecutionEngine
    :return: serialized dashboard template
    """
    # Open dashboard_template file

    repo = execution_engine.factory.repository
    process = execution_engine.factory.process_identifier
    dashboard_module_path = f"{repo}.{process}"

    dashboard_module = import_module(dashboard_module_path)
    # dashboard default file is in process_repository/dashboard/dashboard.json
    dashboard_file = os.path.join(
        os.path.dirname(dashboard_module.__file__),
        DASHBOARD_FOLDER_NAME,
        DASHBOARD_FILE_NAME)

    return get_dashboard_from_file(dashboard_file)


def get_dashboard_from_file(file_path:str)->Dashboard:
    """
    Retrieve dashboard from json file if exists
    Args:
        file_path (str): path to the json file
    Returns:
        Dashboard: dashboard object if file exists, None otherwise
    """
    dashboard = None
    if exists(file_path):
        with open(file_path) as f:
            dashboard_json = json.load(f)
            dashboard = Dashboard.deserialize(dashboard_json)
    return dashboard


def update_namespace_with_new_study_name(namespace_to_update, study_name):
    """
    Update the namespace to replace the old study name by the new one

    :params: namespace_to_update, namespace to update
    :type: str
    :params: study_name, new study name
    :type: str
    :return: updated namespace
    """
    if namespace_to_update is None or len(namespace_to_update) == 0:
        return namespace_to_update
    # split the namespace by '.'
    namespace_parts = namespace_to_update.split('.')
    if len(namespace_parts) > 0:
        # replace the first part by the new study name
        namespace_parts[0] = study_name
        # join the parts back together
        return '.'.join(namespace_parts)
    return namespace_to_update


def update_dashboard_charts(execution_engine, dashboard:Dashboard)->Dashboard:
    """
    Try to generate a dashboard from template located in process repository

    :params: execution_engine, execution engine that hold post processing data
    :type: ExecutionEngine

    :params: post_processings, post processing data of the study
    :type: list of post processings

    :return: serialized dashboard template
    """
    # Update chart data
    study_name  = execution_engine.study_name
    all_charts_by_filters = {}
    filters_list_by_discipline = {}
    new_datas = {}

    post_processing_factory = PostProcessingFactory()
    for key, item in dashboard.data.items():
        if isinstance(item, GraphData):
            item.disciplineName = update_namespace_with_new_study_name(item.disciplineName, study_name)
            module_name = item.name
            filters = item.postProcessingFilters
            if item.disciplineName not in filters_list_by_discipline or filters not in filters_list_by_discipline[item.disciplineName]:
                # build the filters object
                filters_objects = []
                for filter in filters:
                    filter_obj = ChartFilter(name=filter['filterName'],
                                             filter_values=filter['filterValues'],
                                             selected_values=filter['selectedValues'],
                                             filter_key=filter['filterKey'],
                                             multiple_selection=filter['multipleSelection'])

                    filters_objects.append(filter_obj)
                # rebuild the charts list for this filters
                post_processings = post_processing_factory.get_post_processings_by_discipline_name(item.disciplineName, module_name, execution_engine, filters_objects)

                # save the list of filters
                filters_list_by_discipline[item.disciplineName] = filters_list_by_discipline.get(item.disciplineName, [])
                filters_list_by_discipline[item.disciplineName].append(filters)
                filter_index = filters_list_by_discipline[item.disciplineName].index(filters)
                # save the charts list for this filter
                all_charts_by_filters[item.disciplineName] = all_charts_by_filters.get(item.disciplineName, {})
                all_charts_by_filters[item.disciplineName][filter_index] = post_processings
            #if the charts exists, set the data
            filter_index = filters_list_by_discipline[item.disciplineName].index(filters)
            if len(all_charts_by_filters.get(item.disciplineName,{}).get(filter_index,[])) > item.plotIndex:
                item.graphData = all_charts_by_filters[item.disciplineName][filter_index][item.plotIndex]

            #update the new_data dict
            new_datas[item.id()] = item
            # update layout if id has changed
            if item.id() != key:
                dashboard.layout[item.id()] = dashboard.layout[key]
                dashboard.layout[item.id()].item_id = item.id()
                del dashboard.layout[key]
        else:
            new_datas[key] = item

    dashboard.data = new_datas
    return dashboard
