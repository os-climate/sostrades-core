# -*- coding: utf-8 -*-
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

import os
import json
from importlib import import_module

DASHBOARD_MODULE = "dashboard"
DASHBOARD = "dashboard_template.json"


def generate_dashboard(execution_engine, post_processings):
    """
    Try to generate a dashboard from template located in process repository

    :params: execution_engine, execution engine that hold post processing data
    :type: ExecutionEngine

    :params: post_processings, post processing data of the study
    :type: list of post processings

    :return: serialized dashboard template
    """

    try:

        # Open dashboard_template file
        repo = execution_engine.factory.repository
        process = execution_engine.factory.process_identifier
        dashboard_module_path = f"{repo}.{process}"

        dashboard_module = import_module(dashboard_module_path)
        dashboard_file = os.path.join(
            os.path.dirname(dashboard_module.__file__),
            DASHBOARD_MODULE,
            DASHBOARD)

        if not os.path.isfile(dashboard_file):
            return {}

        with open(dashboard_file) as f:
            dashboard_template = json.load(f)

        # Fill dashboard_template
        rows = dashboard_template.get("rows", [])
        if rows:

            all_post_processings = post_processings

            for row in rows:
                for block in row:
                    if block.get("content_type") == "POST_PROCESSING":
                        try:
                            discipline = block.get('content_namespace').format(
                                study_name=execution_engine.study_name)

                            post_processing_bundle = all_post_processings.get(
                                discipline)
                            post_processings = post_processing_bundle[0].post_processings
                            content = post_processings[block.get(
                                "graph_index", 0)]

                            content['layout']['width'] = 1200 // len(row)
                            content['layout']['height'] = 450
                            # content['layout']['autosize'] = True
                            content['config'] = {'responsive': True}

                            block["content"] = content

                        except:
                            block["content_type"] = "TEXT"
                            block["content"] = f'graph {block.get("graph_index", 0)} for {discipline} not found'

                    elif block.get("content_type") == "SCALAR":
                        discipline = block.get('content_namespace').format(
                            study_name=execution_engine.study_name)
                        try:
                            uuid_param = execution_engine.dm.data_id_map[discipline]
                            param = execution_engine.dm.data_dict[uuid_param]
                            block["content"] = {
                                "var_name": param.get("var_name", ""),
                                "value": param.get("value", "")
                            }
                        except:
                            block["content_type"] = "TEXT"
                            block["content"] = f'{discipline} not found'

        return {
            "title": dashboard_template.get("title", ""),
            "rows": rows
        }

    except Exception as e:
        # Exception should not interrupt further process
        execution_engine.logger.exception(
            f'Exception during dashboard generation : {str(e)}')
        return {}
