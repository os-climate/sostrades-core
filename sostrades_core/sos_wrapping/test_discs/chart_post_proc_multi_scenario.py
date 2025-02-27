'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/12-2024/05/17 Copyright 2023 Capgemini

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

# pylint: disable=line-too-long

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.tables.instanciated_table import (
    InstanciatedTable,
)


def post_processing_filters(execution_engine, namespace):
    """Post processing function designed to build filters"""
    filters = []

    chart_list = ['Scenario table']
    filters.append(ChartFilter(
        'Charts', chart_list, chart_list, filter_key='graphs'))
    return filters


def post_processings(execution_engine, namespace, filters):
    """Post processing function designed to build graphs"""
    instanciated_charts = []
    charts_list = []

    # Overload default value with chart filter
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == 'graphs':
                charts_list = chart_filter.selected_values

    if 'Scenario table' in charts_list:
        table_name = 'Scenario dict'
        scenario_list = execution_engine.dm.get_value(
            'MyCase.multi_scenarios.samples_df')['scenario_name'].values.tolist()

        headers = ['Scenarios']

        cells = []
        cells.append(scenario_list)

        new_table = InstanciatedTable(
            table_name, headers, cells)
        instanciated_charts.append(new_table)

    return instanciated_charts
