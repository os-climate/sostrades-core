'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/12-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable


def post_processing_filters(execution_engine, namespace):
    chart_filters = []

    chart_list = ['o vs x']

    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'graphs'))

    return chart_filters


def post_processings(execution_engine, namespace, chart_filters=None):
    instanciated_charts = []

    charts_list = []
    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'graphs':
                charts_list = chart_filter.selected_values

    if 'o vs x' in charts_list:
        chart_name = 'o vs x'

        o = list(execution_engine.dm.get_value(f'{namespace}.o'))
        x = list(execution_engine.dm.get_value(f'{namespace}.x'))

        new_chart = TwoAxesInstanciatedChart('x (-)', 'o (-)',
                                             chart_name=chart_name)
        serie = InstanciatedSeries(
            x, o, '', 'lines')

        new_chart.series.append(serie)

        instanciated_charts.append(new_chart)

    return instanciated_charts
