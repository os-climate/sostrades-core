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

Example of post processing library that can be loaded throught the 'add_post_processing_module_to_namespace'
method of the post processing manager
"""
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries
from sostrades_core.execution_engine.data_manager import DataManager

def post_processing_filters(execution_engine, namespace):
    """ post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :returns: ChartFilter[]
    """

    filters = []
    (x, y) = get_x_and_y(execution_engine, namespace)

    if x is not None and y is not None:
        filters.append(ChartFilter('Selected module charts', filter_values=['Y versus X'], selected_values=[
            'Y versus X'], filter_key='y_vs_x', multiple_selection=True))

    return filters


def post_processings(execution_engine, namespace, filters):
    """ post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :params: filters, list of filters to applies to the post processing
    :type: ChartFilter[]

    :returns: list of post processing
    """
    chart_results = []
    generate_x_vs_y = True

    # If filter argument is not usable nothing will be filtered
    if filters is not None and isinstance(filters, list):
        rc_vs_price_filter = list(
            filter(lambda f: f.filter_key == 'y_vs_x', filters))

        if len(rc_vs_price_filter) > 0:
            generate_x_vs_y = 'Y versus X' in rc_vs_price_filter[
                0].selected_values

    if generate_x_vs_y:
        (x, y) = get_x_and_y(execution_engine, namespace)

        if x is not None and x is not None:
            x_min = x
            x_max = x

            y_min = y
            y_max = y

            new_chart = TwoAxesInstanciatedChart('y (-)', 'x (-)',
                                                 [y_min, y_max],
                                                 [x_min, x_max],
                                                 'Y versus X', True)

            x_y_serie = InstanciatedSeries([y], [x], 'Y vs X', 'lines')

            new_chart.add_series(x_y_serie)

            chart_results.append(new_chart)

    return chart_results


def get_x_and_y(execution_engine, namespace):
    """ post processing function designed to build check if data can be retrieved
    (x and y)

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :returns: tuple (rc dataframe, sale price dataframe)
    """

    x = None
    y = None

    try:
        x_namespace = f'{namespace}.x'
        y_namespace = f'{namespace}.y'

        x_key = execution_engine.dm.get_data_id(x_namespace)
        y_key = execution_engine.dm.get_data_id(y_namespace)

        x = execution_engine.dm.data_dict[x_key][DataManager.VALUE]
        y = execution_engine.dm.data_dict[y_key][DataManager.VALUE]
    except:
        x = None
        y = None

    return (x, y)
