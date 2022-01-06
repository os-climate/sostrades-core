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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
from sos_trades_core.tools.post_processing.tables.table_style import TableStyles
from sos_trades_core.tools.post_processing.post_processing_tools import align_two_y_axes
import plotly.graph_objects as go

import numpy as np
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import InstantiatedPlotlyNativeChart

# Post-processing module to test filters and graphs implementation for gather discipline
# Implemented graphs are implemented to visualize outputs of gather of disc1_scenario.py
# The graphs are tests on test_01_multi_scenario_of_scatter of
# test_l0_35_very_simple_multi_scenario.py


def get_chart_filter_list(discipline):
    """
    Create filters on graphs of gather of Disc1
    """
    chart_filters = []

    chart_list = ['y gather', 'y two axes graph', 'y table']

    chart_filters.append(ChartFilter(
        'Charts', chart_list, chart_list, 'graphs'))

    return chart_filters


def get_instanciated_charts(discipline, chart_filters=None):
    """
    Create graphs to visualize outputs of gather of Disc1
    """
    instanciated_charts = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'graphs':
                charts_list = chart_filter.selected_values

    if 'y gather' in charts_list:

        chart_name = 'y gather'
        if 'y_dict' in discipline.get_data_out():
            y_dict = discipline.get_sosdisc_outputs('y_dict')

            new_chart = TwoAxesInstanciatedChart('scenarios', 'y (-)',
                                                 chart_name=chart_name)
            for scenario, y in y_dict.items():
                serie = InstanciatedSeries(
                    [scenario], [y], '', 'bar')

            new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

    if 'y two axes graph' in charts_list:

        chart_name = 'y two axes graph'

        if 'y_dict' in discipline.get_data_out():
            y_dict = discipline.get_sosdisc_outputs('y_dict')

            # Create figure
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=list(y_dict.keys()),
                    y=list(y_dict.values()),
                    yaxis='y',
                    visible=True,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=list(y_dict.keys()),
                    y=[y**2 for y in y_dict.values()],
                    yaxis='y2',
                    visible=True,
                )
            )

            fig.update_layout(
                autosize=True,
                xaxis=dict(
                    title='Scenarios',
                    titlefont_size=12,
                    tickfont_size=10,
                    automargin=True
                ),
                yaxis=dict(
                    title='y',
                    titlefont_size=12,
                    tickfont_size=10,
                    automargin=True,

                ),
                yaxis2=dict(
                    title='y**2',
                    titlefont_size=12,
                    tickfont_size=10,
                    automargin=True,
                    anchor="x",
                    overlaying="y",
                    side="right",
                ),
                barmode='group',
                # gap between bars of adjacent location coordinates.
                bargap=0.15,
                # gap between bars of the same location coordinate.
                bargroupgap=0.1
            )

            new_chart = None
            fig = align_two_y_axes(fig, GRIDLINES=4)
            new_chart = InstantiatedPlotlyNativeChart(
                fig=fig, chart_name=chart_name)

            instanciated_charts.append(new_chart)

    return instanciated_charts


def get_instanciated_tables(discipline, chart_filters=None):
    """
    Create a table to visualize outputs of gather of Disc1
    """
    instanciated_tables = []

    # Overload default value with chart filter
    if chart_filters is not None:
        for chart_filter in chart_filters:
            if chart_filter.filter_key == 'graphs':
                charts_list = chart_filter.selected_values

    if 'y table' in charts_list:

        table_name = 'y value by scenario by name'

        if 'y_dict' in discipline.get_data_out():

            y_dict = discipline.get_sosdisc_outputs('y_dict')
            headers = ['Scenario', 'Name', 'y value']

            cells = []
            scenario_list = []
            name_list = []
            values_list = []

            for key, value in y_dict.items():
                if '.' in key:
                    scenario_list.append(key.split('.')[0])
                    name_list.append(key.split('.')[1])
                    values_list.append(value)

            cells.append(scenario_list)
            cells.append(name_list)
            cells.append(values_list)

            if np.size(cells) > 0:
                # Prepare overall style for display
                styles = {}
                # update color of scenario_1 rows
                styles.update(
                    {0: TableStyles(background_color='royalblue', font_color='white'),
                     1: TableStyles(background_color='royalblue', font_color='white')})

                # update color of scenario_2 rows
                styles.update(
                    {2: TableStyles(background_color='lightblue', font_color='white'),
                     3: TableStyles(background_color='lightblue', font_color='white')})

                new_table = InstanciatedTable(
                    table_name, headers, cells, styles)
                instanciated_tables.append(new_table)

    return instanciated_tables
