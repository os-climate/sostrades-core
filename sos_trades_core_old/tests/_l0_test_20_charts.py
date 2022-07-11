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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for chart template
'''
import unittest
from os.path import dirname, join

import sostrades_core


class TestChartTemplate(unittest.TestCase):
    """
    Class to test template filters and charts
    """

    def setUp(self):
        self.data_path = join(dirname(sostrades_core.__file__),
                              'tests', 'data', 'charts')

    def test_01_check_chart_filter(self):

        from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter

        chart_list = ['1st graph', '2nd graph']
        chart_list_selected = ['2nd graph']
        chart_filter = ChartFilter(
            'Charts', chart_list, chart_list_selected, 'graphs')
        chart_filter_dict = chart_filter.to_dict()

        chart_filter_new = ChartFilter.from_dict(chart_filter_dict)

        chart_filter_dict_new = chart_filter_new.to_dict()
        self.assertDictEqual(chart_filter_dict, chart_filter_dict_new)

    def test_02_create_chart_template(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesChartTemplate

        dict_obj = {}
        dict_obj['chart_name'] = 'test'
        dict_obj['abscissa_axis_name'] = 'xx'
        dict_obj['primary_ordinate_axis_name'] = 'yy'
        dict_obj['secondary_ordinate_axis_name'] = 'yy2'
        dict_obj['abscissa_axis_range'] = [0, 10]
        dict_obj['primary_ordinate_axis_range'] = [50, 100]
        dict_obj['secondary_ordinate_axis_range'] = [1000, 10000]
        dict_obj['annotation_upper_left'] = {
            'key a': 'value a',
            'key b': 'value b'
        }
        dict_obj['annotation_upper_right'] = {
            'key c': 'value a'
        }
        dict_obj['bar_orientation'] = 'v'
        dict_obj['cumulative_surface'] = False
        dict_obj['stacked_bar'] = False
        dict_obj['series'] = []

        chart_template = TwoAxesChartTemplate()
        chart_template.from_dict(dict_obj)

        self.assertDictEqual(dict_obj, chart_template.to_dict(
        ), f'Data structure is not same anymore between reference and model')

    def test_03_create_chart_template_with_series(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesChartTemplate

        dict_obj = {}
        dict_obj['chart_name'] = 'test'
        dict_obj['abscissa_axis_name'] = 'xx'
        dict_obj['primary_ordinate_axis_name'] = 'yy'
        dict_obj['secondary_ordinate_axis_name'] = 'yy'
        dict_obj['abscissa_axis_range'] = [0, 10]
        dict_obj['primary_ordinate_axis_range'] = [50, 100]
        dict_obj['secondary_ordinate_axis_range'] = [50, 100]
        dict_obj['annotation_upper_left'] = {
            'key a': 'value a',
            'key b': 'value b'
        }
        dict_obj['annotation_upper_right'] = {
            'key c': 'value a'
        }
        dict_obj['bar_orientation'] = 'v'
        dict_obj['cumulative_surface'] = False
        dict_obj['stacked_bar'] = False
        dict_obj['series'] = []

        dict_series = {}
        dict_series['series_name'] = 'nom de la series'
        dict_series['abscissa'] = [0, 1, 2]
        dict_series['ordinate'] = [10, 100, 1000]
        dict_series['display_type'] = 'bar'
        dict_series['visible'] = False
        dict_series['y_axis'] = 'y'
        dict_series['custom_data'] = ''
        dict_series['marker_symbol'] = 'circle'

        dict_obj['series'] = [dict_series]

        chart_template = TwoAxesChartTemplate()
        chart_template.from_dict(dict_obj)

        self.assertDictEqual(dict_obj, chart_template.to_dict(
        ), f'Data structure is not same anymore between reference and model')

    def test_04_create_2D_axes_chart(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_bar = [index for index in range(20)]
        y_bar = np.random.randint(100, size=20).tolist()

        x_scatter = [index for index in range(20)]
        y_scatter = np.random.randint(50, size=20).tolist()

        x_line = [index for index in range(20)]
        y_line = np.random.randint(50, size=20).tolist()

        serie_bar = InstanciatedSeries(
            x_bar, y_bar, 'bar serie', InstanciatedSeries.BAR_DISPLAY)
        serie_lines = InstanciatedSeries(
            x_line, y_line, 'line serie', InstanciatedSeries.LINES_DISPLAY)
        serie_scatter = InstanciatedSeries(
            x_scatter, y_scatter, 'scatter serie', InstanciatedSeries.SCATTER_DISPLAY, marker_symbol='cross')

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        max_y = max([max(y_bar), max(y_scatter)])
        min_y = min([min(y_bar), min(y_scatter)])

        chart = TwoAxesInstanciatedChart('index', 'quantity', [0, 20], [
                                         min_y, max_y], 'Random quantity chart')

        chart.add_series(serie_lines)
        chart.add_series(serie_scatter)
        chart.add_series(serie_bar)
        chart.to_plotly()


    def test_05_create_2D_axes_cumulative_bar_chart(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries

        x_bar_1 = [index for index in range(20)]
        y_bar_1 = [index for index in range(20)]

        x_bar_2 = [index for index in range(20)]
        y_bar_2 = [index * 0.5 for index in range(20)]

        x_bar_3 = [index for index in range(20)]
        y_bar_3 = [index * 2 for index in range(20)]

        serie_bar_1 = InstanciatedSeries(
            x_bar_1, y_bar_1, 'bar serie 1', InstanciatedSeries.BAR_DISPLAY)
        serie_bar_2 = InstanciatedSeries(
            x_bar_2, y_bar_2, 'bar serie 2', InstanciatedSeries.BAR_DISPLAY)
        serie_bar_3 = InstanciatedSeries(
            x_bar_3, y_bar_3, 'bar serie 3', InstanciatedSeries.BAR_DISPLAY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        max_y = max(y_bar_1) + max(y_bar_2) + max(y_bar_3)
        min_y = 0

        chart = TwoAxesInstanciatedChart('index', 'quantity', [0, 20], [
                                         min_y, max_y], 'Stacked quantity chart', stacked_bar=True)

        chart.add_series(serie_bar_1)
        chart.add_series(serie_bar_2)
        chart.add_series(serie_bar_3)
        chart.to_plotly()

    def test_06_create_2D_axes_cumulative_surface_chart(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries

        x_serie_1 = [index for index in range(20)]
        y_serie_1 = [index for index in range(20)]

        x_serie_2 = [index for index in range(20)]
        y_serie_2 = [index * 0.5 for index in range(20)]

        x_serie_3 = [index for index in range(20)]
        y_serie_3 = [index * 2 for index in range(20)]

        serie_serie_1 = InstanciatedSeries(
            x_serie_1, y_serie_1, 'serie 1', InstanciatedSeries.BAR_DISPLAY)
        serie_serie_2 = InstanciatedSeries(
            x_serie_2, y_serie_2, 'serie 2')
        serie_serie_3 = InstanciatedSeries(
            x_serie_3, y_serie_3, 'serie 3', InstanciatedSeries.LINES_DISPLAY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        max_y = max(y_serie_1) + max(y_serie_2) + max(y_serie_3)
        min_y = 0

        chart = TwoAxesInstanciatedChart('index', 'quantity', [0, 20], [
                                         min_y, max_y], 'Cumulative surface quantity chart', cumulative_surface=True)

        chart.add_series(serie_serie_1)
        chart.add_series(serie_serie_2)
        chart.add_series(serie_serie_3)
        chart.to_plotly()

    def test_07_create_2D_axes_cumulative_bar_with_scatter_chart(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_bar_1 = [index for index in range(20)]
        y_bar_1 = [index for index in range(20)]

        x_line_2 = [index for index in range(20)]
        y_line_2 = np.random.randint(50, size=20).tolist()

        x_bar_3 = [index for index in range(20)]
        y_bar_3 = [index * 2 for index in range(20)]

        serie_bar_1 = InstanciatedSeries(
            x_bar_1, y_bar_1, 'bar serie 1', InstanciatedSeries.BAR_DISPLAY)
        serie_line_2 = InstanciatedSeries(
            x_line_2, y_line_2, 'scatter serie 2', InstanciatedSeries.LINES_DISPLAY)
        serie_bar_3 = InstanciatedSeries(
            x_bar_3, y_bar_3, 'bar serie 3', InstanciatedSeries.BAR_DISPLAY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        max_y = max(y_bar_1) + max(y_line_2) + max(y_bar_3)
        min_y = 0

        chart = TwoAxesInstanciatedChart('index', 'quantity', [0, 20], [
                                         min_y, max_y], 'Stacked quantity chart', stacked_bar=True)

        chart.add_series(serie_bar_1)
        chart.add_series(serie_line_2)
        chart.add_series(serie_bar_3)
        chart.to_plotly()

    def test_08_create_2D_axes_chart_with_annotations(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_bar = [index for index in range(20)]
        y_bar = np.random.randint(100, size=20).tolist()

        x_line = [index for index in range(20)]
        y_line = np.random.randint(50, size=20).tolist()

        serie_bar = InstanciatedSeries(
            x_bar, y_bar, 'bar serie', InstanciatedSeries.BAR_DISPLAY)
        serie_line = InstanciatedSeries(
            x_line, y_line, 'line serie', InstanciatedSeries.LINES_DISPLAY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, TwoAxesChartTemplate

        max_y = max([max(y_bar), max(y_line)])
        min_y = min([min(y_bar), min(y_line)])

        chart = TwoAxesInstanciatedChart('index', 'quantity', [
                                         0, 20], [min_y, max_y], 'Random quantity chart')

        chart.add_annotation(
            TwoAxesChartTemplate.ANNOTATION_UPPER_LEFT, 'left annotation', 'example of value')
        chart.add_annotation(TwoAxesChartTemplate.ANNOTATION_UPPER_RIGHT,
                             'Right annotation', 'example of value')
        chart.add_annotation(TwoAxesChartTemplate.ANNOTATION_UPPER_RIGHT,
                             'Another right annotation', 'another example of value')

        chart.add_series(serie_bar)
        chart.add_series(serie_line)
        chart.to_plotly()

    def test_09_create_pie_chart(self):

        from sos_trades_core.tools.post_processing.pie_charts.instanciated_pie_chart import InstanciatedPieChart
        import numpy as np

        x_scatter = [f'value {index}' for index in range(5)]
        y_scatter = np.random.randint(50, size=5).tolist()

        chart = InstanciatedPieChart(
            'Quantity pie chart', x_scatter, y_scatter)
        chart.to_plotly()

    def test_10_create_table(self):

        from sos_trades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
        import numpy as np

        headers = ['Q1', 'Q2', 'Q3', 'Q4']
        values_by_row = []

        for row in range(len(headers)):
            values_by_row.append(np.random.randint(1000, size=20).tolist())

        table = InstanciatedTable('Table example', headers, values_by_row)
        table.to_plotly()

    def test_11_create_2D_axes_horizontal_bar_chart(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries

        x_bar_1 = [index for index in range(20)]
        y_bar_1 = [index for index in range(20)]

        x_bar_2 = [index * -1 for index in range(20)]
        y_bar_2 = [index for index in range(20)]

        serie_bar_1 = InstanciatedSeries(
            x_bar_1, y_bar_1, 'bar serie 1', InstanciatedSeries.BAR_DISPLAY)
        serie_bar_2 = InstanciatedSeries(
            x_bar_2, y_bar_2, 'bar serie 2', InstanciatedSeries.BAR_DISPLAY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        chart = TwoAxesInstanciatedChart('index', 'quantity', [-20, 20], [
                                         0, 20], 'Horizontal bar chart', bar_orientation='h')

        chart.add_series(serie_bar_1)
        chart.add_series(serie_bar_2)
        chart.to_plotly()

    def test_12_create_pie_chart_with_annotations(self):

        from sos_trades_core.tools.post_processing.pie_charts.instanciated_pie_chart import InstanciatedPieChart
        import numpy as np

        x_scatter = [f'value {index}' for index in range(5)]
        y_scatter = np.random.randint(50, size=5).tolist()

        chart = InstanciatedPieChart(
            'Quantity pie chart', x_scatter, y_scatter)

        chart.add_annotation(
            InstanciatedPieChart.ANNOTATION_UPPER_LEFT, 'left annotation', 'example of value')
        chart.add_annotation(InstanciatedPieChart.ANNOTATION_UPPER_RIGHT,
                             'Right annotation', 'example of value')
        chart.add_annotation(InstanciatedPieChart.ANNOTATION_UPPER_RIGHT,
                             'Another right annotation', 'another example of value')

        chart.to_plotly()

    def test_13_create_table_with_annotations(self):

        from sos_trades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
        import numpy as np

        headers = ['Q1', 'Q2', 'Q3', 'Q4']
        values_by_row = []

        for row in range(len(headers)):
            values_by_row.append(np.random.randint(1000, size=20).tolist())

        table = InstanciatedTable('Table example', headers, values_by_row)

        table.add_annotation(
            InstanciatedTable.ANNOTATION_UPPER_LEFT, 'left annotation', 'example of value')
        table.add_annotation(InstanciatedTable.ANNOTATION_UPPER_RIGHT,
                             'Right annotation', 'example of value')
        table.add_annotation(InstanciatedTable.ANNOTATION_UPPER_RIGHT,
                             'Another right annotation', 'another example of value')

        table.to_plotly()

    def test_14_create_2D_axes_chart_with_secondary_ordinate(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_primary = [index for index in range(20)]
        y_primary = np.random.randint(100, size=20).tolist()

        x_secondary = [index for index in range(20)]
        y_secondary = np.random.randint(50, size=20).tolist()

        serie_primary = InstanciatedSeries(
            x_primary, y_primary, 'scatter serie primary axe', InstanciatedSeries.LINES_DISPLAY)
        serie_secondary = InstanciatedSeries(
            x_secondary, y_secondary, 'scatter serie secondary axe', InstanciatedSeries.LINES_DISPLAY,
            y_axis=InstanciatedSeries.Y_AXIS_SECONDARY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        chart = TwoAxesInstanciatedChart('index', 'quantity primary', [0, 20], [
                                         min(y_primary), max(y_primary)], 'Random quantity chart',
                                         secondary_ordinate_axis_name='quantity secondary',
                                         secondary_ordinate_axis_range=[min(y_secondary), max(y_secondary)])

        chart.add_series(serie_primary)
        chart.add_series(serie_secondary)
        chart.to_plotly()

    def test_15_create_2D_axes_chart_with_secondary_ordinate_mixing_series_type(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_primary = [index for index in range(20)]
        y_primary = np.random.randint(100, size=20).tolist()

        x_secondary = [index for index in range(20)]
        y_secondary = np.random.randint(50, size=20).tolist()

        serie_primary = InstanciatedSeries(
            x_primary, y_primary, 'scatter serie primary axe', InstanciatedSeries.BAR_DISPLAY)
        serie_secondary = InstanciatedSeries(
            x_secondary, y_secondary, 'scatter serie secondary axe', InstanciatedSeries.LINES_DISPLAY, y_axis=InstanciatedSeries.Y_AXIS_SECONDARY)

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart

        chart = TwoAxesInstanciatedChart('index', 'quantity primary', [0, 20], [
                                         min(y_primary), max(y_primary)], 'Random quantity chart',
                                         secondary_ordinate_axis_name='quantity secondary', secondary_ordinate_axis_range=[min(y_secondary), max(y_secondary)])

        chart.add_series(serie_primary)
        chart.add_series(serie_secondary)

        chart.to_plotly()

    def test_16_check_nan_values(self):

        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries
        import numpy as np

        x_bar = [1.0, 'test', 40, np.nan, None, np.Infinity]
        y_bar = np.random.randint(100, size=5).tolist()

        serie_bar = InstanciatedSeries(
            x_bar, y_bar, 'bar serie', InstanciatedSeries.BAR_DISPLAY)

        filtered_abscissa_nan = serie_bar.abscissa_filtered()[3]
        filtered_abscissa_infinite = serie_bar.abscissa_filtered()[5]

        self.assertEqual(
            None, filtered_abscissa_nan, 'NaN value not converted to None value')
        self.assertEqual(
            None, filtered_abscissa_infinite, 'Infinite value not converted to None value')

    def test_17_create_pareto_front_optimal_chart(self):

        from sos_trades_core.tools.post_processing.pareto_front_optimal_charts.instanciated_pareto_front_optimal_chart \
            import InstantiatedParetoFrontOptimalChart
        from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart \
            import InstanciatedSeries

        chart_name = 'Test Pareto front optimal'
        new_pareto_chart = InstantiatedParetoFrontOptimalChart(
            abscissa_axis_name=f'abscissa (unit)',
            primary_ordinate_axis_name=f'ordinate (unit)',
            abscissa_axis_range=[- 1.0, 3.0],
            primary_ordinate_axis_range=[- 1.0, 40.0],
            chart_name=chart_name)
        # Add series
        new_serie = InstanciatedSeries(
            [0], [1], 'Scenario_1', 'scatter')
        new_pareto_chart.add_serie(new_serie)

        new_serie = InstanciatedSeries(
            [2], [30], 'Scenario_2', 'scatter')
        new_pareto_chart.add_serie(new_serie)

        new_serie = InstanciatedSeries(
            [1], [20], 'Scenario_3', 'scatter')
        new_pareto_chart.add_serie(new_serie)

        new_serie = InstanciatedSeries(
            [0], [20], 'Scenario_4', 'scatter')
        new_pareto_chart.add_serie(new_serie)
        # Add pareto front
        pareto_front_serie = InstanciatedSeries(
            [0, 2], [1, 30], 'Pareto front', 'lines')
        new_pareto_chart.add_pareto_front_optimal(pareto_front_serie)

        # new_pareto_chart.to_plotly().show()
        new_pareto_chart.to_plotly()

    def test_18_create_spider_chart(self):

        from sos_trades_core.tools.post_processing.spider_charts.instantiated_spider_chart \
            import InstantiatedSpiderChart, SpiderChartTrace
        chart_name = 'Test Spider Chart'
        new_spider_chart = InstantiatedSpiderChart(chart_name)
        # Add traces
        theta_values = ['Axis_1_name', 'Axis_2_name', 'Axis_3_name']
        # Scenario 1
        axis_1 = {'value': 20, 'text': 'Value 20 [m€]'}
        axis_2 = {'value': 50, 'text': 'Value 50 [%]'}
        axis_3 = {'value': 70, 'text': 'Value 20 [%]'}
        radius_values = [axis_1, axis_2, axis_3]
        new_spider_trace = SpiderChartTrace(
            'Scenario_1', theta_values, radius_values)
        new_spider_chart.add_trace(new_spider_trace)
        # Scenario 2
        axis_1 = {'value': 42, 'text': 'Value 42 [m€]'}
        axis_2 = {'value': 20, 'text': 'Value 20 [%]'}
        axis_3 = {'value': 42, 'text': 'Value 42 [%]'}
        radius_values = [axis_1, axis_2, axis_3]
        new_spider_trace = SpiderChartTrace(
            'Scenario_2', theta_values, radius_values)
        new_spider_chart.add_trace(new_spider_trace)

        # new_spider_chart.to_plotly().show()
        new_spider_chart.to_plotly()

    def test_19_create_parallel_coordinates_chart(self):

        from sos_trades_core.tools.post_processing.parallel_coordinates_charts.instantiated_parallel_coordinates_chart \
            import InstantiatedParallelCoordinatesChart, ParallelCoordinatesTrace

        chart_name = 'Test Parallel Coordinates chart'
        parallel_coordinates_chart = InstantiatedParallelCoordinatesChart(
            chart_name)

        new_parallel_coordinates_trace = ParallelCoordinatesTrace('Scenarios',
                                                                  ['Scenario_1', 'Scenario_2',
                                                                      'Scenario_3'],
                                                                  ParallelCoordinatesTrace.TRACE_TEXT)
        parallel_coordinates_chart.add_trace(new_parallel_coordinates_trace)

        new_parallel_coordinates_trace = ParallelCoordinatesTrace(
            'Scale_1 [m $]', [2, 8, 5])
        parallel_coordinates_chart.add_trace(new_parallel_coordinates_trace)

        new_parallel_coordinates_trace = ParallelCoordinatesTrace(
            'Scale_2 [Gt]', [1625, 556, 298])
        parallel_coordinates_chart.add_trace(new_parallel_coordinates_trace)

        new_parallel_coordinates_trace = ParallelCoordinatesTrace('Scale_3 [%]', [
                                                                  0.2, 0.8, 1.4])
        parallel_coordinates_chart.add_trace(new_parallel_coordinates_trace)

        # parallel_coordinates_chart.to_plotly().show()
        parallel_coordinates_chart.to_plotly()

    def test_20_create_plotly_native_chart(self):

        from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart \
            import InstantiatedPlotlyNativeChart
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        t = np.linspace(0, 10, 100)
        y = np.sin(t)

        fig.add_trace(go.Scatter(x=t, y=y, mode='markers'))

        chart_name = 'Test Plotly native chart'
        plotly_native_chart = InstantiatedPlotlyNativeChart(fig, chart_name)

        # plotly_native_chart.to_plotly().show()
        plotly_native_chart.to_plotly()
