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
Class that define a 2 dimensional instantiated chart 
"""
import plotly.graph_objects as go
from sostrades_core.tools.post_processing.post_processing_tools import escape_str_with_comma
from sostrades_core.tools.post_processing.charts.two_axes_chart_template import TwoAxesChartTemplate, SeriesTemplate


class InstanciatedSeriesException(Exception):
    """ Overload Exception basic type 
    """


class InstanciatedSeries (SeriesTemplate):
    """ Class that define a series abscissa and ordinate list with a name
    """


class TwoAxesInstanciatedChart(TwoAxesChartTemplate):
    """ Class that define a 2 dimensional chart template 
    """

    CUMULATIVE_TO_ZERO_Y = 'tozeroy'
    CUMULATIVE_TO_NEXT_Y = 'tonexty'

    def add_series(self, series):
        if isinstance(series, InstanciatedSeries):
            self.series.append(series)
        else:
            raise InstanciatedSeriesException(
                f'given series has the wrong type, {type(series)} instead of InstanciatedSeries')

    def to_plotly(self, logger=None):
        """
        Convert current instance into a plotly object

        :param logger: logging object to log message
        :type logger: Logging.logger

        :return plotly.graph_objects.go instance
        """

        fig = go.Figure()

        # -- Series and cumulative surface management

        # Variable used to build cumulated surface chart
        cumulated_dictionary = {}
        merged_abscissa = []

        if self.cumulative_surface == True:
            cumulative_surface_value = TwoAxesInstanciatedChart.CUMULATIVE_TO_ZERO_Y

            # Homogeneize ordinate for all series
            for serie in self.series:
                merged_abscissa.extend(serie.abscissa_filtered(logger))

        # Remove duplicate values
        merged_abscissa = list(dict.fromkeys(merged_abscissa))

        # Initialize accumulator
        for value in merged_abscissa:
            cumulated_dictionary[value] = 0

        # Manage series to be added into plotly object
        for serie in self.series:

            cumulated_values = []
            abscissa = []

            # first series to add (initialize cumulative surface value)
            if self.cumulative_surface == True and cumulative_surface_value == TwoAxesInstanciatedChart.CUMULATIVE_TO_ZERO_Y:

                abscissa = merged_abscissa

                abscissa_filtered = serie.abscissa_filtered(logger)
                ordinate_filtered = serie.ordinate_filtered(logger)
                for index in range(len(abscissa_filtered)):
                    cumulated_dictionary[abscissa_filtered[index]
                                         ] = ordinate_filtered[index]

                cumulated_values = list(cumulated_dictionary.values())

            elif self.cumulative_surface == True and cumulative_surface_value == TwoAxesInstanciatedChart.CUMULATIVE_TO_NEXT_Y:

                abscissa = merged_abscissa
                abscissa_filtered = serie.abscissa_filtered(logger)
                ordinate_filtered = serie.ordinate_filtered(logger)

                for index in range(len(abscissa_filtered)):
                    cumulated_dictionary[abscissa_filtered[index]
                                         ] += ordinate_filtered[index]

                cumulated_values = list(cumulated_dictionary.values())

            else:
                abscissa = serie.abscissa_filtered(logger)
                cumulated_values = serie.ordinate_filtered(logger)

            if self.cumulative_surface == True and \
                    (cumulative_surface_value == TwoAxesInstanciatedChart.CUMULATIVE_TO_ZERO_Y or cumulative_surface_value == TwoAxesInstanciatedChart.CUMULATIVE_TO_NEXT_Y):

                fig.add_trace(go.Scatter(x=abscissa, y=cumulated_values, name=serie.series_name, visible=True if serie.visible else 'legendonly',
                                         fill=cumulative_surface_value))

                if cumulative_surface_value == TwoAxesInstanciatedChart.CUMULATIVE_TO_ZERO_Y:
                    cumulative_surface_value = TwoAxesInstanciatedChart.CUMULATIVE_TO_NEXT_Y

            elif serie.display_type == InstanciatedSeries.SCATTER_DISPLAY:
                fig.add_trace(go.Scatter(x=abscissa, y=cumulated_values, name=serie.series_name,
                                         marker_symbol=serie.marker_symbol, mode='markers', yaxis=serie.y_axis,
                                         visible=True if serie.visible else 'legendonly'))
            elif serie.display_type == InstanciatedSeries.BAR_DISPLAY:
                fig.add_trace(go.Bar(x=abscissa, y=cumulated_values, name=serie.series_name,
                                     orientation=self.bar_orientation,
                                     visible=True if serie.visible else 'legendonly', yaxis=serie.y_axis))
            elif serie.display_type == InstanciatedSeries.LINES_DISPLAY:
                fig.add_trace(go.Scatter(x=abscissa, y=cumulated_values, name=serie.series_name,
                                        mode='lines', yaxis=serie.y_axis,
                                         visible=True if serie.visible else 'legendonly'))

        # -- Annotations management
        chart_annotations = []
        # Upper left annotations
        upper_left_annotations = self.get_default_annotations_upper_left_layout()
        if len(upper_left_annotations.keys()) > 0:
            chart_annotations.append(upper_left_annotations)

        # Upper right annotations
        upper_right_annotations = self.get_default_annotations_upper_right_layout()
        if len(upper_right_annotations.keys()) > 0:
            chart_annotations.append(upper_right_annotations)

        xaxis = {}
        if len(self.abscissa_axis_range) > 0:
            xaxis.update({'range': self.abscissa_axis_range})
        xaxis.update({'title': self.abscissa_axis_name})
        xaxis.update({'automargin': True})

        yaxis = {}
        if len(self.primary_ordinate_axis_range) > 0:
            yaxis.update({'range': self.primary_ordinate_axis_range})
        yaxis.update({'title': self.primary_ordinate_axis_name})
        yaxis.update({'automargin': True})
        if self.y_axis_log:
            fig.update_yaxes(type='log')

        yaxis2 = {}
        if len(self.secondary_ordinate_axis_range) > 0:
            yaxis2.update({'range': self.secondary_ordinate_axis_range})
        yaxis2.update({'title': self.secondary_ordinate_axis_name})
        yaxis2.update({'automargin': True})
        yaxis2.update({'anchor': 'x'})
        yaxis2.update({'overlaying': 'y'})
        yaxis2.update({'side': 'right'})

        layout = {}
        layout.update({'barmode': 'relative' if self.stacked_bar else 'group'})
        layout.update(
            {'title': self.get_default_title_layout(self.chart_name)})
        layout.update({'xaxis': xaxis})
        layout.update({'yaxis': yaxis})
        layout.update({'yaxis2': yaxis2})
        layout.update({'width': 600})
        layout.update({'height': 450})
        layout.update({'autosize': False})
        layout.update({'legend': self.get_default_legend_layout()})
        layout.update({'font': self.get_default_font_layout()})

        if len(chart_annotations) > 0:
            layout.update({'annotations': chart_annotations})

        fig.update_layout(layout)

        return fig

    def __to_csv(self):
        global_list = []
        header = []
        max_len = 0

        for serie in self.series:
            if serie.series_name is not None and len(serie.series_name) > 0:
                header.append(f'{escape_str_with_comma(serie.series_name)} '
                              f'{escape_str_with_comma(self.abscissa_axis_name)}')
                header.append(f'{escape_str_with_comma(serie.series_name)} '
                              f'{escape_str_with_comma(self.primary_ordinate_axis_name)}')
            else:
                header.append(
                    f'{escape_str_with_comma(self.abscissa_axis_name)}')
                header.append(
                    f'{escape_str_with_comma(self.primary_ordinate_axis_name)}')

            global_list.append(serie.abscissa)
            if len(serie.abscissa) > max_len:
                max_len = len(serie.abscissa)

            global_list.append(serie.ordinate)
            if len(serie.ordinate) > max_len:
                max_len = len(serie.ordinate)

        csv_list = [','.join(header)]

        for i in range(max_len):
            csv_line = []
            for gl in global_list:
                if i < len(gl):
                    csv_line.append(escape_str_with_comma(f'{gl[i]}'))
                else:
                    csv_line.append('')
            csv_list.append(','.join(csv_line))

        self.set_csv_data(csv_list)

    def to_plotly_dict(self, logger=None):
        """
         Method that convert current instance to plotly object and then to a dictionary

        :param logger: logger instance
        :type logger: Logging.loger
        """
        json = self.to_plotly(logger).to_dict()

        if self._plot_csv_data is None:
            self.__to_csv()

        json[self.CSV_DATA] = self._plot_csv_data
        json[self.LOGO_NOTOFFICIAL] = self.logo_notofficial
        json[self.LOGO_OFFICIAL] = self.logo_official
        json[self.LOGO_WORK_IN_PROGRESS] = self.logo_work_in_progress

        return json
