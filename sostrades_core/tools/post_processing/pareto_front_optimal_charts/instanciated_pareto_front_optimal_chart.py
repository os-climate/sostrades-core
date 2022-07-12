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
Class that define a pareto front optimal chart display as post post processing
"""
import plotly.graph_objects as go

from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, \
    InstanciatedSeriesException
from sostrades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling


class InstantiatedParetoFrontOptimalChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define pareto front optimal chart display as post post processing
    """

    def __init__(self, abscissa_axis_name='', primary_ordinate_axis_name='', abscissa_axis_range=[],
                 primary_ordinate_axis_range=[], chart_name=''):
        """ Create a new chart definition

        :params: abscissa_axis_name : string that contains chart abscissa axis name
        :type: str
        :params: primary_ordinate_axis_name : string that contains chart primary ordinate axis name
        :type: str
        :params: abscissa_axis_range : array(2) with min and max value range for abscissa axes
        :type: list [min, max]
        :params: primary_ordinate_axis_range : array(2) with min and max value range for primary ordinate axes
        :type: list [min, max]
        :params: chart_name : string that contains chart name
        :type: str
        """
        super().__init__()
        # Host the list of series for this chart
        self.__series = []

        # Chart axis name
        self.abscissa_axis_name = abscissa_axis_name
        self.primary_ordinate_axis_name = primary_ordinate_axis_name

        # Axis value range
        # two value min range (index 0) and max range (index 1)
        self.abscissa_axis_range = abscissa_axis_range
        self.primary_ordinate_axis_range = primary_ordinate_axis_range

        # Chart name
        self.chart_name = chart_name

        # Initialize annotation properties
        self.annotation_upper_left = {}
        self.annotation_upper_right = {}

    def __add_serie_to_chart(self, serie):
        """ Private method to add serie to current pareto chart

        :params: serie, serie instance to add
        :type: InstanciatedSeries
        """
        if isinstance(serie, InstanciatedSeries):
            self.__series.append(serie)
        else:
            raise InstanciatedSeriesException(
                f'given series has the wrong type, {type(serie)} instead of InstanciatedSeries')

    def add_serie(self, serie):
        """ Method to add scatter serie

        :params: serie, serie instance to add
        :type: InstanciatedSeries
        """
        if isinstance(serie, InstanciatedSeries) and serie.display_type == InstanciatedSeries.SCATTER_DISPLAY:
            self.__add_serie_to_chart(serie)
        else:
            raise InstanciatedSeriesException(
                f'given series has the wrong type, {type(serie)} '
                f'instead of InstanciatedSeries or has an invalid display type,'
                f' different of scatter : {serie.display_type}')

    def add_pareto_front_optimal(self, serie):
        """ Method to add line serie, corresponding to pareto optimal front

        :params: serie, serie instance to add
        :type: InstanciatedSeries
        """
        if isinstance(serie, InstanciatedSeries) and serie.display_type == InstanciatedSeries.LINES_DISPLAY:
            self.__add_serie_to_chart(serie)
        else:
            raise InstanciatedSeriesException(
                f'given series has the wrong type, {type(serie)} '
                f'instead of InstanciatedSeries or has an invalid display type, '
                f'different of lines : {serie.display_type}')

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """

        fig = go.Figure()

        # Manage series to be added into plotly object
        for serie in self.__series:

            abscissa = serie.abscissa
            ordinate = serie.ordinate

            if serie.display_type == InstanciatedSeries.SCATTER_DISPLAY:
                fig.add_trace(go.Scatter(x=abscissa, y=ordinate, name=serie.series_name,
                                         visible=True if serie.visible else 'legendonly', mode='markers',
                                         yaxis=serie.y_axis, customdata=[serie.custom_data]))

            elif serie.display_type == InstanciatedSeries.LINES_DISPLAY:
                fig.add_trace(go.Scatter(x=abscissa, y=ordinate, name=serie.series_name,
                                         visible=True if serie.visible else 'legendonly', mode='lines', yaxis=serie.y_axis, hoverinfo='none'))

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

        layout = {}
        layout.update({'barmode': 'group'})
        layout.update(
            {'title': self.get_default_title_layout(self.chart_name)})
        layout.update({'xaxis': xaxis})
        layout.update({'yaxis': yaxis})
        layout.update({'width': 600})
        layout.update({'height': 450})
        layout.update({'autosize': False})
        layout.update({'showlegend': False})
        layout.update({'font': self.get_default_font_layout()})
        layout.update({'annotations': chart_annotations})

        fig.update_layout(layout)

        return fig

    def __to_csv(self):
        global_list = []
        header = []
        max_len = 0

        for serie in self.__series:
            if serie.series_name is not None and len(serie.series_name) > 0:
                header.append(f'{serie.series_name} {self.abscissa_axis_name}')
                header.append(
                    f'{serie.series_name} {self.primary_ordinate_axis_name}')
            else:
                header.append(f'{self.abscissa_axis_name}')
                header.append(f'{self.primary_ordinate_axis_name}')

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
                    csv_line.append(f'{gl[i]}')
                else:
                    csv_line.append('')
            csv_list.append(','.join(csv_line))

        self.set_csv_data(csv_list)

    def to_plotly_dict(self, logger=None):
        """ Method that convert current instance to plotly object and then to a dictionary

        @param logger: logger instance
        @type Logging.loger
        """
        json = self.to_plotly(logger).to_dict()

        if self._plot_csv_data is None:
            self.__to_csv()

        json[self.CSV_DATA] = self._plot_csv_data
        json[self.LOGO_NOTOFFICIAL] = self.logo_notofficial
        json[self.LOGO_OFFICIAL] = self.logo_official
        json[self.LOGO_WORK_IN_PROGRESS] = self.logo_work_in_progress

        return json
