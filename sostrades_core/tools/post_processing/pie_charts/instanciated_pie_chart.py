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
Class that define a pie chart display as post post processing
"""
import plotly.graph_objects as go
from sostrades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling
from sostrades_core.tools.post_processing.post_processing_tools import escape_str_with_comma


class InstanciatedPieChartException(Exception):
    """ Overload Exception basic type 
    """


class InstanciatedPieChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define a pie chart display as post post processing 
    """

    PIE_CHART_NAME = 'pie_chart_name'
    LABELS = 'labels'
    VALUES = 'values'
    STYLES = 'styles'

    def __init__(self, pie_chart_name='', labels=[], values=[]):
        """ Create a new table

        @param pie_chart_name : string that contains pie chart name
        @param labels : string list that contains labels for each pie chart values
        @param values : string list of list that contains each value the pie chart
        """
        super().__init__()

        # Set the pie chart name
        self.pie_chart_name = pie_chart_name

        # Pie chart datas
        if not isinstance(labels, list):
            message = f'"labels" argument is intended to be a list not {type(labels)}'
            raise TypeError(message)
        self.labels = labels

        if not isinstance(values, list):
            message = f'"values" argument is intended to be a list not {type(values)}'
            raise TypeError(message)
        self.values = values

    def __repr__(self):
        """ Overload of the class representation

        @return str, string representation of the instance
        """
        result = f' \nname: {self.pie_chart_name} \n'
        result += f'header: {self.labels} \n'
        result += f'cells: {self.values}\n'
        result += f'annotations\n{super().__repr__()}'

        return result

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """
        pie_chart = go.Pie(labels=self.labels, values=self.values, sort=False)

        fig = go.Figure(data=[pie_chart])

        # -- Annotations management
        chart_annotations = []
        # Upper left annotations
        upper_left_annotations = self.get_default_annotations_upper_left_layout(
            pos_x=-0.12, pos_y=1.15)
        if len(upper_left_annotations.keys()) > 0:
            chart_annotations.append(upper_left_annotations)

        # Upper right annotations
        upper_right_annotations = self.get_default_annotations_upper_right_layout(
            pos_x=-1.2, pos_y=1.15)
        if len(upper_right_annotations.keys()) > 0:
            chart_annotations.append(upper_right_annotations)

        layout = {}
        layout.update({'width': 600})
        layout.update({'height': 450})
        layout.update({'autosize': False})
        layout.update(
            {'title': self.get_default_title_layout(self.pie_chart_name)})
        layout.update({'font': self.get_default_font_layout()})
        layout.update({'annotations': chart_annotations})

        fig.update_layout(layout)

        return fig

    def __to_csv(self):
        label_text_list = [escape_str_with_comma(
            str(lb)) for lb in self.labels]
        values_text_list = [escape_str_with_comma(
            str(vl)) for vl in self.values]

        csv_list = [','.join(label_text_list), ','.join(values_text_list)]

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
