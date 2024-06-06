'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
import plotly.graph_objects as go

from sostrades_core.tools.post_processing.post_processing_plotly_tooling import (
    AbstractPostProcessingPlotlyTooling,
)

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Class that define a spider chart display as post post processing
"""


class SpiderChartTrace:
    """ Class that define spider chart trace
    """

    def __init__(self, trace_name='', theta_values=[], radius_values=[]):
        """  Init of the class

        @param trace_name, name of the trace
        @param str

        @param theta_values, values of spider chart axis = Name of the axis
        @param list

        @param radius_values, values of spider chart on radius with value as text
        @type list
        """

        self.trace_name = trace_name

        if not isinstance(theta_values, list):
            message = f'"theta_values" argument is intended to be a list not {type(theta_values)}'
            raise TypeError(message)
        self.theta_values = theta_values

        if not isinstance(radius_values, list):
            message = f'"radius_values" argument is intended to be a list not {type(radius_values)}'
            raise TypeError(message)
        self.radius_values = radius_values

        if len(self.theta_values) != len(self.radius_values):
            message = f'"theta_values" and "radius_values" must have same length ' \
                      f'{type(theta_values)} != {len(radius_values)}'
            raise ValueError(message)


class InstantiatedSpiderChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define spider chart display as post post processing
    """

    def __init__(self, chart_name=''):
        """  Init of the class

        @param chart_name: name of the chart
        @type str
        """
        super().__init__()

        self.__traces = []

        # Chart name
        self.chart_name = chart_name

    def add_trace(self, trace):
        """  Method to add trace to current spider chart

        @param trace: trace instance to add
        @type SpiderChartTrace
        """
        if not isinstance(trace, SpiderChartTrace):
            message = f'"trace" argument is intended to be a SpiderChartTrace not {type(trace)}'
            raise TypeError(message)
        self.__traces.append(trace)

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """
        fig = go.Figure()

        for trace in self.__traces:
            radius_values = trace.radius_values
            theta_values = trace.theta_values
            # Adding last point to close lines
            radius_values.append(trace.radius_values[0])
            theta_values.append(trace.theta_values[0])

            fig.add_trace(go.Scatterpolar(
                name=trace.trace_name,
                r=[rad['value'] for rad in radius_values],
                text=[rad['text'] for rad in radius_values],
                theta=theta_values,
                mode='lines'
            ))

        layout = {}
        layout.update(
            {'title': self.get_default_title_layout(self.chart_name)})
        layout.update({'width': 600})
        layout.update({'height': 450})
        layout.update({'autosize': False})
        layout.update({'font': self.get_default_font_layout()})

        fig.update_layout(layout)

        return fig

    def to_plotly_dict(self, logger=None):
        """ Method that convert current instance to plotly object and then to a dictionary

        @param logger: logger instance
        @type Logging.loger
        """
        json = self.to_plotly(logger).to_dict()
        json[self.CSV_DATA] = self._plot_csv_data
        json[self.LOGO_NOTOFFICIAL] = self.logo_notofficial
        json[self.LOGO_OFFICIAL] = self.logo_official
        json[self.LOGO_WORK_IN_PROGRESS] = self.logo_work_in_progress

        return json
