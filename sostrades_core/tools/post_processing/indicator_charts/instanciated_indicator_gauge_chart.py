'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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
Class that define a pie chart display as post post processing
"""


class InstanciatedIndicatorChartException(Exception):
    """ Overload Exception basic type
    """


class InstanciatedIndiactorChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define a pie chart display as post post processing
    """

    CHART_NAME = 'chart_name'
    VALUE = 'value'



    def __init__(self, chart_name='', value = [], mode = '', title ={}, gauge = {}):
        """ Create a new table

        @param pie_chart_name : string that contains pie chart name
        @param labels : string list that contains labels for each pie chart values
        @param values : string list of list that contains each value the pie chart
        """
        super().__init__()

        # Set the pie chart name
        self.chart_name = chart_name

        # Pie chart datas
        if not isinstance(value, float):
            message = f'"values" argument is intended to be a float not {type(value)}'
            raise TypeError(message)
        self.value = value
        self.title = title
        self.gauge = gauge
        self.mode = mode


    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """

        indicator = go.Indicator(value=self.value, mode=self.mode, title=self.title, gauge=self.gauge)
        fig = go.Figure(data=[indicator])
        return fig

    def to_plotly_dict(self, logger=None):
        """ Method that convert current instance to plotly object and then to a dictionary

        @param logger: logger instance
        @type Logging.loger
        """
        json = self.to_plotly(logger).to_dict()
        json[self.CSV_DATA] = self.value
        json[self.LOGO_NOTOFFICIAL] = self.logo_notofficial
        json[self.LOGO_OFFICIAL] = self.logo_official
        json[self.LOGO_WORK_IN_PROGRESS] = self.logo_work_in_progress
        json[self.POST_PROCESSING_SECTION_NAME] = self.post_processing_section_name.capitalize()

        return json
