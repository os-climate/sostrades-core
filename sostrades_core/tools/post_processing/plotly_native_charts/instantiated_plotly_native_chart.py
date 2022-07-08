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
import pandas as pd
from sos_trades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling


class InstantiatedPlotlyNativeChartException(Exception):
    """ Overload Exception basic type
    """


class InstantiatedPlotlyNativeChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define a native plotly chart display as post post processing
    """

    def __init__(self, fig, chart_name='', default_title=True, default_legend=True,
                 default_font=True, with_default_annotations=True):
        """ Create a new chart definition

        :params: chart_name : string that contains chart name
        :type: str
        :params: default_title : add to fig default title layout
        :type: bool
        :params: default_legend : add to fig default legend layout
        :type: bool
        :params: default_font : add to fig default font layout
        :type: bool
        :params: with_default_annotations : add to fig default annotations layout
        :type: bool
        """
        super().__init__()

        # set plotly figure
        if fig is not None:
            self.__fig = fig
        else:
            raise InstantiatedPlotlyNativeChartException(
                'Fig parameter is mandatory, cannot be None')

        # base bool parameters for layout automatic application to plotly fig
        self.chart_name = chart_name
        self.default_title = default_title
        self.default_legend = default_legend
        self.default_font = default_font
        self.with_default_annotations = with_default_annotations

    @property
    def plotly_fig(self):
        return self.__fig

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """
        if self.default_title:
            self.__fig.update_layout(
                {'title': self.get_default_title_layout(self.chart_name)})

        if self.default_legend:
            self.__fig.update_layout(
                {'legend': self.get_default_legend_layout()})

        if self.default_font:
            self.__fig.update_layout({'font': self.get_default_font_layout()})

        if self.with_default_annotations:
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

            self.__fig.update_layout({'annotations': chart_annotations})

        return self.__fig

    def set_csv_data_from_dataframe(self, dataframe):
        if dataframe is not None:
            if isinstance(dataframe, pd.DataFrame):
                csv_data = dataframe.to_csv(index=False).splitlines()
                self.set_csv_data(csv_data)
            else:
                raise InstantiatedPlotlyNativeChartException(f'dataframe parameter is intended to pandas '
                                                             f'dataframe, not {type(dataframe)}')
        else:
            raise InstantiatedPlotlyNativeChartException(
                'dataframe parameter is mandatory, cannot be None')

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
