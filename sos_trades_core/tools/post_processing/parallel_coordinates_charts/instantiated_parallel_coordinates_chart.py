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
Class that define a parallel coordinates chart display as post post processing
"""
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import escape_str_with_comma
from sos_trades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling


class ParallelCoordinatesTrace:
    """ Class that define parallel coordinate chart trace
    """
    TRACE_TEXT = 'text'
    TRACE_NUMBER = 'number'

    def __init__(self, trace_name='', trace_values=[], trace_type=TRACE_NUMBER):
        """  Init of the class

        @param trace_name: name of the trace
        @type str

        @param trace_values: values of each vertical axis
        @type list

        @param trace_type: type of the trace (TRACE_TEXT or TRACE_NUMBER)
        @type str
        """

        self.trace_name = trace_name

        if not isinstance(trace_values, list):
            message = f'"trace_values" argument is intended to be a list not {type(trace_values)}'
            raise TypeError(message)

        if not (trace_type == self.TRACE_NUMBER or trace_type == self.TRACE_TEXT):
            message = f'"trace_type" argument is intended to be "number" or "text"'
            raise TypeError(message)

        self.trace_values = trace_values
        self.trace_type = trace_type


class InstantiatedParallelCoordinatesChart(AbstractPostProcessingPlotlyTooling):
    """ Class that define parallel coordinates display as post post processing
    """

    def __init__(self, chart_name=''):
        """  Init of the class

        @param chart_name: name of the chart
        @type str
        """
        super().__init__()

        # List of traces
        self.__traces = []

        # Chart name
        self.chart_name = chart_name

    def add_trace(self, trace):
        """ Private method to add trace to current parallel coordinates chart

        @param trace: trace instance to add
        @type ParallelCoordinatesTrace
        """
        if not isinstance(trace, ParallelCoordinatesTrace):
            message = f'"trace" argument is intended to be a ParallelCoordinatesTrace not {type(trace)}'
            raise TypeError(message)

        # Check if trace with text already in list
        if trace.trace_type == ParallelCoordinatesTrace.TRACE_TEXT:
            if len(list(filter(lambda tr: tr.trace_type == ParallelCoordinatesTrace.TRACE_TEXT, self.__traces))) > 0:
                message = f'You have already set a trace with trace_type text, only one is allowed for the chart'
                raise TypeError(message)

        self.__traces.append(trace)

    def to_plotly(self, logger=None):
        """ Convert current instance into a plotly object

        @param logger: logging object to log message
        @type Logging.logger

        @return plotly.graph_objects.go instance
        """
        pc_dimensions = []

        # First add number traces
        for trace in self.__traces:
            if trace.trace_type == ParallelCoordinatesTrace.TRACE_NUMBER:
                range_values = abs(max(trace.trace_values)) - \
                    abs(min(trace.trace_values))
                pc_dimensions.append(dict(label=trace.trace_name,
                                          values=trace.trace_values,
                                          range=[min(trace.trace_values) - 0.10 * range_values,
                                                 max(trace.trace_values) + 0.10 * range_values]))

        # Second add text traces
        line_config = dict(autocolorscale=True,
                           showscale=False)

        for trace in self.__traces:
            if trace.trace_type == ParallelCoordinatesTrace.TRACE_TEXT:
                id_values = []
                tick_texts = []
                for index, tick_text in enumerate(trace.trace_values, start=1):
                    id_values.append(index)
                    tick_texts.append(tick_text)

                pc_dimensions.append(dict(label=trace.trace_name,
                                          values=id_values,
                                          tickvals=id_values,
                                          ticktext=tick_texts))
                line_config['color'] = id_values

        fig = go.Figure(data=go.Parcoords(
            line=line_config,
            dimensions=list(pc_dimensions)
        )
        )

        layout = {}
        layout.update(
            {'title': self.get_default_title_layout(self.chart_name)})
        layout.update({'width': 600})
        layout.update({'height': 450})
        layout.update({'autosize': False})
        layout.update({'font': self.get_default_font_layout()})

        fig.update_layout(layout)

        return fig

    def __to_csv(self):
        global_list = []
        header = []
        max_len = 0

        for trace in self.__traces:
            if trace.trace_name is not None and len(trace.trace_name):
                header.append(escape_str_with_comma(f'{trace.trace_name}'))

            global_list.append(trace.trace_values)
            if len(trace.trace_values) > max_len:
                max_len = len(trace.trace_values)

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
