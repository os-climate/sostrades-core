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
Class that define a 2 dimensional chart template 
"""

# pylint: disable=line-too-long
from sostrades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling
from sostrades_core.tools.post_processing.post_processing_tools import convert_nan


class SeriesTemplateException(Exception):
    """ Overload Exception basic type 
    """


class SeriesTemplate:
    """ Class that define a series abscissa and ordinate list with a name
    """

    DISPLAY_TYPE_VALUES = ['lines', 'scatter', 'bar']

    SERIES_NAME = 'series_name'
    ABSCISSA = 'abscissa'
    ORDINATE = 'ordinate'
    DISPLAY_TYPE = 'display_type'
    VISIBLE = 'visible'
    Y_AXIS = 'y_axis'

    LINES_DISPLAY = 'lines'
    SCATTER_DISPLAY = 'scatter'
    BAR_DISPLAY = 'bar'

    Y_AXIS_PRIMARY = 'y'
    Y_AXIS_SECONDARY = 'y2'
    CUSTOM_DATA = 'custom_data'
    MARKER_SYMBOL = 'marker_symbol'

    def __init__(self, abscissa=[], ordinate=[], series_name='', display_type='lines', visible=True,
                 y_axis=Y_AXIS_PRIMARY, custom_data=[''], marker_symbol='circle'):
        """ Create a new series to add in a chart

        :param abscissa: list of number values for abscissa
        :type abscissa: list of number
        :param ordinate: list of number values for ordinate
        :type ordinate: list of number
        :param series_name: name of the series
        :type series_name: str
        :param display_type: type of display allowed for the series (cf. SeriesTemplate.DISPLAY_TYPE_VALUES)
        :type display_type: str
        :param visible: default visibility of the series
        :type visible: bool
        :param y_axis: default axis of the series
        :type y_axis: str
        :param custom_data: custom_data of the series
        :type custom_data: str
        :param marker_symbol: symbol to use to display point on chart ('circle' by default) see: https://plotly.com/python/marker-style/
        :type marker_symbol: str
        """

        self.__ordinate = []
        self.__abscissa = []

        self.series_name = series_name

        # Assign values via property
        self.abscissa = abscissa
        self.ordinate = ordinate

        if display_type not in SeriesTemplate.DISPLAY_TYPE_VALUES:
            message = f'"display_type" argument is intended to be one of those values {SeriesTemplate.DISPLAY_TYPE_VALUES} not {display_type}'
            raise TypeError(message)
        self.display_type = display_type

        self.visible = visible
        self.y_axis = y_axis

        self.custom_data = custom_data
        self.marker_symbol = marker_symbol

    @property
    def abscissa(self):
        return self.__abscissa

    @abscissa.setter
    def abscissa(self, values):
        if not isinstance(values, list):
            message = f'"abscissa" argument is intended to be a list not {type(values)}'
            raise TypeError(message)
        self.__abscissa = values

    def abscissa_filtered(self, logger=None):
        """
        return abscissa values filtered on NaN and Infinite values

        :param logger: logging system to use
        :type logger: logging.logger

        :return filtered list of values
        """

        return self.__filter_values(self.abscissa, 'Abscissa', logger)

    @property
    def ordinate(self):
        return self.__ordinate

    @ordinate.setter
    def ordinate(self, values):
        if not isinstance(values, list):
            message = f'"ordinate" argument is intended to be a list not {type(values)}'
            raise TypeError(message)
        self.__ordinate = values

    def ordinate_filtered(self, logger=None):
        """
        return ordinate values filtered on NaN and Infinite values

        :param logger: logging system to use
        :type logger: logging.logger

        :return filtered list of values
        """

        return self.__filter_values(self.ordinate, 'Ordinate', logger)

    def __filter_values(self, values, property_name, logger=None):
        """
        return values filtered on NaN and Infinite values

        :param values: values to filter
        :type values: list
        :param property_name: property filtered
        :type property_name: str
        :param logger: logging system to use
        :type logger: logging.logger
        :return filterred list of values
        """

        filtered_series, has_nan = convert_nan(values)

        if logger is not None and has_nan:
            logger.warn(
                f'{property_name} of series "{self.series_name}" contains NaN/Infinite values')

        return filtered_series

    def __repr__(self):
        """
        Overload of the class representation

        Allow to hide password_hash from serializer point of view

        :return str, string representation of the instance
        """

        series_string = [f'\nname: {self.series_name}',
                         f'abscissa: {self.abscissa}',
                         f'ordinate: {self.ordinate}',
                         f'display type: {self.display_type}\n',
                         f'visible: {self.visible}\n',
                         f'y_axis: {self.y_axis}\n',
                         f'custom_data: {self.custom_data}\n',
                         f'marker_symbol: {self.marker_symbol}\n'
                         ]

        return '\n'.join(series_string)

    def to_dict(self):
        """
        Method that serialize as dict the SeriesTemplate class

        :return dict
        """

        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({SeriesTemplate.SERIES_NAME: self.series_name})

        # Serialize abscissa parameter attribute
        dict_obj.update(
            {SeriesTemplate.ABSCISSA: self.abscissa})

        # Serialize ordinate parameter attribute
        dict_obj.update(
            {SeriesTemplate.ORDINATE: self.ordinate})

        # Serialize display type attribute
        dict_obj.update({SeriesTemplate.DISPLAY_TYPE: self.display_type})

        # Serialize visible attribute
        dict_obj.update({SeriesTemplate.VISIBLE: self.visible})

        # Serialize y axis attribute
        dict_obj.update({SeriesTemplate.Y_AXIS: self.y_axis})

        # Serialize custom_data attribute
        dict_obj.update({SeriesTemplate.CUSTOM_DATA: self.custom_data})

        # Serialize marker_symbol attribute
        dict_obj.update({SeriesTemplate.MARKER_SYMBOL: self.marker_symbol})

        return dict_obj

    def from_dict(self, dict_obj):
        """
        Method that initialize from dict the SeriesTemplate class

        :param dict_obj: dictionary with value to initialize instance
        :type dict_obj: dict

        :return sostrades_core.post-processing.charts.chart_filter.ChartFilter
        """
        # Deserialize name attribute
        self.series_name = dict_obj[SeriesTemplate.SERIES_NAME]

        # Deserialize abscissa parameter attribute
        self.abscissa = dict_obj[SeriesTemplate.ABSCISSA]

        # Deserialize ordinate parameter attribute
        self.ordinate = dict_obj[SeriesTemplate.ORDINATE]

        # Deserialize display type attribute
        self.display_type = dict_obj[SeriesTemplate.DISPLAY_TYPE]

        # Deserialize visible attribute
        if SeriesTemplate.VISIBLE in dict_obj:
            self.visible = dict_obj[SeriesTemplate.VISIBLE]

        # Deserialize y_axis attribute
        if SeriesTemplate.Y_AXIS in dict_obj:
            self.y_axis = dict_obj[SeriesTemplate.Y_AXIS]

        # Deserialize custom_data attribute
        if SeriesTemplate.CUSTOM_DATA in dict_obj:
            self.custom_data = dict_obj[SeriesTemplate.CUSTOM_DATA]

        # Deserialize marker_symbol attribute
        if SeriesTemplate.MARKER_SYMBOL in dict_obj:
            self.marker_symbol = dict_obj[SeriesTemplate.MARKER_SYMBOL]


class TwoAxesChartTemplate(AbstractPostProcessingPlotlyTooling):
    """ Class that define a 2 dimensional chart template 
    """

    CHART_NAME = 'chart_name'
    ABSCISSA_AXIS_NAME = 'abscissa_axis_name'
    PRIMARY_ORDINATE_AXIS_NAME = 'primary_ordinate_axis_name'
    SECONDARY_ORDINATE_AXIS_NAME = 'secondary_ordinate_axis_name'
    ABSCISSA_AXIS_RANGE = 'abscissa_axis_range'
    PRIMARY_ORDINATE_AXIS_RANGE = 'primary_ordinate_axis_range'
    SECONDARY_ORDINATE_AXIS_RANGE = 'secondary_ordinate_axis_range'
    STACKED_BAR = 'stacked_bar'
    BAR_ORIENTATION = 'bar_orientation'
    CUMULATIVE_SURFACE = 'cumulative_surface'
    SERIES = 'series'

    def __init__(self, abscissa_axis_name='', primary_ordinate_axis_name='', abscissa_axis_range=[], 
                 primary_ordinate_axis_range=[], chart_name='', stacked_bar=False, bar_orientation='v', 
                 cumulative_surface=False, secondary_ordinate_axis_name='', secondary_ordinate_axis_range=[]):
        """
         Create a new chart definition

        :param abscissa_axis_name: string that contains chart abscissa axis name
        :type abscissa_axis_name: str
        :param primary_ordinate_axis_name: string that contains chart primary ordinate axis name
        :type primary_ordinate_axis_name: str
        :param abscissa_axis_range: array(2) with min and max value range for abscissa axes
        :type abscissa_axis_range: list [min, max]
        :param primary_ordinate_axis_range: array(2) with min and max value range for primary ordinate axes
        :type primary_ordinate_axis_range: list [min, max]
        :param chart_name: string that contains chart name
        :type chart_name: str
        :param stacked_bar: boolean that make series values to be stacked
        :type stacked_bar: boolean
        :param bar_orientation: allow to set bar orientation to horizontal or vertical
        :type bar_orientation: str ('v', 'h')
        :param cumulative_surface: cumulate series values as surface
        :type cumulative_surface: boolean
        :param secondary_ordinate_axis_name: string that contains chart secondary ordinate axis name
        :type secondary_ordinate_axis_name: str
        :param secondary_ordinate_axis_range: array(2) with min and max value range for secondary ordinate axes
        :type secondary_ordinate_axis_range: list [min, max]
        """

        super().__init__()

        # Host the list of series for this chart
        self.series = []

        # Chart axis name
        self.abscissa_axis_name = abscissa_axis_name
        self.primary_ordinate_axis_name = primary_ordinate_axis_name
        self.secondary_ordinate_axis_name = secondary_ordinate_axis_name

        # Axis value range
        # two value min range (index 0) and max range (index 1)
        self.abscissa_axis_range = abscissa_axis_range
        self.primary_ordinate_axis_range = primary_ordinate_axis_range
        self.secondary_ordinate_axis_range = secondary_ordinate_axis_range

        # Chart name
        self.chart_name = chart_name

        # Bar orientation
        self.bar_orientation = bar_orientation

        # Stacked bar
        if self.bar_orientation == 'h':
            self.stacked_bar = True
        else:
            self.stacked_bar = stacked_bar

        # Initialize annotation properties
        self.annotation_upper_left = {}
        self.annotation_upper_right = {}

        # Cumulative surface display
        self.cumulative_surface = cumulative_surface

    def add_series(self, series):
        """
        Add a series instance to the current Chart instance

        :param series: series instance to add
        :type series: SeriesTemplate
        """

        if isinstance(series, SeriesTemplate):
            self.series.append(series)
        else:
            raise SeriesTemplateException(
                f'given series has the wrong type, {type(series)} instead of SeriesTemplate')

    def __repr__(self):
        """
        Overload of the class representation

        :return str: string representation of the instance
        """

        inline_series_string = '\n'.join(
            [str(series) for series in self.series])

        abs_axis_range = []
        prim_ord_axis_range = []
        sec_ord_axis_range = []
        if len(self.abscissa_axis_range) > 1:
            abs_axis_range = [self.abscissa_axis_range[0],
                              self.abscissa_axis_range[1]]
        if len(self.primary_ordinate_axis_range) > 1:
            prim_ord_axis_range = [
                self.primary_ordinate_axis_range[0],
                self.primary_ordinate_axis_range[1]]
        if len(self.secondary_ordinate_axis_range) > 1:
            sec_ord_axis_range = [
                self.secondary_ordinate_axis_range[0], self.secondary_ordinate_axis_range[1]]

        chart_string = [f'\nname: {self.chart_name}',
                        f'Abs axis name: {self.abscissa_axis_name}',
                        f'Prim ord axis name: {self.primary_ordinate_axis_name}',
                        f'Sec ord axis name: {self.secondary_ordinate_axis_name}',
                        f'Abs axis range: {abs_axis_range}',
                        f'Prim ord axis range: {prim_ord_axis_range}',
                        f'Sec ord axis range: {sec_ord_axis_range}',
                        f'annotations\n{super().__repr__()}\n',
                        f'stacked bar: {self.stacked_bar}\n',
                        f'bar orientation: {self.bar_orientation}\n',
                        f'series: {inline_series_string}\n'
                        ]

        return '\n'.join(chart_string)

    def to_dict(self):
        """
        Method that serialize as dict the SeriesTemplate class

        :return dict
        """

        dict_obj = {}
        # Serialize chart name attribute
        dict_obj.update({TwoAxesChartTemplate.CHART_NAME: self.chart_name})

        # Serialize abscissa name attribute
        dict_obj.update(
            {TwoAxesChartTemplate.ABSCISSA_AXIS_NAME: self.abscissa_axis_name})

        # Serialize ordinate name attribute
        dict_obj.update(
            {TwoAxesChartTemplate.PRIMARY_ORDINATE_AXIS_NAME: self.primary_ordinate_axis_name})
        dict_obj.update(
            {TwoAxesChartTemplate.SECONDARY_ORDINATE_AXIS_NAME: self.secondary_ordinate_axis_name})

        # Serialize abscissa range attribute
        dict_obj.update(
            {TwoAxesChartTemplate.ABSCISSA_AXIS_RANGE: self.abscissa_axis_range})

        # Serialize ordinate range attribute
        dict_obj.update(
            {TwoAxesChartTemplate.PRIMARY_ORDINATE_AXIS_RANGE: self.primary_ordinate_axis_range})
        dict_obj.update(
            {TwoAxesChartTemplate.SECONDARY_ORDINATE_AXIS_RANGE: self.secondary_ordinate_axis_range})

        # Serialize stacked bar attribute
        dict_obj.update({TwoAxesChartTemplate.STACKED_BAR: self.stacked_bar})

        # Serialize bar orientation attribute
        dict_obj.update(
            {TwoAxesChartTemplate.BAR_ORIENTATION: self.bar_orientation})

        # Serialize annotations
        dict_obj.update(super().to_dict())

        # Serialize cumulative surface attribute
        dict_obj.update(
            {TwoAxesChartTemplate.CUMULATIVE_SURFACE: self.cumulative_surface})

        # Serialize series attribute
        dict_child = []
        for series in self.series:
            dict_child.append(series.to_dict())
        dict_obj.update({TwoAxesChartTemplate.SERIES: dict_child})

        return dict_obj

    def from_dict(self, dict_obj):
        """
        Method that initialize from dict the SeriesTemplate class

        :param dict_obj: dictionary with value to initialize instance
        :type dict_obj: dict

        :return sostrades_core.post-processing.charts.chart_filter.ChartFilter
        """

        super().from_dict(dict_obj)

        # Serialize chart name attribute
        self.chart_name = dict_obj[TwoAxesChartTemplate.CHART_NAME]

        # Serialize abscissa name attribute
        self.abscissa_axis_name = dict_obj[TwoAxesChartTemplate.ABSCISSA_AXIS_NAME]

        # Serialize ordinate name attribute
        self.primary_ordinate_axis_name = dict_obj[TwoAxesChartTemplate.PRIMARY_ORDINATE_AXIS_NAME]
        self.secondary_ordinate_axis_name = dict_obj[TwoAxesChartTemplate.SECONDARY_ORDINATE_AXIS_NAME]

        # Serialize abscissa range attribute
        self.abscissa_axis_range = dict_obj[TwoAxesChartTemplate.ABSCISSA_AXIS_RANGE]

        # Serialize ordinate range attribute
        self.primary_ordinate_axis_range = dict_obj[TwoAxesChartTemplate.PRIMARY_ORDINATE_AXIS_RANGE]
        self.secondary_ordinate_axis_range = dict_obj[TwoAxesChartTemplate.SECONDARY_ORDINATE_AXIS_RANGE]

        # Serialize stacked bar attribute
        self.stacked_bar = dict_obj[TwoAxesChartTemplate.STACKED_BAR]

        # Serialize bar orientation attribute
        if TwoAxesChartTemplate.BAR_ORIENTATION in dict_obj:
            self.bar_orientation = dict_obj[TwoAxesChartTemplate.BAR_ORIENTATION]

        # Deserialize cuulative surface attribute if exist
        if TwoAxesChartTemplate.CUMULATIVE_SURFACE in dict_obj:
            self.cumulative_surface = dict_obj[TwoAxesChartTemplate.CUMULATIVE_SURFACE]

        # Serialize series attribute
        self.series = []
        for series_dict in dict_obj[TwoAxesChartTemplate.SERIES]:
            series = SeriesTemplate()
            series.from_dict(series_dict)
            self.series.append(series)
