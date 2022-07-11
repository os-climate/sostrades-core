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
model that store filter for chart
"""


class ChartFilter:
    """ Class that define a chart filter
    """
    FILTER_NAME = 'filter_name'
    FILTER_VALUES = 'filter_values'
    SELECTED_VALUES = 'selected_values'
    FILTER_KEY = 'filter_key'
    MULTIPLE_SELECTION = 'multiple_selection'

    def __init__(self, name='', filter_values=[], selected_values=[], filter_key=None, multiple_selection=True):
        """ Create a filter use to filter post processing building

        @param name : string that contains table name
        @type str

        @param filter_values : list of filter items that can be used to filter post processing element
        @type list of string

        @param selected_values : list of filter items currently selected for the given filter 
        (one or several depending of the 'multiple_selection argument)
        @type list of string

        @param filter_key : unique key used to identify the current filter
        @type str

        @param multiple_selection : unique key used to identify the current filter
        @type str
        """

        self.name = name

        # Pie chart datas
        if not isinstance(filter_values, list):
            message = f'"filter_values" argument is intended to be a list not {type(filter_values)}'
            raise TypeError(message)
        self.filter_values = filter_values

        self.selected_values = selected_values

        self.filter_key = filter_key or ''
        self.multiple_selection = multiple_selection

    def __repr__(self):
        """ Overload of the class representation

        @return str, string representation of the instance
        """

        series_string = [f'\nname: {self.name}',
                         f'values: {self.filter_values}',
                         f'selected values: {self.selected_values}',
                         f'filter key: {self.filter_key}',
                         f'multiple selection: {self.multiple_selection}'
                         ]

        return '\n'.join(series_string)

    def to_dict(self):
        """ Method that serialize as dict the SeriesTemplate class

        @return dict
        """

        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({ChartFilter.FILTER_NAME: self.name})

        # Serialize values parameter attribute
        dict_obj.update(
            {ChartFilter.FILTER_VALUES: self.filter_values})

        # Serialize selected values parameter attribute
        dict_obj.update(
            {ChartFilter.SELECTED_VALUES: self.selected_values})

        # Serialize filter key parameter attribute
        dict_obj.update(
            {ChartFilter.FILTER_KEY: self.filter_key})

        # Serialize multiple selection parameter attribute
        dict_obj.update(
            {ChartFilter.MULTIPLE_SELECTION: self.multiple_selection})

        return dict_obj

    @staticmethod
    def from_dict(dict_obj):
        """ Method that initialize from dict the SeriesTemplate class

        @param dictçobj: dictionary with value to initialize instance
        @type dict

        @return ssostrades_corepost-processing.charts.chart_filter.ChartFilter
        """
        result = ChartFilter()

        # Serialize name attribute
        result.name = dict_obj[ChartFilter.FILTER_NAME]

        # Serialize values parameter attribute
        result.filter_values = dict_obj[ChartFilter.FILTER_VALUES]

        # Serialize selected values parameter attribute
        result.selected_values = dict_obj[ChartFilter.SELECTED_VALUES]

        # Serialize filter_key parameter attribute
        result.filter_key = dict_obj[ChartFilter.FILTER_KEY]

        # Serialize multiple selection parameter attribute
        result.multiple_selection = dict_obj[ChartFilter.MULTIPLE_SELECTION]

        return result
