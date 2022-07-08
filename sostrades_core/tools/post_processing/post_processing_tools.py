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
Bundle of tools related to chart creation
"""
import math
import numpy as np


def format_currency_legend(currency_value, currency_symbol='€'):
    """ Given an numeric value of currency, return the value with the correct legend to display it in short format
        (euros, kilo euros, etc...) rounded at two digits
        ex : 8765432 € => 8.76 m€

    :params: currency_value, numeric currency values
    :type: float

    :params: currency_symbol, symbol of the currency (default = '€')
    :type: string

    :returns: string
    """

    value = currency_value
    legend_letter = ''
    if isinstance(currency_value, str):
        return currency_value
    else:
        if abs(currency_value) >= 1.0e9:
            value = currency_value / 1.0e9
            legend_letter = 'B'
        elif 1.0e9 > abs(currency_value) >= 1.0e6:
            value = currency_value / 1.0e6
            legend_letter = 'M'
        elif 1.0e6 > abs(currency_value) >= 1.0e3:
            value = currency_value / 1.0e3
            legend_letter = 'k'

        return f'{round(value, 2)} {legend_letter}{currency_symbol}'


# inspired from https:#github.com/VictorBezak/Plotly_Multi-Axes_Gridlines
def align_two_y_axes(figure, GRIDLINES=4):
    y1Values = []
    y2Values = []
    if len(figure.data) > 1:
        for trace in figure.data:
            if trace.yaxis == 'y':
                y1Values = [*y1Values, *trace.y]
            elif trace.yaxis == 'y2':
                y2Values = [*y2Values, *trace.y]

        if any([x != 0 for x in y1Values]) and any([x != 0 for x in y2Values]):
            ranges, dticks = calculate_alignment_for_two_y_axes(
                y1Values, y2Values, GRIDLINES)
            figure.update_layout(
                yaxis=dict(
                    range=ranges['y1'],
                    dtick=dticks['y1']),
                yaxis2=dict(
                    range=ranges['y2'],
                    dtick=dticks['y2']))
    return figure


# inspired from https:#github.com/VictorBezak/Plotly_Multi-Axes_Gridlines
def calculate_alignment_for_two_y_axes(y1Values, y2Values, GRIDLINES=4):
    # ************************************************************************
    # Y1 Calculations

    y1_min = min(y1Values)
    y1_max = max(y1Values)

    if (y1_min < 0):
        y1_range = y1_max - y1_min
    else:
        y1_range = y1_max

    y1_range = y1_range * 1000  # mult by 1000 to account for ranges < 1
    y1_len = len(str(math.floor(y1_range)))  # find number fo digits

    # find 10^x where x == num of digits in y_max - 1
    y1_pow10_divisor = math.pow(10, y1_len - 1)
    # find leading digit of y_max and multiply
    y1_firstdigit = math.floor(y1_range / y1_pow10_divisor)
    y1_max_base = y1_pow10_divisor * y1_firstdigit / \
        1000  # div by 1000 to account for ranges < 1

    y1_dtick = y1_max_base / GRIDLINES

    # y1_pow10_divisor = math.pow(10, y1_len - 1) / \
    #     1000  # reset for logging purposes
    # y1_range = y1_range / 1000  # range reset

    # ************************************************************************
    # Y2 Calculations

    y2_min = min(y2Values)
    y2_max = max(y2Values)

    if (y2_min < 0):
        y2_range = y2_max - y2_min
    else:
        y2_range = y2_max

    y2_range = y2_range * 1000  # mult by 1000 to account for ranges < 1
    y2_len = len(str(math.floor(y2_range)))

    y2_pow10_divisor = math.pow(10, y2_len - 1)
    y2_firstdigit = math.floor(y2_range / y2_pow10_divisor)
    y2_max_base = y2_pow10_divisor * y2_firstdigit / \
        1000  # div by 1000 to account for ranges < 1

    y2_dtick = y2_max_base / GRIDLINES

    # y2_pow10_divisor = math.pow(10, y2_len - 1) / \
    #     1000  # reset for logging purposes
    # y2_range = y2_range / 1000  # range reset

    '''**************************************************************************'''
    # Capture the highest dtick ratio as your global dtick ratio.
    #
    # All other axes will have their positive and negative ranges scaled to
    # make their dtick_ratios match the global ratio. When the ratios match,
    # the gridlines match!
    '''**************************************************************************'''

    y1_dtick_ratio = y1_range / y1_dtick
    y2_dtick_ratio = y2_range / y2_dtick

    global_dtick_ratio = max(y1_dtick_ratio, y2_dtick_ratio)

    '''**************************************************************************'''
    # Calculate Range Minimums
    #
    # 1. This is done by first finding the negative ratio for all axes:
    #     1. what percentage of the range is coming from negative values
    #     2. multiply percentage by global ratio to get the percentage of the
    #        global ratio (percentage of total gridlines) that should be shown
    #        under the zero baseline.
    #
    #     NEGATIVE RATIO == NUMBER OF GRIDLINES NEEDED FOR NEGATIVE VALUES
    #
    # 2. Capturing the highest negative ratio as the global negative ratio
    #
    # 3. Then applying the negative ratio to all of your axis minimumsto get
    #    their new proportionally scaled range minimums
    '''**************************************************************************'''

    negative = False  # Are there any negative values present

    if (y1_min < 0):
        negative = True
        y1_negative_ratio = abs(y1_min / y1_range) * global_dtick_ratio
    else:
        y1_negative_ratio = 0

    if (y2_min < 0):
        negative = True
        y2_negative_ratio = abs(y2_min / y2_range) * global_dtick_ratio
    else:
        y2_negative_ratio = 0

    # Increase the ratio by 0.1 so that your range minimums are extended just
    # far enough to not cut off any part of your lowest value
    global_negative_ratio = max(y1_negative_ratio, y2_negative_ratio) + 0.1

    # If any negative value is present, you must proportionally extend the
    # range minimum of all axes
    if (negative):
        y1_range_min = (global_negative_ratio) * y1_dtick * -1
        y2_range_min = (global_negative_ratio) * y2_dtick * -1
    else:  # If no negatives, baseline is set to zero
        y1_range_min = 0
        y2_range_min = 0

    # ************************************************************************
    # Calculate Range Maximums
    #
    # 1. This is done by first finding the positive ratio for all axes:
    #     1. what percentage of the range is coming from positive values
    #     2. multiply percentage by global ratio to get the percentage of the
    #        global ratio (percentage of total gridlines) that should be shown
    #        above the zero baseline.
    #
    #     POSITIVE RATIO == NUMBER OF GRIDLINES NEEDED FOR POSITIVE VALUES
    #
    # 2. Capturing the highest positive ratio as the global positive ratio
    #
    # 3. Then applying the positive ratio to all of your axis maximums to get
    #    their new proportionally scaled range maximums
    '''**************************************************************************'''

    y1_positive_ratio = abs(y1_max / y1_range) * global_dtick_ratio
    y2_positive_ratio = abs(y2_max / y2_range) * global_dtick_ratio

    # Increase the ratio by 0.1 so that your range maximums are extended just
    # far enough to not cut off any part of your highest value
    global_positive_ratio = max(y1_positive_ratio, y2_positive_ratio) + 0.1

    y1_range_max = (global_positive_ratio) * y1_dtick
    y2_range_max = (global_positive_ratio) * y2_dtick

    # ************************************************************************
    # Results
    ranges = {'y1': [y1_range_min, y1_range_max],
              'y2': [y2_range_min, y2_range_max]}
    dticks = {'y1': y1_dtick, 'y2': y2_dtick}
    return ranges, dticks


def escape_str_with_comma(str_to_escape):

    str_escaped = ''
    if str_to_escape is not None:
        if ',' in str_to_escape:
            str_escaped = f'"{str_to_escape}"'
        else:
            str_escaped = str_to_escape

    return str_escaped


def convert_nan(values):
    """ Check if value list contains NaN values
    A warning is displayed is any NaN value is found into list and then converted to None

    @param values : list of number to check
    @type list of number

    @return (list with converted values, nan_found)
    """

    result = []
    has_nan = False
    for value in values:

        if check_isnan(value):
            has_nan = True

            # Convert Nan/Infinity to None (to ensure json decoding)
            result.append(None)
        else:
            result.append(value)

    return result, has_nan


def check_isnan(value):
    """ Method to tried to identify Not A Number numpy value

    @return boolean
    """
    import numbers

    result = False
    if isinstance(value, numbers.Number) and (np.isnan(value).any() or np.isinf(value).any()):
        result = True

    return result
