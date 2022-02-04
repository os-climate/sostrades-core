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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

import pandas as pd
import numpy as np
from numpy import float32, float64

from sos_trades_core.execution_engine.sos_gradients import SoSGradients
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.execution_engine.sos_eval import SoSEval


class FORMAnalysis(SoSGradients):
    '''
    First Order Reliability Method Analysis based on the gradient computation
    '''

    DESC_IN = {
        'grad_method': {'type': 'string', 'unit': None, 'possible_values': ['Complex Step', '1st order FD', '2nd order FD']},
        'variation_list': {'default': ['+/-10%'], 'type': 'string_list', 'unit': None, 'possible_values': ['+/-5%', '+/-10%', '+/-20%', '+/-50%']},
    }
    DESC_IN.update(SoSEval.DESC_IN)

    DESC_OUT = {
        'FORM_outputs': {'type': 'dict', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY},
    }

    def run(self):
        '''
            Overloaded SoSEval method
        '''
        eval_inputs, eval_outputs, grad_method = self.get_sosdisc_inputs(
            ['eval_inputs', 'eval_outputs', 'grad_method'])
        self.set_eval_in_out_lists(eval_inputs, eval_outputs)
        gradient_outputs = self.launch_gradient_analysis(grad_method)

        variation_list = self.get_sosdisc_inputs('variation_list')

        variation_list = [float(variation[3:-1])
                          for variation in variation_list]
        form_outputs = self.compute_form_outputs(
            gradient_outputs, variation_list)
        dict_values = {'FORM_outputs': form_outputs}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values, update_dm=True)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        ac_list = None
        year_start = None
        year_end = None

        if len(self.dm.get_all_namespaces_from_var_name('AC_list')) > 0:
            ac_list = self.dm.get_value(
                self.dm.get_all_namespaces_from_var_name('AC_list')[0])

        if len(self.dm.get_all_namespaces_from_var_name('year_start')) > 0:
            year_start = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])
            year_end = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])

        if year_start is not None and year_end is not None:
            year_list = np.arange(year_start, year_end + 1)
            chart_filters.append(ChartFilter(
                'Year List', list(year_list), [2050], 'year_list'))

        # Retrieve the tco_df that host ToT values
        grad_inputs, grad_outputs, variation_list = self.get_sosdisc_inputs(
            ['eval_inputs', 'eval_outputs', 'variation_list'])

        chart_filters.append(ChartFilter(
            'Inputs variables', grad_inputs, grad_inputs, 'inputs'))

        chart_filters.append(ChartFilter(
            'Outputs variables', grad_outputs, grad_outputs, 'outputs'))

        chart_filters.append(ChartFilter(
            'Variation List (%)', variation_list, variation_list, 'variation_list'))

        if ac_list is not None:
            chart_filters.append(ChartFilter(
                'Aircraft List', ac_list, ac_list, 'ac_list'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a bar graph with gradients values

        instanciated_charts = []

        year_start = None
        year_end = None

        if len(self.dm.get_all_namespaces_from_var_name('year_start')) > 0:
            year_start = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])
            year_end = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])
            year_list = list(np.arange(year_start, year_end + 1))

        # Retrieve the tco_df that host ToT values
        FORM_outputs = self.get_sosdisc_outputs('FORM_outputs')
        inputs_list = []
        outputs_list = []
        selected_ac = []
        selected_variation = []
        selected_years = []
        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'inputs':
                    inputs_list = chart_filter.selected_values
                if chart_filter.filter_key == 'outputs':
                    outputs_list = chart_filter.selected_values
                if chart_filter.filter_key == 'ac_list':
                    selected_ac = chart_filter.selected_values
                if chart_filter.filter_key == 'variation_list':
                    selected_variation = chart_filter.selected_values
                if chart_filter.filter_key == 'year_list':
                    selected_years = chart_filter.selected_values

        for variation in selected_variation:

            FORM_relative = FORM_outputs[f'{variation[3:-1]}.0%_relative']
            for output_filter in outputs_list:
                output_grad = {}
                grad_type = {}
                grad_keys = {}
                key_outputs = []
                # Sort only gradients that we need considering filters
                for key_grad in FORM_relative.keys():
                    key_output = key_grad.split(' vs ')[0]
                    if key_output.endswith(output_filter):
                        output_grad[key_grad] = FORM_relative[key_grad]
                        grad_type[key_output] = type(FORM_relative[key_grad])
                        if type(FORM_relative[key_grad]) is dict:
                            grad_keys[key_output] = list(
                                FORM_relative[key_grad].keys())
                        key_outputs.append(key_output)

                # To deal with multi instances of an output variable
                key_outputs = list(set(key_outputs))
                for output in key_outputs:
                    gradient_output = {}
                    for grad in output_grad.keys():
                        if grad.split(' vs ')[0] == output:
                            gradient_output[grad] = output_grad[grad]
                # Case for dict (aggregation of aircraft results from multiinstances
                # self or dict of float like program_infos)
                    if grad_type[output] is dict:

                        for program in grad_keys[output]:

                            key_zero = list(gradient_output.keys())[0]
                            # For dict of dict (ex:
                            # cashflow_program_infos_dict)
                            if type(gradient_output[key_zero][program]) is dict:
                                if program not in selected_ac:
                                    continue
                                for output_data in gradient_output[key_zero][program]:
                                    if len(inputs_list) == 1:
                                        chart_name = f'{program} {output_data} sensitivity to a {variation} {inputs_list[0]} variation'
                                    else:
                                        chart_name = f'{program} {output_data} sensitivity to a {variation} input variations'

                                    max_index = 0
                                    min_value = 0.0
                                    max_value = 0.0
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][program][output_data]
                                                max_value = max(
                                                    max_value, sensitivity_grad)
                                                min_value = min(
                                                    min_value, sensitivity_grad)
                                                if sensitivity_grad != 0.0:
                                                    max_index += 1
                                    #
                                    new_chart = TwoAxesInstanciatedChart('FORM sensitivity (%)', '',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                    abscissa_list = []
                                    ordinate_list = []
                                    ordinate_list_minus = []
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][program][output_data]
                                                if sensitivity_grad != 0.0:
                                                    abscissa_list.append(
                                                        input_name)
                                                    ordinate_list.append(
                                                        sensitivity_grad)
                                                    ordinate_list_minus.append(
                                                        -sensitivity_grad)
                                    abs_list = [abs(ordinate)
                                                for ordinate in ordinate_list]
                                    abscissa_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, abscissa_list))]
                                    ordinate_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list))]
                                    ordinate_list_minus = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list_minus))]
                                    new_series = InstanciatedSeries(
                                        ordinate_list, abscissa_list, 'Df', 'bar')
                                    new_series2 = InstanciatedSeries(
                                        ordinate_list_minus, abscissa_list, '-Df', 'bar')
                                    new_chart.series.append(
                                        new_series)
                                    new_chart.series.append(
                                        new_series2)
                                    instanciated_charts.append(new_chart)
                            # For dict of dataframe (ex: cashflow_program_dict)
                            elif type(gradient_output[key_zero][program]) is pd.DataFrame:
                                if program not in selected_ac:
                                    continue
                                df = gradient_output[key_zero][program]
                                for column_df in list(df):
                                    if column_df == 'year' or column_df == 'years':
                                        continue

                                    # if only one input the sensitivity is plotted over
                                    # years and not over input list
                                    if len(inputs_list) == 1:

                                        max_index = 0
                                        min_value = 0.0
                                        max_value = 0.0
                                        for input_name in inputs_list:
                                            for key_grad in gradient_output:
                                                if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                    max_sensitivity_grad = gradient_output[key_grad][program][column_df].max(
                                                    )
                                                    max_value = max(
                                                        max_value, max_sensitivity_grad)
                                                    min_sensitivity_grad = gradient_output[key_grad][program][column_df].min(
                                                    )
                                                    min_value = min(
                                                        min_value, min_sensitivity_grad)
                                                    if min_sensitivity_grad != 0.0 or max_sensitivity_grad != 0.0:
                                                        max_index += 1
                                        if max_index != 0:
                                            chart_name = f'{program} {column_df} sensitivity to a {variation} {inputs_list[0]} variation'
                                            new_chart = TwoAxesInstanciatedChart('FORM sensitivity (%)', 'years',
                                                                                 chart_name=chart_name, stacked_bar=True, bar_orientation='h')

                                            for input_name in inputs_list:
                                                for key_grad in gradient_output:
                                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                        sensitivity_grad = gradient_output[key_grad][program][column_df].values.tolist(
                                                        )
                                                        minus_sens_grad = [
                                                            -grad for grad in sensitivity_grad]
                                                        if not all(v == 0 for v in sensitivity_grad):
                                                            if year_start is not None and year_end is not None:
                                                                new_series = InstanciatedSeries(sensitivity_grad,
                                                                                                list(np.arange(year_start, year_end + 1)),  input_name, 'bar')

                                                                new_chart.series.append(
                                                                    new_series)
                                                                new_series2 = InstanciatedSeries(minus_sens_grad,
                                                                                                 list(np.arange(year_start, year_end + 1)), '-Df', 'bar')

                                                                new_chart.series.append(
                                                                    new_series2)
                                            instanciated_charts.append(
                                                new_chart)
                                    else:

                                        for selected_year in selected_years:
                                            year_index = year_list.index(
                                                selected_year)
                                            max_index = 0
                                            min_value = 0.0
                                            max_value = 0.0

                                            for input_name in inputs_list:
                                                for key_grad in gradient_output:
                                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                        sensitivity_grad = gradient_output[key_grad][
                                                            program][column_df].iloc[year_index]
                                                        max_value = max(
                                                            max_value, sensitivity_grad)
                                                        min_value = min(
                                                            min_value, sensitivity_grad)
                                                        if sensitivity_grad != 0.0:
                                                            max_index += 1
                                            if max_index != 0:
                                                chart_name = f'{program} {column_df} sensitivity to a {variation} input variations in year {selected_year}'
                                                new_chart = TwoAxesInstanciatedChart('FORM sensitivity (%)', '',
                                                                                     chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                                abscissa_list = []
                                                ordinate_list = []
                                                ordinate_list_minus = []
                                                for input_name in inputs_list:
                                                    for key_grad in gradient_output:
                                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                            sensitivity_grad = gradient_output[key_grad][
                                                                program][column_df].iloc[year_index]
                                                            if sensitivity_grad != 0.0:
                                                                abscissa_list.append(
                                                                    input_name)
                                                                ordinate_list.append(
                                                                    sensitivity_grad)
                                                                ordinate_list_minus.append(
                                                                    -sensitivity_grad)
                                                abs_list = [abs(ordinate)
                                                            for ordinate in ordinate_list]
                                                abscissa_list = [abscissa for _, abscissa in sorted(
                                                    zip(abs_list, abscissa_list))]
                                                ordinate_list = [abscissa for _, abscissa in sorted(
                                                    zip(abs_list, ordinate_list))]
                                                ordinate_list_minus = [abscissa for _, abscissa in sorted(
                                                    zip(abs_list, ordinate_list_minus))]
                                                new_series = InstanciatedSeries(
                                                    ordinate_list, abscissa_list, 'Df', 'bar')
                                                new_series2 = InstanciatedSeries(
                                                    ordinate_list_minus, abscissa_list, '-Df', 'bar')
                                                new_chart.series.append(
                                                    new_series)
                                                new_chart.series.append(
                                                    new_series2)
                                                instanciated_charts.append(
                                                    new_chart)
                            # For dict of float (ex: cashflow_program_infos)
                            elif type(gradient_output[key_zero][program]) in [float, float64, float32, int]:
                                aircraft = ''
                                max_index = 0
                                input_only = 'inputs variables'
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            sensitivity_grad = gradient_output[key_grad][program]
                                            if sensitivity_grad != 0.0:
                                                max_index += 1
                                                aircraft = key_grad.split(
                                                    ' vs ')[0].split('.')[-2]
                                                input_only = key_grad.split(
                                                    ' vs ')[-1].split('.')[-1]
                                if aircraft not in selected_ac:
                                    continue
                                if max_index == 1:
                                    chart_name = f'{aircraft} {program} sensitivity to a {variation} {input_only} variation'
                                else:
                                    chart_name = f'{aircraft} {program} sensitivity to a {variation} input variations'

                                new_chart = TwoAxesInstanciatedChart(
                                    'FORM sensitivity (%)', '', chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                abscissa_list = []
                                ordinate_list = []
                                ordinate_list_minus = []
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            sensitivity_grad = gradient_output[key_grad][program]
                                            if sensitivity_grad != 0.0:
                                                abscissa_list.append(
                                                    input_name)
                                                ordinate_list.append(
                                                    sensitivity_grad)
                                                ordinate_list_minus.append(
                                                    -sensitivity_grad)
                                abs_list = [abs(ordinate)
                                            for ordinate in ordinate_list]
                                abscissa_list = [abscissa for _, abscissa in sorted(
                                    zip(abs_list, abscissa_list))]
                                ordinate_list = [abscissa for _, abscissa in sorted(
                                    zip(abs_list, ordinate_list))]
                                ordinate_list_minus = [abscissa for _, abscissa in sorted(
                                    zip(abs_list, ordinate_list_minus))]
                                new_series = InstanciatedSeries(
                                    ordinate_list, abscissa_list, 'Df', 'bar')
                                new_series2 = InstanciatedSeries(
                                    ordinate_list_minus, abscissa_list, '-Df', 'bar')
                                new_chart.series.append(
                                    new_series)
                                new_chart.series.append(
                                    new_series2)
                                instanciated_charts.append(new_chart)

                    elif grad_type[output] is pd.DataFrame:

                        key_zero = list(gradient_output.keys())[0]
                        df = gradient_output[key_zero]

                        for column_df in list(df):
                            if column_df == 'year' or column_df == 'years':
                                continue

                            # if only one input the sensitivity is plotted over
                            # years and not over input list
                            if len(inputs_list) == 1:

                                max_index = 0
                                min_value = 0.0
                                max_value = 0.0
                                aircraft = []
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            max_sensitivity_grad = gradient_output[key_grad][column_df].max(
                                            )
                                            max_value = max(
                                                max_value, max_sensitivity_grad)
                                            min_sensitivity_grad = gradient_output[key_grad][column_df].min(
                                            )
                                            min_value = min(
                                                min_value, min_sensitivity_grad)
                                            if min_sensitivity_grad != 0.0 or max_sensitivity_grad != 0.0:
                                                max_index += 1
                                                aircraft = key_grad.split(' vs ')[0].split(
                                                    '.')[-2]
                                if max_index != 0:
                                    chart_name = f'{aircraft} {column_df} sensitivity to a {variation} {inputs_list[0]} variation'
                                    new_chart = TwoAxesInstanciatedChart('FORM sensitivity (%)', 'years ',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')

                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].values.tolist(
                                                )
                                                minus_sens_grad = [
                                                    -grad for grad in sensitivity_grad]
                                                if not all(v == 0 for v in sensitivity_grad):
                                                    if year_start is not None and year_end is not None:
                                                        new_series = InstanciatedSeries(sensitivity_grad,
                                                                                        list(np.arange(year_start, year_end + 1)), 'Df', 'bar')

                                                        new_chart.series.append(
                                                            new_series)
                                                        new_series2 = InstanciatedSeries(minus_sens_grad,
                                                                                         list(np.arange(year_start, year_end + 1)), '-Df', 'bar')

                                                        new_chart.series.append(
                                                            new_series2)
                                    instanciated_charts.append(new_chart)
                            else:

                                max_index = 0
                                min_value = 0.0
                                max_value = 0.0
                                aircraft = ''
                                for selected_year in selected_years:
                                    year_index = year_list.index(
                                        selected_year)
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].iloc[year_index]
                                                max_value = max(
                                                    max_value, sensitivity_grad)
                                                min_value = min(
                                                    min_value, sensitivity_grad)
                                                if sensitivity_grad != 0.0:
                                                    max_index += 1
                                                    aircraft = key_grad.split(' vs ')[0].split(
                                                        '.')[-2]
                                    #
                                    if aircraft == '':
                                        continue
                                    chart_name = f'{aircraft} {column_df} sensitivity to a {variation} input variation in year {selected_year}'
                                    new_chart = TwoAxesInstanciatedChart('FORM sensitivity (%)', '',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                    abscissa_list = []
                                    ordinate_list = []
                                    ordinate_list_minus = []
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].iloc[year_index]
                                                if sensitivity_grad != 0.0:
                                                    abscissa_list.append(
                                                        input_name)
                                                    ordinate_list.append(
                                                        sensitivity_grad)
                                                    ordinate_list_minus.append(
                                                        -sensitivity_grad)
                                    abs_list = [abs(ordinate)
                                                for ordinate in ordinate_list]
                                    abscissa_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, abscissa_list))]
                                    ordinate_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list))]
                                    ordinate_list_minus = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list_minus))]
                                    new_series = InstanciatedSeries(
                                        ordinate_list, abscissa_list, 'Df', 'bar')
                                    new_series2 = InstanciatedSeries(
                                        ordinate_list_minus, abscissa_list, '-Df', 'bar')
                                    new_chart.series.append(
                                        new_series)
                                    new_chart.series.append(
                                        new_series2)
                                    instanciated_charts.append(new_chart)

        return instanciated_charts
