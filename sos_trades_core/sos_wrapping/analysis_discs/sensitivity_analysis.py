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

from sos_trades_core.execution_engine.sos_sensitivity import SoSSensitivity
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
import numpy as np
import pandas as pd
from numpy import float32, float64

from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.execution_engine.sos_eval import SoSEval


class SensitivityAnalysis(SoSSensitivity):
    '''
    Generic Sensitivity Analysis class
    '''

    DESC_IN = {
        'variation_list': {'default': ['+/-10%'], 'type': 'string_list', 'unit': None, 'possible_values': ['+/-5%', '+/-10%', '+/-20%', '+/-50%']}
    }
    DESC_IN.update(SoSEval.DESC_IN)

    DESC_OUT = {
        'sensitivity_outputs': {'type': 'dict', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY},
    }

    def run(self):
        '''
            Overloaded SoSEval method
        '''
        sensitivity_inputs, sensitivity_outputs, variation_list = self.get_sosdisc_inputs(
            ['eval_inputs', 'eval_outputs', 'variation_list'])

        variation_list = [float(variation[3:-1])
                          for variation in variation_list]

        self.set_eval_in_out_lists(
            sensitivity_inputs, sensitivity_outputs)
        sensitivity_outputs = self.launch_sensitivity_analysis(variation_list)

        sens_dict = self.compute_df(sensitivity_outputs)

        dict_values = {'sensitivity_outputs': sens_dict}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

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

        # Retrieve eval inputs and eval outputs
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

        if year_start is not None and year_end is not None:
            year_list = np.arange(year_start, year_end + 1)
            chart_filters.append(ChartFilter(
                'Year List', list(year_list), [2050], 'year_list'))

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):

        # For the outputs, making a bar graph with gradients values

        instanciated_charts = []

        list_aircrafts = None
        year_start = None
        year_end = None

        if len(self.dm.get_all_namespaces_from_var_name('AC_list')) > 0:
            list_aircrafts = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('AC_list')[
                0])

        if len(self.dm.get_all_namespaces_from_var_name('year_start')) > 0:
            year_start = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])
            year_end = self.dm.get_value(self.dm.get_all_namespaces_from_var_name('year_start')[
                0])
            year_list = list(np.arange(year_start, year_end + 1))

        # Retrieve the results of the sensitivity analysis
        sensitivity_outputs = self.get_sosdisc_outputs('sensitivity_outputs')
        #ac_list_pano = self.dm.get_value('AC_list_panorama')
        # Add the two lists of aircrafts to deal with all aircrafts
#         if ac_list_pano is not None:
#             list_aircrafts += ac_list_pano

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
            # Get the correct output following the selected variation.
            # The selected variation is a string '+/-5%', [3:-1] to retrieve the
            # '5'
            sensitivity_relative = sensitivity_outputs[f'+{variation[3:-1]}.0%_relative']
            sensitivity_relative_neg = sensitivity_outputs[f'-{variation[3:-1]}.0%_relative']
            # Loop over the list of outputs selected in the filter
            for output_filter in outputs_list:
                output_grad = {}
                output_grad_neg = {}
                grad_type = {}
                grad_keys = {}
                key_outputs = []
                # Sort only gradients that we need considering filters
                # Loop over all gradients computed in the simulation
                for key_grad in sensitivity_relative.keys():
                    # A gradient output key is created as 'eval_output vs
                    # eval_input'
                    key_output = key_grad.split(' vs ')[0]
                    if key_output.endswith(output_filter):
                        output_grad[key_grad] = sensitivity_relative[key_grad]
                        output_grad_neg[key_grad] = sensitivity_relative_neg[key_grad]
                        # Check the type of the output (df or dict or float)
                        grad_type[key_output] = type(
                            sensitivity_relative[key_grad])
                        # If it is a dict we store all keys of the dict to plot all
                        # values
                        if type(sensitivity_relative[key_grad]) is dict:
                            grad_keys[key_output] = list(
                                sensitivity_relative[key_grad].keys())
                        key_outputs.append(key_output)
                # Grad keys contains now all outputs to plot and key_outputs all keys
                # To deal with multi instances of an output variable
                key_outputs = list(set(key_outputs))
                # Loop over all output keys
                for output in key_outputs:
                    gradient_output = {}
                    gradient_output_neg = {}
                    for grad in output_grad.keys():
                        if grad.split(' vs ')[0] == output:
                            gradient_output[grad] = output_grad[grad]
                            gradient_output_neg[grad] = output_grad_neg[grad]
                # Case for dict (aggregation of aircraft results from multiinstances
                # self or dict of float like program_infos)
                    if grad_type[output] is dict:

                        for program in grad_keys[output]:

                            key_zero = list(gradient_output.keys())[0]
                            # Case for dict of dict (ex:
                            # cashflow_program_infos_dict)
                            if type(gradient_output[key_zero][program]) is dict:
                                # Check if the A/C is in the selected A/C
                                if program not in selected_ac and 'actor' not in key_zero:
                                    continue
                                for output_data in gradient_output[key_zero][program]:
                                    # If the input list is equal to 1 the chart
                                    # name is different
                                    if len(inputs_list) == 1:
                                        chart_name = f'{program} {output_data} sensitivity to a {variation}  {inputs_list[0]} variation'
                                    else:
                                        chart_name = f'{program} {output_data} sensitivity to a {variation}  input variations'
                                    #
                                    new_chart = TwoAxesInstanciatedChart('Sensitivity (%)', '',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                    abscissa_list = []
                                    ordinate_list = []
                                    ordinate_list_minus = []
                                    # Construct all lists to plot in the chart
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            # If the key ends with the input_name
                                            # then
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                # Get the +5 and -5 % variation
                                                sensitivity_grad = gradient_output[key_grad][program][output_data]
                                                sensitivity_grad_neg = gradient_output_neg[
                                                    key_grad][program][output_data]
                                                # All zero values are not plotted
                                                # in the chart to avoid too much
                                                # information
                                                if not isinstance(sensitivity_grad, str) and (sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0):
                                                    program_name = ''
                                                    input_name_big = key_grad.split(
                                                        ' vs ')[-1]
                                                    # Check if the aircraft name in the input_name is in the list of aircrafts
                                                    # If there is only one aircraft
                                                    # in the list of needed
                                                    # aircraft, no need to specify
                                                    # the aircraft
                                                    for aircraft in list_aircrafts:
                                                        if aircraft in input_name_big and len(list_aircrafts) != 1:
                                                            program_name = aircraft
                                                            break
                                                    abscissa_list.append(
                                                        program_name + ' ' + input_name)
                                                    ordinate_list.append(
                                                        sensitivity_grad)
                                                    ordinate_list_minus.append(
                                                        sensitivity_grad_neg)
                                    # Sort all lists by the absolute value of the
                                    # ordinate list
                                    abs_list = [abs(ordinate)
                                                for ordinate in ordinate_list]
                                    abscissa_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, abscissa_list))]
                                    ordinate_list = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list))]
                                    ordinate_list_minus = [abscissa for _, abscissa in sorted(
                                        zip(abs_list, ordinate_list_minus))]
                                    # Finally instanciate the two series
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
                                if program not in selected_ac and 'actor' not in key_zero:
                                    continue
                                df = gradient_output[key_zero][program]
                                # Loop over the columns of the dataframe
                                for column_df in list(df):
                                    # Do not plot the sensitivity on years
                                    if column_df == 'year' or column_df == 'years':
                                        continue

                                    # if only one input the sensitivity is plotted over
                                    # years and not over input list
                                    if len(inputs_list) == 1:

                                        max_index = 0
                                        for input_name in inputs_list:
                                            for key_grad in gradient_output:
                                                if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                    max_sensitivity_grad = gradient_output[key_grad][program][column_df].max(
                                                    )
                                                    min_sensitivity_grad = gradient_output[key_grad][program][column_df].min(
                                                    )
                                                    max_sensitivity_grad_neg = gradient_output_neg[key_grad][program][column_df].max(
                                                    )
                                                    min_sensitivity_grad_neg = gradient_output_neg[key_grad][program][column_df].min(
                                                    )
                                                    if min_sensitivity_grad != 0.0 or max_sensitivity_grad != 0.0 or \
                                                            min_sensitivity_grad_neg != 0.0 or max_sensitivity_grad_neg != 0.0:
                                                        max_index += 1
                                        if max_index != 0:
                                            chart_name = f'{program} {column_df} sensitivity to a {variation} {inputs_list[0]} variation'
                                            new_chart = TwoAxesInstanciatedChart('Sensitivity (%)', 'years',
                                                                                 chart_name=chart_name, stacked_bar=True, bar_orientation='h')

                                            for input_name in inputs_list:
                                                for key_grad in gradient_output:
                                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                        sensitivity_grad = gradient_output[key_grad][program][column_df].values.tolist(
                                                        )
                                                        sensitivity_grad_neg = gradient_output_neg[key_grad][program][column_df].values.tolist(
                                                        )
                                                        # Check if the entire
                                                        # column is not zero
                                                        # (meaning that the
                                                        # variable is not
                                                        # sensitivie to the
                                                        # input)
                                                        if not all((np.isnan(v) or v == 0) for v in sensitivity_grad) or not all((np.isnan(v) or v == 0) for v in sensitivity_grad_neg):
                                                            if year_start is not None and year_end is not None:
                                                                new_series = InstanciatedSeries(sensitivity_grad,
                                                                                                list(np.arange(year_start, year_end + 1)),  'Df', 'bar')

                                                                new_chart.series.append(
                                                                    new_series)
                                                                new_series2 = InstanciatedSeries(sensitivity_grad_neg,
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

                                            for input_name in inputs_list:
                                                for key_grad in gradient_output:
                                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                        sensitivity_grad = gradient_output[key_grad][
                                                            program][column_df].iloc[year_index]
                                                        sensitivity_grad_neg = gradient_output_neg[key_grad][
                                                            program][column_df].iloc[year_index]
                                                        if sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0:
                                                            max_index += 1
                                            if max_index != 0:
                                                chart_name = f'{program} {column_df} sensitivity to a {variation} input variations in year {selected_year}'
                                                new_chart = TwoAxesInstanciatedChart('Sensitivity (%)', '',
                                                                                     chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                                abscissa_list = []
                                                ordinate_list = []
                                                ordinate_list_minus = []
                                                for input_name in inputs_list:
                                                    for key_grad in gradient_output:
                                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                            sensitivity_grad = gradient_output[key_grad][
                                                                program][column_df].iloc[year_index]
                                                            sensitivity_grad_neg = gradient_output_neg[key_grad][
                                                                program][column_df].iloc[year_index]
                                                            if sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0:
                                                                program_name = ''
                                                                aircraft = key_grad.split(
                                                                    ' vs ')[-1].split('.')[-2]
                                                                if aircraft in list_aircrafts and len(list_aircrafts) != 1:
                                                                    program_name = aircraft
                                                                abscissa_list.append(
                                                                    program_name + ' ' + input_name)
                                                                ordinate_list.append(
                                                                    sensitivity_grad)
                                                                ordinate_list_minus.append(
                                                                    sensitivity_grad_neg)
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
                                            sensitivity_grad_neg = gradient_output_neg[key_grad][program]
                                            # The name of the A/C is here in the
                                            # namespace of the input and not in the
                                            # key as for dict
                                            if type(sensitivity_grad) != str and (sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0):
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
                                    'Sensitivity (%)', '', chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                abscissa_list = []
                                ordinate_list = []
                                ordinate_list_minus = []
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            sensitivity_grad = gradient_output[key_grad][program]
                                            sensitivity_grad_neg = gradient_output_neg[key_grad][program]
                                            if type(sensitivity_grad) != str and (sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0):
                                                program_name = ''
                                                aircraft = key_grad.split(
                                                    ' vs ')[-1].split('.')[-2]
                                                if aircraft in list_aircrafts and len(list_aircrafts) != 1:
                                                    program_name = aircraft
                                                abscissa_list.append(
                                                    program_name + ' ' + input_name)
                                                ordinate_list.append(
                                                    sensitivity_grad)
                                                ordinate_list_minus.append(
                                                    sensitivity_grad_neg)
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
                    # Output for dataframe
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
                                aircraft = []
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            max_sensitivity_grad = gradient_output[key_grad][column_df].max(
                                            )
                                            min_sensitivity_grad = gradient_output[key_grad][column_df].min(
                                            )
                                            max_sensitivity_grad_neg = gradient_output_neg[key_grad][column_df].max(
                                            )
                                            min_sensitivity_grad_neg = gradient_output_neg[key_grad][column_df].min(
                                            )
                                            if min_sensitivity_grad != 0.0 or max_sensitivity_grad != 0.0 or \
                                                    min_sensitivity_grad_neg != 0.0 or max_sensitivity_grad_neg != 0.0:
                                                max_index += 1
                                                aircraft = key_grad.split(' vs ')[0].split(
                                                    '.')[-2]
                                if max_index != 0:
                                    chart_name = f'{aircraft} {column_df} sensitivity to a {variation} {inputs_list[0]} variation'
                                    new_chart = TwoAxesInstanciatedChart('Sensitivity (%)', 'years ',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')

                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].values.tolist(
                                                )
                                                sensitivity_grad_neg = gradient_output_neg[key_grad][column_df].values.tolist(
                                                )
                                                if not all((np.isnan(v) or v == 0) for v in sensitivity_grad) or not all((np.isnan(v) or v == 0) for v in sensitivity_grad_neg):
                                                    if year_start is not None and year_end is not None:
                                                        new_series = InstanciatedSeries(sensitivity_grad,
                                                                                        list(np.arange(year_start, year_end + 1)), 'Df', 'bar')

                                                        new_chart.series.append(
                                                            new_series)
                                                        new_series2 = InstanciatedSeries(sensitivity_grad_neg,
                                                                                         list(np.arange(year_start, year_end + 1)), '-Df', 'bar')

                                                        new_chart.series.append(
                                                            new_series2)
                                    instanciated_charts.append(new_chart)
                            else:

                                max_index = 0
                                aircraft = ''
                                for selected_year in selected_years:
                                    year_index = year_list.index(
                                        selected_year)
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].iloc[year_index]
                                                sensitivity_grad_neg = gradient_output_neg[
                                                    key_grad][column_df].iloc[year_index]

                                                if sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0:
                                                    max_index += 1
                                                    aircraft = key_grad.split(' vs ')[0].split(
                                                        '.')[-2]
                                    #
                                    if aircraft == '':
                                        continue
                                    chart_name = f'{aircraft} {column_df} sensitivity to a {variation} input variation in year {selected_year}'
                                    new_chart = TwoAxesInstanciatedChart('Sensitivity (%)', '',
                                                                         chart_name=chart_name, stacked_bar=True, bar_orientation='h')
                                    abscissa_list = []
                                    ordinate_list = []
                                    ordinate_list_minus = []
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][column_df].iloc[year_index]
                                                sensitivity_grad_neg = gradient_output_neg[
                                                    key_grad][column_df].iloc[year_index]
                                                if sensitivity_grad != 0.0 or sensitivity_grad_neg != 0.0:
                                                    program_name = ''
                                                    aircraft = key_grad.split(
                                                        ' vs ')[-1].split('.')[-2]
                                                    if aircraft in list_aircrafts and len(list_aircrafts) != 1:
                                                        program_name = aircraft
                                                    abscissa_list.append(
                                                        program_name + input_name)
                                                    ordinate_list.append(
                                                        sensitivity_grad)
                                                    ordinate_list_minus.append(
                                                        sensitivity_grad_neg)
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
