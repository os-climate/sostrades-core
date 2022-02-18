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


class GradientAnalysis(SoSGradients):
    '''
    Generic Gradient Analysis class
    '''

    DESC_IN = {
        'grad_method': {'type': 'string', 'unit': None, 'possible_values': ['Complex Step', '1st order FD', '2nd order FD']}
    }
    DESC_IN.update(SoSEval.DESC_IN)

    DESC_OUT = {
        'gradient_outputs': {'type': 'dict', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY},
    }

    def eval_run(self):
        '''
            Overloaded SoSEval method
        '''
        grad_inputs, grad_outputs, grad_method = self.get_sosdisc_inputs(
            ['eval_inputs', 'eval_outputs', 'grad_method'])
        self.set_eval_in_out_lists(grad_inputs, grad_outputs)
        gradient_outputs = self.launch_gradient_analysis(grad_method)

        dict_values = {'gradient_outputs': gradient_outputs}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        ac_list = None

        if len(self.dm.get_all_namespaces_from_var_name('AC_list')) > 0:
            ac_list = self.dm.get_value(
                self.dm.get_all_namespaces_from_var_name('AC_list')[0])

        # Retrieve the tco_df that host ToT values
        grad_inputs, grad_outputs = self.get_sosdisc_inputs(
            ['eval_inputs', 'eval_outputs'])

        chart_filters.append(ChartFilter(
            'Inputs variables', grad_inputs, grad_inputs, 'inputs'))

        chart_filters.append(ChartFilter(
            'Outputs variables', grad_outputs, grad_outputs, 'outputs'))

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
        # Retrieve the tco_df that host ToT values
        gradient_outputs = self.get_sosdisc_outputs('gradient_outputs')

        inputs_list = []
        outputs_list = []
        selected_ac = []

        # Overload default value with chart filter
        if chart_filters is not None:
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'inputs':
                    inputs_list = chart_filter.selected_values
                if chart_filter.filter_key == 'outputs':
                    outputs_list = chart_filter.selected_values
                if chart_filter.filter_key == 'ac_list':
                    selected_ac = chart_filter.selected_values

        serie_index = 0

        for output_filter in outputs_list:
            output_grad = {}
            grad_type = {}
            grad_keys = {}
            key_outputs = []
            # Sort only gradients that we need considering filters
            for key_grad in gradient_outputs.keys():
                key_output = key_grad.split(' vs ')[0]
                if key_output.endswith(output_filter):
                    output_grad[key_grad] = gradient_outputs[key_grad]
                    grad_type[key_output] = type(gradient_outputs[key_grad])
                    if type(gradient_outputs[key_grad]) is dict:
                        grad_keys[key_output] = list(
                            gradient_outputs[key_grad].keys())
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
                        # For dict of dict (ex: cashflow_program_infos_dict)
                        if type(gradient_output[key_zero][program]) is dict:
                            if program not in selected_ac:
                                continue
                            for output_data in gradient_output[key_zero][program]:
                                if len(inputs_list) == 1:
                                    chart_name = f'{program} {output_data} sensitivity to {inputs_list[0]}'
                                else:
                                    chart_name = f'{program} {output_data} sensitivity to inputs variables'

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
                                new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                     [-1, max_index], [min_value, max_value], chart_name=chart_name)
                                serie_index = 0
                                for input_name in inputs_list:
                                    for key_grad in gradient_output:
                                        if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                            sensitivity_grad = gradient_output[key_grad][program][output_data]
                                            if sensitivity_grad != 0.0:
                                                new_series = InstanciatedSeries(
                                                    [serie_index], [sensitivity_grad], input_name, 'bar')
                                                serie_index += 1
                                                new_chart.series.append(
                                                    new_series)
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
                                    chart_name = f'{program} {column_df} sensitivity to {inputs_list[0]}'
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
                                    #
                                    new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                         [year_start, year_end], [min_value, max_value], chart_name=chart_name)
                                    serie_index = 0
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][program][column_df].values.tolist(
                                                )

                                                if not all(v == 0 for v in sensitivity_grad):
                                                    if year_start is not None and year_end is not None:
                                                        new_series = InstanciatedSeries(
                                                            list(np.arange(year_start, year_end + 1)), sensitivity_grad, input_name, 'bar')
                                                        serie_index += 1
                                                        new_chart.series.append(
                                                            new_series)
                                    instanciated_charts.append(new_chart)
                                else:
                                    chart_name = f'{program} {column_df} sensitivity to inputs variables'

                                    max_index = 0
                                    min_value = 0.0
                                    max_value = 0.0
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][program][column_df].mean(
                                                )
                                                max_value = max(
                                                    max_value, sensitivity_grad)
                                                min_value = min(
                                                    min_value, sensitivity_grad)
                                                if sensitivity_grad != 0.0:
                                                    max_index += 1
                                    #
                                    new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                         [-1, max_index], [min_value, max_value], chart_name=chart_name)
                                    serie_index = 0
                                    for input_name in inputs_list:
                                        for key_grad in gradient_output:
                                            if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                                sensitivity_grad = gradient_output[key_grad][program][column_df].mean(
                                                )
                                                if sensitivity_grad != 0.0:
                                                    new_series = InstanciatedSeries(
                                                        [serie_index], [sensitivity_grad], input_name, 'bar')
                                                    serie_index += 1
                                                    new_chart.series.append(
                                                        new_series)
                                    instanciated_charts.append(new_chart)
                        # For dict of float (ex: cashflow_program_infos)
                        elif type(gradient_output[key_zero][program]) in [float, float64, float32, int]:
                            max_index = 0
                            min_value = 0.0
                            max_value = 0.0
                            aircraft = ''
                            input_only = 'inputs variables'
                            for input_name in inputs_list:
                                for key_grad in gradient_output:
                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                        sensitivity_grad = gradient_output[key_grad][program]
                                        max_value = max(
                                            max_value, sensitivity_grad)
                                        min_value = min(
                                            min_value, sensitivity_grad)
                                        if sensitivity_grad != 0.0:
                                            max_index += 1
                                            aircraft = key_grad.split(
                                                ' vs ')[0].split('.')[-2]
                                            input_only = key_grad.split(
                                                ' vs ')[-1].split('.')[-1]
                            if aircraft not in selected_ac:
                                continue
                            if max_index == 1:
                                chart_name = f'{aircraft} {program} sensitivity to {input_only}'
                            else:
                                chart_name = f'{aircraft} {program} sensitivity to inputs variables'

                            new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                 [-1, max_index], [min_value, max_value], chart_name=chart_name)
                            serie_index = 0
                            for input_name in inputs_list:
                                for key_grad in gradient_output:
                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                        sensitivity_grad = gradient_output[key_grad][program]
                                        if sensitivity_grad != 0.0:
                                            new_series = InstanciatedSeries(
                                                [serie_index], [sensitivity_grad], input_name, 'bar')
                                            serie_index += 1
                                            new_chart.series.append(new_series)
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
                                            aircraft = key_grad.split('.')[-3]
                            #
                            chart_name = f'{aircraft} {column_df} sensitivity to {inputs_list[0]}'
                            new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                 [year_start, year_end], [min_value, max_value], chart_name=chart_name)
                            serie_index = 0
                            for input_name in inputs_list:
                                for key_grad in gradient_output:
                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                        sensitivity_grad = gradient_output[key_grad][column_df].values.tolist(
                                        )

                                        if not all(v == 0 for v in sensitivity_grad):
                                            if year_start is not None and year_end is not None:
                                                new_series = InstanciatedSeries(
                                                    list(np.arange(year_start, year_end + 1)), sensitivity_grad, input_name, 'bar')
                                                serie_index += 1
                                                new_chart.series.append(
                                                    new_series)
                            instanciated_charts.append(new_chart)
                        else:

                            max_index = 0
                            min_value = 0.0
                            max_value = 0.0
                            aircraft = ''
                            for input_name in inputs_list:
                                for key_grad in gradient_output:
                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                        sensitivity_grad = gradient_output[key_grad][column_df].mean(
                                        )
                                        max_value = max(
                                            max_value, sensitivity_grad)
                                        min_value = min(
                                            min_value, sensitivity_grad)
                                        if sensitivity_grad != 0.0:
                                            max_index += 1
                                            aircraft = key_grad.split('.')[-3]
                            #
                            chart_name = f'{aircraft} {column_df} mean sensitivity to inputs variables'
                            new_chart = TwoAxesInstanciatedChart('inputs_variables ', 'sensitivity gradient',
                                                                 [-1, max_index], [min_value, max_value], chart_name=chart_name)
                            serie_index = 0
                            for input_name in inputs_list:
                                for key_grad in gradient_output:
                                    if key_grad.split(' vs ')[-1].endswith('.' + input_name):
                                        sensitivity_grad = gradient_output[key_grad][column_df].mean(
                                        )
                                        if sensitivity_grad != 0.0:
                                            new_series = InstanciatedSeries(
                                                [serie_index], [sensitivity_grad], input_name, 'bar')
                                            serie_index += 1
                                            new_chart.series.append(
                                                new_series)
                            instanciated_charts.append(new_chart)

        return instanciated_charts
