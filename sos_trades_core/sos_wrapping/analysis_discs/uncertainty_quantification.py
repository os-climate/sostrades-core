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

from copy import deepcopy
from scipy.stats.tests.test_relative_risk import test_relative_risk_confidence_interval
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sos_trades_core.execution_engine.sos_sensitivity import SoSSensitivity
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
import numpy as np
import pandas as pd
from numpy import float32, float64
import random

from sos_trades_core.execution_engine.data_connector.ontology_data_connector import (
    OntologyDataConnector)

import openturns as ot
from openturns.viewer import View
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
import chaospy as cp

from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries
from sos_trades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import align_two_y_axes, format_currency_legend
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


class UncertaintyQuantification(SoSDiscipline):
    '''
    Generic Uncertainty Quantification class
    '''

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        # 'icon': 'fas fa-chart-line fa-fw',
        'version': '',
    }
    DESC_IN = {
        'samples_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY, 'namespace': 'ns_uncertainty_quantification', 'structuring': True},
        'data_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY, 'namespace': 'ns_uncertainty_quantification', },
    }

    DESC_OUT = {
        'input_parameters_samples_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY, 'namespace': 'ns_uncertainty_quantification'},
        'output_interpolated_values_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.LOCAL_VISIBILITY, 'namespace': 'ns_uncertainty_quantification'},
    }

    def setup_sos_disciplines(self):
        if self._data_in != {}:

            dynamic_outputs = {}
            dynamic_inputs = {}

            if ('samples_df' in self._data_in) & ('data_df' in self._data_in):
                if (self.get_sosdisc_inputs('samples_df') is not None) & (self.get_sosdisc_inputs('data_df') is not None):
                    samples_df = self.get_sosdisc_inputs(
                        'samples_df')
                    in_param = list(samples_df.columns)[1:]
                    data_df = self.get_sosdisc_inputs(
                        'data_df')
                    out_param = list(data_df.columns)[1:]

                    # ontology name
                    ontology_connector = OntologyDataConnector()
                    data_connection = {
                        'endpoint': 'https://sostradesdemo.eu.airbus.corp:31234/api/ontology'
                    }
                    args = in_param + out_param
                    args = [val.split('.')[-1]
                            for val in args]
                    ontology_connector.set_connector_request(
                        data_connection, OntologyDataConnector.PARAMETER_REQUEST, args)
                    conversion_full_ontology = ontology_connector.load_data(
                        data_connection)

                    possible_distrib = ['Normal', 'PERT']
                    distrib = [possible_distrib[random.randrange(
                        len(possible_distrib))] for i in range(len(in_param))]

                    input_distribution_default = pd.DataFrame(
                        {'parameter': in_param, 'distribution': distrib, 'lower_parameter': 80, 'upper_parameter': 100, 'most_probable_value': 95})
                    # no need most probable value for Normal distribution
                    input_distribution_default.loc[input_distribution_default['distribution']
                                                   == 'Normal', 'most_probable_value'] = np.nan

                    data_details_default = pd.DataFrame()
                    for input in in_param:
                        [name, unit] = conversion_full_ontology[input.split(
                            '.')[-1]]
                        data_details_default = data_details_default.append(
                            {'type': 'input', 'variable': input, 'name': name, 'unit': unit}, ignore_index=True)
                    for output in out_param:
                        [name, unit] = conversion_full_ontology[output.split(
                            '.')[-1]]
                        data_details_default = data_details_default.append(
                            {'type': 'output', 'variable': output, 'name': name, 'unit': unit}, ignore_index=True)

                    dynamic_inputs['input_distribution_parameters_df'] = {
                        'type': 'dataframe',
                        'dataframe_descriptor': {
                            'parameter': ('string', None, False),
                            'distribution': ('string', None, True),
                            'lower_parameter': ('float', None, True),
                            'upper_parameter': ('float', None, True),
                            'most_probable_value': ('float', None, True),
                        },
                        'unit': None,
                        'visibility': SoSDiscipline.SHARED_VISIBILITY,
                        'namespace': 'ns_uncertainty_quantification',
                        'default': input_distribution_default,
                        'structuring': True
                    }

                    dynamic_inputs['data_details_df'] = {
                        'type': 'dataframe',
                        'dataframe_descriptor': {
                            'type': ('string', None, False),
                            'variable': ('string', None, False),
                            'name': ('string', None, True),
                            'unit': ('string', None, True),
                        },
                        'unit': None,
                        'visibility': SoSDiscipline.SHARED_VISIBILITY,
                        'namespace': 'ns_uncertainty_quantification',
                        'default': data_details_default,
                        'structuring': True
                    }

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def run(self):
        inputs_dict = self.get_sosdisc_inputs()
        samples_df = inputs_dict['samples_df']
        data_df = inputs_dict['data_df']
        input_parameters_names = list(samples_df.columns)[1:]
        output_names = list(data_df.columns)[1:]
        input_distribution_parameters_df = inputs_dict['input_distribution_parameters_df']
        input_distribution_parameters_df['values'] = [sorted(
            list(samples_df[input_name].unique())) for input_name in input_parameters_names]

        # input_distribution_parameters_df['lower_parameter'] = [i+5 for i in list(input_distribution_parameters_df['lower_parameter'])]
        # input_distribution_parameters_df['upper_parameter'] = [i-5 for i in list(input_distribution_parameters_df['upper_parameter'])]

        all_data_df = samples_df.merge(data_df, on='scenario', how='left')
        all_data_df = all_data_df.sort_values(by=input_parameters_names)

        # INPUT PARAMETERS DISTRIBUTION IN [NORMAL, PERT]
        n = 10000
        input_parameters_distrib_df = pd.DataFrame()
        distrib_list = []
        for input_name in input_parameters_names:
            if input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter'] == input_name]['distribution'].values[0] == 'Normal':
                distrib = self.Normal_distrib(
                    input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter']
                                                         == input_name]['lower_parameter'].values[0],
                    input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter']
                                                         == input_name]['upper_parameter'].values[0],
                    confidence_interval=0.99
                )
            elif input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter'] == input_name]['distribution'].values[0] == 'PERT':
                distrib = self.PERT_distrib(
                    input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter']
                                                         == input_name]['lower_parameter'].values[0],
                    input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter']
                                                         == input_name]['upper_parameter'].values[0],
                    input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter']
                                                         == input_name]['most_probable_value'].values[0],
                )
            else:
                self.logger.exception(
                    'Exception occurred: possible values in distribution are [Normal, PERT].'
                )
            distrib_list.append(distrib)
            input_parameters_distrib_df[f'{input_name}'] = pd.DataFrame(
                np.array(distrib.getSample(n)))

        # MONTECARLO COMPOSED DISTRIBUTION
        R = ot.CorrelationMatrix(len(input_parameters_names))
        copula = ot.NormalCopula(R)
        distribution = ot.ComposedDistribution(distrib_list, copula)
        composed_distrib_sample = distribution.getSample(n)

        # for i in range(len(input_parameters)):
        #     graph = distribution.drawMarginal1DPDF(i, 90, 100, 256)
        #     graph.setTitle(
        #         f'{input_parameters[i]} {input_param_dict[input_parameters[i]]["distribution"]} distribution')
        #     view = View(graph, plot_kw={'color': 'blue'})

        # INTERPOLATION
        input_parameters_values_tuple = tuple([input_distribution_parameters_df.loc[input_distribution_parameters_df['parameter'] == input_name]['values'].values[0]
                                               for input_name in input_parameters_names])
        input_dim_tuple = tuple([len(sub_t)
                                 for sub_t in input_parameters_values_tuple])

        output_distrib_df = pd.DataFrame()
        for output_name in output_names:
            y = list(all_data_df[output_name])
            output_values = np.reshape(y, input_dim_tuple)
            f = RegularGridInterpolator(
                input_parameters_values_tuple, output_values, bounds_error=False)
            output_distrib = f(composed_distrib_sample)
            output_distrib_df[f'{output_name}'] = output_distrib

        dict_values = {
            'input_parameters_samples_df': input_parameters_distrib_df,
            'output_interpolated_values_df': output_distrib_df,
        }

        self.store_sos_outputs_values(dict_values)

    def Normal_distrib(self, lower_bnd, upper_bnd, confidence_interval=0.95):
        # Normal distribution
        # 90% confidence interval : ratio = 3.29
        # 95% confidence interval : ratio = 3.92
        # 99% confidence interval : ratio = 5.15
        if confidence_interval == 0.9:
            ratio = 3.29
            ratio = norm.ppf(0.95) - norm.ppf(0.05)
        if confidence_interval == 0.95:
            ratio = 3.92
        if confidence_interval == 0.99:
            ratio = 5.15

        mu = (lower_bnd + upper_bnd) / 2
        sigma = (upper_bnd - lower_bnd) / ratio
        distrib = ot.Normal(mu, sigma)

        # plot
        # graph = Normaldistribution.drawMarginal1DPDF(0, lower_bnd, upper_bnd, 256)
        # view = View(graph, plot_kw={'color': 'blue'})

        return distrib

    def PERT_distrib(self, lower_bnd, upper_bnd, most_probable_val):
        # PERT distribution (from chaopsy library cause ot doesnt have it)
        chaospy_dist = cp.PERT(lower_bnd, most_probable_val, upper_bnd)
        distrib = ot.Distribution(
            ot.ChaospyDistribution(chaospy_dist))

        # plot
        # graph = PERTdistribution.drawMarginal1DPDF(0, lower_bnd, upper_bnd, 256)
        # view = View(graph, plot_kw={'color': 'blue'})

        return distrib

    def get_chart_filter_list(self):

        # For the outputs, making a graph for tco vs year for each range and for specific
        # value of ToT with a shift of five year between then

        chart_filters = []

        chart_list = ['Output distrib']
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'Charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        # For the outputs, making a bar graph with gradients values

        instanciated_charts = []

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'Charts':
                    graphs_list = chart_filter.selected_values

        if 'Output distrib' in graphs_list:
            if 'output_interpolated_values_df' in self.get_sosdisc_outputs():
                output_distrib_df = deepcopy(
                    self.get_sosdisc_outputs('output_interpolated_values_df')
                )
            if 'input_parameters_samples_df' in self.get_sosdisc_outputs():
                input_parameters_distrib_df = deepcopy(
                    self.get_sosdisc_outputs('input_parameters_samples_df')
                )
            if 'data_details_df' in self.get_sosdisc_inputs():
                self.data_details = deepcopy(
                    self.get_sosdisc_inputs(['data_details_df']))

        for input_name in list(input_parameters_distrib_df.columns):
            input_distrib = list(input_parameters_distrib_df[input_name])
            new_chart = self.histogram_graph(
                input_distrib, input_name)
            instanciated_charts.append(new_chart)

        for output_name in list(output_distrib_df.columns):
            output_distrib = list(output_distrib_df[output_name])
            new_chart = self.histogram_graph(
                output_distrib, output_name)
            instanciated_charts.append(new_chart)

        return instanciated_charts

    def histogram_graph(self, data, title):
        name = self.data_details.loc[self.data_details["variable"]
                                     == title]["name"].values[0]
        unit = self.data_details.loc[self.data_details["variable"]
                                     == title]["unit"].values[0]
        hist_y = go.Figure()
        hist_y.add_trace(go.Histogram(x=list(data),
                                      nbinsx=100, histnorm='probability'))

        y_describe = pd.Series(list(data)).describe()
        hist_y.update_layout(xaxis=dict(
            title=name,
            ticksuffix=unit),
            yaxis=dict(title='Probability'))

        hist_y.add_shape(type='line', xref='x', yref='paper',
                         x0=y_describe.loc['mean'],
                         x1=y_describe.loc['mean'],
                         y0=0, y1=1)

        hist_y.add_shape(type='rect',  xref='x', yref='paper',
                         x0=y_describe.loc['25%'],
                         x1=y_describe.loc['75%'],
                         y0=0, y1=1,
                         line=dict(color="LightSeaGreen"),
                         fillcolor="PaleTurquoise", opacity=0.2)
        hist_y.add_trace(go.Scatter(x=[y_describe.loc['mean'], y_describe.loc['25%'], y_describe.loc['75%']],
                                    y=[0.01, 0.01, 0.01],
                                    textfont=dict(color="black", size=12),
                                    text=["Mean", "Q1", "Q3"], mode="text"))

        hist_y.update_layout(showlegend=False)

        percent_pos = len([p for p in data if p > 0]) / len(data) * 100

        text_left = {
            'Mean':  f"{format_currency_legend(y_describe.loc['mean'],unit)}",
            'Standard deviation': f"{format_currency_legend(y_describe.loc['std'],unit)}"}

        text_right = {
            'First quartile (25%)': f"{format_currency_legend(y_describe.loc['25%'],unit)}",
            'Third quartile (75%)': f"{format_currency_legend(y_describe.loc['75%'],unit)}",
            'Percentage of positive values':  f'{percent_pos:9.4f} %'}

        new_chart = InstantiatedPlotlyNativeChart(
            fig=hist_y, chart_name=f'{name} Distribution', default_legend=False)

        new_chart.annotation_upper_left = text_left
        new_chart.annotation_upper_right = text_right
        # new_chart.to_plotly().show()

        return new_chart
