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
from gemseo.algos.doe.doe_factory import DOEFactory
from numpy import array

from sos_trades_core.api import get_sos_logger
from sos_trades_core.sos_wrapping.analysis_discs.doe_eval import DoeEval
import itertools
import copy
import numpy as np


import itertools
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart,\
    InstanciatedSeries
from sos_trades_core.tools.post_processing.tables.instanciated_table import InstanciatedTable
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import align_two_y_axes, format_currency_legend
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import \
    InstantiatedPlotlyNativeChart


class GridSearchEval(DoeEval):
    '''
    Generic Grid Search evaluation class
    '''

    INPUT_TYPE = ['float']
    EVAL_INPUTS = 'eval_inputs'
    EVAL_OUTPUTS = 'eval_outputs'
    NB_POINTS = 'nb_points'
    DESC_IN = {
        EVAL_INPUTS: {'type': 'dataframe',
                      'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                               'full_name': ('string', None, False)},
                      'dataframe_edition_locked': False,
                      'structuring': True},
        EVAL_OUTPUTS: {'type': 'dataframe',
                       'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                'full_name': ('string', None, False)},
                       'dataframe_edition_locked': False,
                       'structuring': True}
    }

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create the design space only
        """

        dynamic_inputs = {}
        dynamic_outputs = {}

        if (self.EVAL_INPUTS in self._data_in) & (self.EVAL_OUTPUTS in self._data_in):

            eval_outputs = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)

            # we fetch the inputs and outputs selected by the user
            selected_outputs = eval_outputs[eval_outputs['selected_output']
                                            == True]['full_name']
            selected_inputs = eval_inputs[eval_inputs['selected_input']
                                          == True]['full_name']

            # select inputs till  maximum selected input number
            # selected_inputs_full = [self.conversion_short_full[val]
            #                         for val in list(selected_inputs) if val in self.conversion_short_full.keys()]
            # selected_outputs_full = [self.conversion_short_full[val]
            # for val in list(selected_outputs) if val in
            # self.conversion_short_full.keys()]

            # self.selected_inputs = selected_inputs_full[
            #     : self.max_inputs_nb]
            # self.selected_outputs = selected_outputs_full
            self.selected_inputs = selected_inputs[
                : self.max_inputs_nb]
            self.selected_outputs = selected_outputs

            self.set_eval_in_out_lists(
                self.selected_inputs, self.selected_outputs)

            # grid8seqrch can be done only for selected inputs and outputs
            if (len(self.eval_in_list) > 0):
                # setting dynamic outputs. One output of type dict per selected
                # output
                if (len(self.eval_out_list) > 0):
                    for out_var in self.eval_out_list:
                        dynamic_outputs.update(
                            {f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
                                                                                    'namespace': 'ns_doe'}})

                # setting dynamic design space with default value if not
                # specified
                default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                     self.LOWER_BOUND: 0.0,
                                                     self.UPPER_BOUND: 100,
                                                     self.NB_POINTS: 2
                                                     })
                dynamic_inputs.update(
                    {'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
                                      }})

                if ('design_space' in self._data_in):
                    design_space = self.get_sosdisc_inputs(self.DESIGN_SPACE)
                    if (set(design_space['variable'].to_list()) != set(selected_inputs)):
                        default_design_space = pd.DataFrame({self.VARIABLES: self.selected_inputs,
                                                             self.LOWER_BOUND: 0.0,
                                                             self.UPPER_BOUND: 100,
                                                             self.NB_POINTS: 2
                                                             })
                        self._data_in['design_space']['value'] = default_design_space

                # algo_options to match with doe and specify processes nb
                default_dict = {'n_processes': 1,
                                'wait_time_between_samples': 0.0}
                dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
                                                        'dataframe_edition_locked': False,
                                                        'dataframe_descriptor': {
                                                            self.VARIABLES: ('string', None, False),
                                                            self.VALUES: ('string', None, True)},
                                                        'user_level': 99,
                                                        'editable': False}})

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        ee.ns_manager.add_ns('ns_doe', ee.study_name)
        super(GridSearchEval, self).__init__(sos_name, ee, cls_builder)
        self.doe_factory = DOEFactory()
        self.logger = get_sos_logger(f'{self.ee.logger.name}.GridSearch')
        self.eval_input_types = ['float', 'int', 'string']
        self.eval_in_list = []
        self.eval_out_list = []
        self.max_inputs_nb = 3
        self.conversion_full_short = {}

    def generate_shortest_name(self, var_list):
        list_shortest_name = [[] for i in range(len(var_list))]
        if len(var_list) > 0:
            list_shortest_name[0].append(var_list[0].split('.')[-1])
            for a, b in itertools.combinations(var_list, 2):
                a_split = a.split('.')
                b_split = b.split('.')
                var = ''
                while (a_split[-1] == b_split[-1]):
                    var = '.' + a_split[-1] + var
                    del a_split[-1]
                    del b_split[-1]
                a_shortest = a_split[-1] + var
                b_shortest = b_split[-1] + var

                list_shortest_name[var_list.index(a)].append(a_shortest)
                list_shortest_name[var_list.index(b)].append(b_shortest)

            list_shortest_name = [max(item, key=len)
                                  for item in list_shortest_name]

        self.conversion_full_short.update({
            key: value for key, value in zip(var_list, list_shortest_name)})
        return list_shortest_name

    def generate_samples_from_doe_factory(self):
        """
        Generating samples for the GridSearch with algo fullfact using the Doe Factory
        """
        algo_name = 'fullfact'
        ds = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        options = {'levels': ds['nb_points'].to_list()}

        self.design_space = self.create_design_space()

        filled_options = {}
        for algo_option in options:
            if options[algo_option] != 'default':
                filled_options[algo_option] = options[algo_option]

        if self.N_SAMPLES not in options:
            self.logger.warning("N_samples is not defined; pay attention you use fullfact algo "
                                "and that levels are well defined")

        self.logger.info(filled_options)
        filled_options[self.DIMENSION] = self.design_space.dimension
        filled_options[self._VARIABLES_NAMES] = self.design_space.variables_names
        filled_options[self._VARIABLES_SIZES] = self.design_space.variables_sizes

        algo = self.doe_factory.create(algo_name)
        self.samples = algo._generate_samples(**filled_options)

        unnormalize_vect = self.design_space.unnormalize_vect
        round_vect = self.design_space.round_vect
        samples = []
        for sample in self.samples:
            x_sample = round_vect(unnormalize_vect(sample))
            self.design_space.check_membership(x_sample)
            samples.append(x_sample)
        self.samples = samples

        return self.prepare_samples()

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # the eval process to analyse is stored as the only child of SoSEval
        # (coupling chain of the eval process or single discipline)
        analyzed_disc = self.sos_disciplines[0]

        possible_in_values_full, possible_out_values_full = self.fill_possible_values(
            analyzed_disc)

        possible_in_values_full, possible_out_values_full = self.find_possible_values(
            analyzed_disc, possible_in_values_full, possible_out_values_full)

        # Take only unique values in the list
        possible_in_values_full = list(set(possible_in_values_full))
        possible_out_values_full = list(set(possible_out_values_full))

        # Fill the possible_values of eval_inputs
        possible_in_values_full.sort()
        possible_out_values_full.sort()

        # shortest name
        self.generate_shortest_name(
            list(set(possible_in_values_full)))
        self.generate_shortest_name(
            list(set(possible_out_values_full)))

        possible_in_values_short = [
            self.conversion_full_short[val] for val in possible_in_values_full]
        possible_out_values_short = [
            self.conversion_full_short[val] for val in possible_out_values_full]

        # possible_in_values_short = list(set(possible_in_values_short))
        # possible_out_values_short = list(set(possible_out_values_short))

        # possible_in_values_short.sort()
        # possible_out_values_short.sort()

        # default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values_full_short],
        #                                      'full_name': possible_in_values_full_short})
        # default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values_full_short],
        #                                       'full_name': possible_out_values_full_short})

        default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values_full],
                                             'full_name': possible_in_values_full,
                                             'shortest_name': possible_in_values_short})
        default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values_full],
                                              'full_name': possible_out_values_full,
                                              'shortest_name': possible_out_values_short})

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')

        # if eval input not set
        if eval_input_new_dm is None:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)

        # if eval input set for only certain var
        elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
            default_dataframe = copy.deepcopy(default_in_dataframe)
            if sum(eval_input_new_dm['selected_input'].to_list()) > self.max_inputs_nb:
                self.logger.warning(
                    "You have selected more than 3 inputs. Only the 3 first inputs will be considered.")
                already_set_names = eval_input_new_dm['full_name'].tolist()[
                    :self.max_inputs_nb]
                already_set_values = eval_input_new_dm['selected_input'].tolist()[
                    :self.max_inputs_nb]
            else:
                already_set_names = eval_input_new_dm['full_name'].tolist()
                already_set_values = eval_input_new_dm['selected_input'].tolist(
                )
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
                    index]
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_dataframe, check_value=False)
        # if eval input set for True value number_var>max_number_var
        elif sum(eval_input_new_dm['selected_input'].to_list()) > self.max_inputs_nb:
            self.logger.warning(
                "You have selected more than 3 inputs. Only the 3 first inputs will be considered.")
            default_dataframe = copy.deepcopy(default_in_dataframe)
            eval_input_new_dm_true = eval_input_new_dm.loc[eval_input_new_dm['selected_input'] == True]
            already_set_names = eval_input_new_dm_true['full_name'].tolist()[
                :self.max_inputs_nb]
            already_set_values = eval_input_new_dm_true['selected_input'].tolist()[
                :self.max_inputs_nb]
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
                    index]
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_dataframe, check_value=False)

        if eval_output_new_dm is None:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)

    def prepare_chart_dict(self, outputs_discipline_dict):

        # outputs_discipline_dict=self.outputs_discipline_dict
        doe_samples_df = outputs_discipline_dict['doe_samples_dataframe']
        inputs = [col for col in doe_samples_df.columns if col not in ['scenario']]

        inputs_combin = list(itertools.combinations(inputs, 2))

        full_chart_list = []
        # for key in outputs_discipline_dict.keys():
        #     if key != 'doe_samples_dataframe':
        #         cf_dict=outputs_discipline_dict[key]
        #         outs_cols_vble=cf_dict.select_dtypes(include='float').columns.to_list()
        #         full_chart_list+=list(itertools.product(inputs_combin,outs_cols_vble))
        cf_dict = outputs_discipline_dict[list(
            outputs_discipline_dict.keys())[1]]

        if isinstance(cf_dict, dict):
            outs_cols_vble = cf_dict[list(cf_dict.keys())[0]].select_dtypes(
                include='float').columns.to_list()
            # outs_cols_vble = cf_dict.select_dtypes(include='float').columns.to_list()
            full_chart_list += list(itertools.product(inputs_combin,
                                                      outs_cols_vble))

        chart_dict = {}
        for chart in full_chart_list:
            z_vble = chart[1]
            x_vble = chart[0][0]
            y_vble = chart[0][1]

            slide_vble = []
            for col in inputs:
                if col not in chart[0]:
                    slide_vble.append(col)

            chart_name = f'{z_vble} based on {x_vble} vs {y_vble}'
            output_name = list(outputs_discipline_dict.keys())[
                1].split('.')[-1][:-5]
            chart_dict[chart_name] = {
                'x': x_vble,
                'x_unit': self.ee.dm.get_data(self.ee.dm.get_all_namespaces_from_var_name(x_vble)[0])['unit'],
                'y': y_vble,
                'y_unit': self.ee.dm.get_data(self.ee.dm.get_all_namespaces_from_var_name(y_vble)[0])['unit'],
                'z': z_vble,
                'z_unit': self.ee.dm.get_data(self.ee.dm.get_all_namespaces_from_var_name(output_name)[0])['unit'],
                'slider': slide_vble
            }

        # 'graph1':{
        #     'x':{'RC'},
        #     'y':'NRC',
        #      z:'NPV',
        #     sliders:['COC']}

        return chart_dict

    def get_chart_filter_list(self):

        chart_filters = []

        outputs_dict = self.get_sosdisc_outputs()
        chart_dict = self.prepare_chart_dict(outputs_dict)

        chart_list = list(chart_dict.keys())
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'Charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        outputs_dict = self.get_sosdisc_outputs()
        chart_dict = self.prepare_chart_dict(outputs_dict)

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'Charts':
                    graphs_list = chart_filter.selected_values

        doe_samples_df = outputs_dict['doe_samples_dataframe']
        cf_dict = outputs_dict[list(outputs_dict.keys())[1]]
        # if 'scenario' not in cf_dict.columns:
        #     cf_dict.rename(columns={'variable': 'scenario'}, inplace=True)
        output_df = None

        for scenario, df in cf_dict.items():
            filtered_df = df.copy(deep=True)
            filtered_df['scenario'] = f'{scenario}'

            if output_df is None:
                output_df = filtered_df.copy(deep=True)
            else:
                output_df = pd.concat(
                    [output_df, filtered_df], axis=0)

        cont_plot_df = doe_samples_df.merge(
            output_df, how="left", on='scenario')

        for name, chart_info in chart_dict.items():
            fig = go.Figure()
            if name in graphs_list:
                if len(chart_info['slider']) == 0:

                    x_data = cont_plot_df[chart_info['x']].to_list()
                    y_data = cont_plot_df[chart_info['y']].to_list()
                    z_data = cont_plot_df[chart_info['z']].to_list()

                    fig.add_trace(
                        go.Contour(
                            x=x_data,
                            y=y_data,
                            z=z_data,
                            colorscale='YlGnBu',
                            contours=dict(
                                coloring='heatmap',
                                showlabels=True,  # show labels on contours
                                labelfont=dict(  # label font properties
                                    size=10,
                                    # color = 'white',
                                )
                            ),
                            colorbar=dict(
                                title=f'{chart_info["z"]} ({chart_info["z_unit"]})',
                                nticks=10,
                                # ticks='outside',
                                ticklen=5, tickwidth=1,
                                # showticklabels=True,
                                tickangle=0, tickfont_size=10),
                            visible=True,
                        )
                    )

                    fig.update_layout(
                        autosize=True,
                        xaxis=dict(
                            title=chart_info['x'],
                            ticksuffix=chart_info["x_unit"],
                            titlefont_size=12,
                            tickfont_size=10,
                            automargin=True
                        ),
                        yaxis=dict(
                            title=chart_info['y'],
                            ticksuffix=chart_info["y_unit"],
                            titlefont_size=12,
                            tickfont_size=10,
                            # ticksuffix='$',
                            # tickformat=',.0%',
                            automargin=True,
                        ),
                        # margin=dict(l=0.25, b=100)
                    )

                    if len(fig.data) > 0:
                        chart_name = f'<b> {name} </b>'
                        new_chart = InstantiatedPlotlyNativeChart(
                            fig=fig, chart_name=chart_name, default_legend=False)
                        instanciated_charts.append(new_chart)
                        # new_chart.to_plotly().show()

                if len(chart_info['slider']) == 1:
                    col_slider = chart_info['slider'][0]
                    slider_values = cont_plot_df[col_slider].unique()
                    fig = go.Figure()
                    for slide_value in slider_values:
                        x_data = cont_plot_df.loc[cont_plot_df[col_slider]
                                                  == slide_value][chart_info['x']].to_list()
                        y_data = cont_plot_df.loc[cont_plot_df[col_slider]
                                                  == slide_value][chart_info['y']].to_list()
                        z_data = cont_plot_df.loc[cont_plot_df[col_slider]
                                                  == slide_value][chart_info['z']].to_list()

                        # Initialization Slider
                        if slide_value == slider_values[-1]:
                            visible = True
                        else:
                            visible = False

                        fig.add_trace(
                            go.Contour(
                                x=x_data,
                                y=y_data,
                                z=z_data,
                                colorscale='YlGnBu',
                                contours=dict(
                                    coloring='heatmap',
                                    showlabels=True,  # show labels on contours
                                    labelfont=dict(  # label font properties
                                        size=10,
                                        # color = 'white',
                                    )
                                ),
                                colorbar=dict(
                                    title=f'{chart_info["z"]} ({chart_info["z_unit"]})',
                                    nticks=10,
                                    ticks='outside',
                                    ticklen=5, tickwidth=1,
                                    # showticklabels=True,
                                    tickangle=0, tickfont_size=10),
                                visible=visible,
                            )
                        )

                    # Create and add slider
                    steps = []

                    for i in range(int(len(fig.data))):

                        step = dict(
                            method="update",
                            args=[{"visible": [False] * len(fig.data)},
                                  {"title": f'<b>{name} for : {col_slider} = {slider_values[i]} % </b>'}],
                            label=f'{slider_values[i]}'  # layout attribute
                        )
                        # Toggle i'th trace to 'visible'
                        step["args"][0]["visible"][i] = True
                        steps.append(step)

                    sliders = [dict(
                        active=len(steps) - 1,
                        currentvalue={'visible': True,
                                      "prefix": f' {col_slider}:  '},
                        steps=steps,
                        pad=dict(t=50),
                    )]

                    fig.update_layout(
                        sliders=sliders,
                        autosize=True,

                        xaxis=dict(
                            title=f'{chart_info["x"]}',
                            ticksuffix=chart_info["x_unit"],
                            titlefont_size=12,
                            tickfont_size=10,
                            automargin=True
                        ),
                        yaxis=dict(
                            title=f'{chart_info["y"]}',
                            titlefont_size=12,
                            tickfont_size=10,
                            ticksuffix=chart_info["y_unit"],
                            # tickformat=',.0%',
                            automargin=True,
                        ),
                        # margin=dict(l=0.25, b=100)
                    )
                    # Create native plotly chart
                    last_value = slider_values[-1]
                    if len(fig.data) > 0:
                        chart_name = f'<b>{name} for : {chart_info["slider"][0]} = {last_value} % </b>'
                        new_chart = InstantiatedPlotlyNativeChart(
                            fig=fig, chart_name=chart_name, default_legend=False)
                        instanciated_charts.append(new_chart)

                    # new_chart.to_plotly().show()
        return instanciated_charts
