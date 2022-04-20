from __future__ import annotations
from weakref import ref
import pandas as pd
from gemseo.algos.doe.doe_factory import DOEFactory
from numpy import array

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.sos_wrapping.analysis_discs.doe_eval import DoeEval
import itertools
import copy
import numpy as np
import re


import itertools
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    TwoAxesInstanciatedChart,
    InstanciatedSeries,
)
from sos_trades_core.tools.post_processing.tables.instanciated_table import (
    InstanciatedTable,
)
import plotly.graph_objects as go
from sos_trades_core.tools.post_processing.post_processing_tools import (
    align_two_y_axes,
    format_currency_legend,
)
from sos_trades_core.tools.post_processing.plotly_native_charts.instantiated_plotly_native_chart import (
    InstantiatedPlotlyNativeChart,
)
from plotly.validators.scatter.marker import SymbolValidator

from sos_trades_core.execution_engine.data_connector.ontology_data_connector import (
    OntologyDataConnector)

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


class GridSearchEval(DoeEval):
    """
    Generic Grid Search evaluation class
    """

    # ontology information
    _ontology_data = {
        'label': 'Core Grid Search Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    INPUT_TYPE = ['float']
    EVAL_INPUTS = 'eval_inputs'
    EVAL_OUTPUTS = 'eval_outputs'
    NB_POINTS = 'nb_points'
    DESC_IN = {
        EVAL_INPUTS: {
            'type': 'dataframe',
            'dataframe_descriptor': {
                'selected_input': ('bool', None, True),
                'full_name': ('string', None, False),
                'shortest_name': ('string', None, False),
                # 'ontology_name': ('string', None, False),je
            },
            'dataframe_edition_locked': False,
            'structuring': True, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_grid_search'
        },
        EVAL_OUTPUTS: {
            'type': 'dataframe',
            'dataframe_descriptor': {
                'selected_output': ('bool', None, True),
                'full_name': ('string', None, False),
                'shortest_name': ('string', None, False),
                # 'ontology_name': ('string', None, False),
            },
            'dataframe_edition_locked': False,
            'structuring': True, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_grid_search'
        },
        'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
        'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},
        'scenario_name': {'type': 'string', 'user_level': 99, 'optional': True, 'visibility': 'Local','default':"grid_search"},
    }
    DESC_OUT = {'samples_outputs_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                                       'namespace': 'ns_grid_search'}}
    DESC_OUT.update(DoeEval.DESC_OUT)

    def setup_sos_disciplines(self):
        """
        Overload setup_sos_disciplines to create the design space only
        """

        dynamic_inputs = {}
        dynamic_outputs = {}
        selected_inputs_has_changed = False

        if (self.EVAL_INPUTS in self._data_in) & (self.EVAL_OUTPUTS in self._data_in):

            eval_outputs = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)

            # we fetch the inputs and outputs selected by the user
            selected_outputs = eval_outputs[eval_outputs['selected_output'] == True][
                'full_name'
            ]
            selected_inputs = eval_inputs[eval_inputs['selected_input'] == True][
                'full_name'
            ]

            if set(selected_inputs.tolist()) != set(self.selected_inputs):
                selected_inputs_has_changed = True
                # self.selected_inputs = selected_inputs.tolist()
                # self.dm.set_data(
                #     f'{self.get_disc_full_name()}.eval_inputs',
                #     'value',
                #     eval_inputs.sort_values(
                #         by=['selected_input', 'full_name'], ascending=False).reset_index(drop=True),
                #     check_value=False,
                # )
            if set(selected_outputs.tolist()) != set(self.selected_outputs):
                self.dm.set_data(
                    f'{self.get_disc_full_name()}.eval_outputs',
                    'value',
                    eval_outputs.sort_values(
                        by=['selected_output', 'full_name'], ascending=False).reset_index(drop=True),
                    check_value=False,
                )
            self.selected_outputs = selected_outputs.tolist()

            self.selected_inputs = selected_inputs.tolist()
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_inputs',
                'value',
                eval_inputs.sort_values(
                    by=['selected_input', 'full_name'], ascending=False).reset_index(drop=True),
                check_value=False,
            )

            selected_inputs_short = eval_inputs[eval_inputs['selected_input'] == True][
                'shortest_name'
            ]

            self.selected_inputs = self.selected_inputs[: self.max_inputs_nb]
            selected_inputs_short = selected_inputs_short[: self.max_inputs_nb]
            self.set_eval_in_out_lists(
                self.selected_inputs, self.selected_outputs)

            # grid_search can be done only for selected inputs and outputs
            if len(self.eval_in_list) > 0:
                # setting dynamic outputs. One output of type dict per selected
                # output
                if len(self.eval_out_list) > 0:
                    for out_var in self.eval_out_list:
                        dynamic_outputs.update(
                            {
                                f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {
                                    'type': 'dict',
                                    'visibility': 'Shared',
                                    'namespace': 'ns_doe',
                                }
                            }
                        )

                # setting dynamic design space with default value if not
                # specified
                default_design_space = pd.DataFrame(
                    {
                        'shortest_name': selected_inputs_short.tolist(),
                        # self.VARIABLES:
                        # self.selected_inputs,
                        self.LOWER_BOUND: 0.0,
                        self.UPPER_BOUND: 100.0,
                        self.NB_POINTS: 2,
                        'full_name': self.selected_inputs,
                    }
                )
                dynamic_inputs.update(
                    {
                        'design_space': {
                            'type': 'dataframe',
                            self.DEFAULT: default_design_space,
                            'dataframe_descriptor': {
                                'shortest_name': ('string', None, False),
                                self.LOWER_BOUND: ('float', None, True),
                                self.UPPER_BOUND: ('float', None, True),
                                self.NB_POINTS: ('int', None, True),
                                'full_name': ('string', None, False),
                            },
                        }
                    }
                )

                if 'design_space' in self._data_in and selected_inputs_has_changed:
                    self._data_in['design_space']['value'] = default_design_space

                # algo_options to match with doe and specify processes nb
                default_dict = {'n_processes': 1,
                                'wait_time_between_samples': 0.0}
                dynamic_inputs.update(
                    {
                        'algo_options': {
                            'type': 'dict',
                            self.DEFAULT: default_dict,
                            'dataframe_edition_locked': False,
                            'dataframe_descriptor': {
                                self.VARIABLES: ('string', None, False),
                                self.VALUES: ('string', None, True),
                            },
                            'user_level': 99,
                            'editable': False,
                        }
                    }
                )

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def __init__(self, sos_name, ee, cls_builder):
        """
        Constructor
        """
        ee.ns_manager.add_ns('ns_doe', ee.study_name)
        super(GridSearchEval, self).__init__(sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.GridSearch')
        self.eval_input_types = ['float', 'int', 'string']
        self.max_inputs_nb = 3
        self.conversion_full_short = {}
        self.chart_dict = {}

    def generate_shortest_name(self, var_list):
        list_shortest_name = [[] for i in range(len(var_list))]
        if len(var_list) > 0:
            list_shortest_name[0].append(var_list[0].split('.')[-1])
            for a, b in itertools.combinations(var_list, 2):
                a_split = a.split('.')
                b_split = b.split('.')
                var = ''
                while a_split[-1] == b_split[-1]:
                    var = '.' + a_split[-1] + var
                    del a_split[-1]
                    del b_split[-1]
                a_shortest = a_split[-1] + var
                b_shortest = b_split[-1] + var

                list_shortest_name[var_list.index(a)].append(a_shortest)
                list_shortest_name[var_list.index(b)].append(b_shortest)

            list_shortest_name = [max(item, key=len)
                                  for item in list_shortest_name]

        self.conversion_full_short.update(
            {key: value for key, value in zip(var_list, list_shortest_name)}
        )
        return list_shortest_name

    def run(self):
        DoeEval.run(self)
        outputs_dict = self.get_sosdisc_outputs()
        inputs_dict = self.get_sosdisc_inputs()

        if 'samples_inputs_df' in outputs_dict:
            if outputs_dict['samples_inputs_df'] is not None:
                self.chart_dict, output_df = self.prepare_chart_dict( outputs_dict, inputs_dict)
                self.store_sos_outputs_values(
                    {'samples_outputs_df': output_df})

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
            self.logger.warning(
                "N_samples is not defined; pay attention you use fullfact algo "
                "and that levels are well defined"
            )

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
        """
        Once all disciplines have been run through,
        set the possible values for eval_inputs and eval_outputs in the DM
        """
        # the eval process to analyse is stored as the only child of SoSEval
        # (coupling chain of the eval process or single discipline)
        analyzed_disc = self.sos_disciplines[0]

        possible_in_values_full, possible_out_values_full = self.fill_possible_values(
            analyzed_disc
        )

        possible_in_values_full, possible_out_values_full = self.find_possible_values(
            analyzed_disc, possible_in_values_full, possible_out_values_full
        )

        # Take only unique values in the list
        possible_in_values_full = list(set(possible_in_values_full))
        possible_out_values_full = list(set(possible_out_values_full))

        # Fill the possible_values of eval_inputs
        possible_in_values_full.sort()
        possible_out_values_full.sort()

        # shortest name
        self.generate_shortest_name(list(set(possible_in_values_full)))
        self.generate_shortest_name(list(set(possible_out_values_full)))

        # possible_in_values_short = [
        #     self.conversion_full_short[val] for val in possible_in_values_full
        # ]
        # possible_out_values_short = [
        #     self.conversion_full_short[val] for val in possible_out_values_full
        # ]

        # ontology name
        ontology_connector = OntologyDataConnector()
        data_connection = {
            'endpoint': 'https://sostradesdemo.eu.airbus.corp:31234/api/ontology'
        }
        inputs_val_list = [val.split('.')[-1]
                           for val in possible_in_values_full]
        outputs_val_list = [val.split('.')[-1]
                            for val in possible_out_values_full]
        args = inputs_val_list + outputs_val_list
        ontology_connector.set_connector_request(
            data_connection, OntologyDataConnector.PARAMETER_REQUEST, args)
        conversion_full_ontology = ontology_connector.load_data(
            data_connection)

        possible_in_values_short = [
            f'{conversion_full_ontology[val.split(".")[-1]][0]} {"-".join(self.conversion_full_short[val].split(".")[:-1])}'
            for val in possible_in_values_full]
        # [conversion_full_ontology[val] for val in inputs_val_list]
        possible_out_values_short = [
            f'{conversion_full_ontology[val.split(".")[-1]][0]} {"-".join(self.conversion_full_short[val].split(".")[:-1])}'
            for val in possible_out_values_full]
        # [conversion_full_ontology[val] for val in outputs_val_list]

        default_in_dataframe = pd.DataFrame(
            {
                'selected_input': [False for invar in possible_in_values_full],
                'shortest_name': possible_in_values_short,
                'full_name': possible_in_values_full,
            }
        )
        default_out_dataframe = pd.DataFrame(
            {
                'selected_output': [False for invar in possible_out_values_full],
                'shortest_name': possible_out_values_short,
                'full_name': possible_out_values_full,
            }
        )

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')

        # if eval input not set
        if eval_input_new_dm is None:
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_inputs',
                'value',
                default_in_dataframe,
                check_value=False,
            )

        # if eval input set for only certain var
        elif set(eval_input_new_dm['full_name'].tolist()) != (
                set(default_in_dataframe['full_name'].tolist())
        ):
            default_dataframe = copy.deepcopy(default_in_dataframe)
            if sum(eval_input_new_dm['selected_input'].to_list()) > self.max_inputs_nb:
                self.logger.warning(
                    "You have selected more than 3 inputs. Only the 3 first inputs will be considered."
                )
                already_set_names = eval_input_new_dm['full_name'].tolist()[
                                    : self.max_inputs_nb
                                    ]
                already_set_values = eval_input_new_dm['selected_input'].tolist()[
                                     : self.max_inputs_nb
                                     ]
            else:
                already_set_names = eval_input_new_dm['full_name'].tolist()
                already_set_values = eval_input_new_dm['selected_input'].tolist(
                )
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[
                    default_dataframe['full_name'] == name, 'selected_input'
                ] = already_set_values[index]
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_inputs',
                'value',
                default_dataframe,
                check_value=False,
            )
        # if eval input set for True value number_var>max_number_var
        elif sum(eval_input_new_dm['selected_input'].to_list()) > self.max_inputs_nb:
            self.logger.warning(
                "You have selected more than 3 inputs. Only the 3 first inputs will be considered."
            )
            default_dataframe = copy.deepcopy(default_in_dataframe)
            eval_input_new_dm_true = eval_input_new_dm.loc[
                eval_input_new_dm['selected_input'] == True
                ]
            already_set_names = eval_input_new_dm_true['full_name'].tolist()[
                                : self.max_inputs_nb
                                ]
            already_set_values = eval_input_new_dm_true['selected_input'].tolist()[
                                 : self.max_inputs_nb
                                 ]
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[
                    default_dataframe['full_name'] == name, 'selected_input'
                ] = already_set_values[index]
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_inputs',
                'value',
                default_dataframe,
                check_value=False,
            )

        if eval_output_new_dm is None:
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_outputs',
                'value',
                default_out_dataframe.sort_values(
                    by=['selected_output'], ascending=False).reset_index(drop=True),
                check_value=False,
            )

        # if eval output set for only certain var
        elif set(eval_output_new_dm['full_name'].tolist()) != (
                set(default_out_dataframe['full_name'].tolist())
        ):
            default_dataframe = copy.deepcopy(default_out_dataframe)
            already_set_names = eval_output_new_dm['full_name'].tolist()
            already_set_values = eval_output_new_dm['selected_output'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[
                    default_dataframe['full_name'] == name, 'selected_output'
                ] = already_set_values[index]
            self.dm.set_data(
                f'{self.get_disc_full_name()}.eval_outputs',
                'value',
                default_dataframe.sort_values(
                    by=['selected_output'], ascending=False).reset_index(drop=True),
                check_value=False,
            )

    def prepare_chart_dict(self, outputs_discipline_dict, inputs_discipline_dict={}):

        # generate mapping between longname and short name
        inputs_name_mapping = {}
        if 'eval_inputs' in inputs_discipline_dict:
            eval_inputs = inputs_discipline_dict['eval_inputs']
            eval_inputs_filtered = eval_inputs.loc[
                eval_inputs['selected_input'] == True, [
                    'full_name', 'shortest_name']
            ]
            eval_inputs_filtered_dict = eval_inputs_filtered.set_index(
                'full_name'
            ).to_dict('index')
            inputs_name_mapping = {
                k: v['shortest_name'] for k, v in eval_inputs_filtered_dict.items()
            }

        doe_samples_df = outputs_discipline_dict['samples_inputs_df']


        # retrive full input list
        inputs_list = [
            col for col in doe_samples_df.columns if col not in ['scenario']]
        inputs_list.sort(reverse=True)
        scenarii = len(doe_samples_df['scenario'])

        # generate all combinations of 2 inputs which will correspond to the
        # number of charts
        inputs_combin = list(itertools.combinations(inputs_list, 2))

        full_chart_list = []

        # output_df to stock all results
        output_df = None
        output_df_temp = None

        outputs_names_list = list(outputs_discipline_dict.keys())

        output_info_dict = {}

        if len(outputs_names_list) > 0:
            outputs_names = outputs_names_list.copy()
            outputs_names.remove('samples_inputs_df')
            for single_output in outputs_names:
                output_df_dict = outputs_discipline_dict[single_output]

                if isinstance(output_df_dict, dict):

                    if isinstance(list(output_df_dict.values())[0], dict):
                        # change from a dict of dicts to a dict of df
                        output_df_dict = {key: pd.DataFrame.from_records(
                            [output_df_dict[key]]) for key in output_df_dict}

                    if isinstance(list(output_df_dict.values())[0], float):
                        output_df_dict = {
                            key: pd.DataFrame({re.sub(r'_dict$', '', single_output).split('.')[-1]: value}, index=[0]) for
                            (key, value) in output_df_dict.items()}

                        # output_df=pd.DataFrame({'scenario':list(output_df_dict.keys()),
                        #     re.sub(r'_dict$', '', single_output).split('.')[-1]:list(output_df_dict.values())})

                    if (isinstance(list(output_df_dict.values())[0], pd.DataFrame)) and (
                            len(list(output_df_dict.values())[0]) == 1):

                        # we extract the columns of the dataframe of type float which will represents the possible outputs
                        # we assume that all dataframes contains the same columns
                        # and only look at the first element

                        # We select the outputs to plot at the first element
                        # list(output_df_dict.values())[0].replace('NA', np.nan, inplace=True)
                        filtered_name = [col for col in list(output_df_dict.values())[0].columns if
                                         ((list(output_df_dict.values())[
                                               0][col].dtype == 'float') or (list(output_df_dict.values())[0][col][0] == 'NA'))]

                        if len(filtered_name) > 0:

                            # to delete the outputs values that are nan or 'NA'
                            for col in filtered_name:
                                if all(pd.isna([list(output_df_dict.values())[i][col] for i in range(scenarii)])) or all(
                                        (list(output_df_dict.values())[i][col][0] == 'NA') for i in range(scenarii)):
                                    filtered_name.remove(col)

                            # Transform the output_dict into  dataframe with length = number of scenarios (output_df)
                            # and number of columns = all results for all
                            # single outputs
                            for scenario, df in output_df_dict.items():
                                filtered_df = df.copy(deep=True)
                                filtered_df = filtered_df[filtered_name]
                                filtered_df['scenario'] = f'{scenario}'
                                filtered_df.replace('NA', np.nan, inplace=True)

                                for name in filtered_name:
                                    if name not in list(output_info_dict.keys()):
                                        output_info_dict[name] = {
                                            'output_info_name': re.sub(r'_dict$', '', single_output),
                                            'unit': self.ee.dm.get_data(self.ee.dm.get_all_namespaces_from_var_name(
                                                re.sub(r'_dict$', '', single_output))[0])['unit']}

                                if output_df is None:
                                    output_df = filtered_df.copy(deep=True)
                                elif len(output_df) == scenarii:

                                    if output_df_temp is None:
                                        output_df_temp = filtered_df.copy(
                                            deep=True)
                                    else:
                                        output_df_temp = pd.concat(
                                            [output_df_temp, filtered_df], axis=0, ignore_index=True)
                                else:
                                    output_df = pd.concat(
                                        [output_df, filtered_df], axis=0, ignore_index=True)

                            if output_df_temp is not None:
                                output_df_temp.replace(
                                    'NA', np.nan, inplace=True)
                                columns_temp = list(output_df_temp.columns)
                                columns_temp.remove('scenario')
                                for col in columns_temp:
                                    if col not in output_df:
                                        output_df.merge(
                                            output_df_temp[['scenario', col]], on='scenario', how='left')
                                output_df_temp = None

            # Select only float type results
            output_variables = output_df.select_dtypes(
                include='float').columns.to_list()
            cont_plot_df = doe_samples_df.merge(
                output_df, how="left", on='scenario')

            if 'reference' in list(cont_plot_df['scenario'].values):
                # Stock the scenario reference row
                reference_row = cont_plot_df.loc[cont_plot_df['scenario'] == 'reference', :].reset_index(drop=True)
                # remove the scenario reference row from cont_plot_df
                cont_plot_df = cont_plot_df.loc[cont_plot_df['scenario'] != 'reference', :]
            else:
                reference_row = []

            # we constitute the full_chart_list by making a product
            # between the possible inputs combination and outputs list
            chart_tuples = list(
                itertools.product(
                    inputs_combin, output_variables)
            )
            # retrieve z variable name by removing _dict from the output name

            chart_list = [list(chart_tuple) + [output_info_dict[chart_tuple[1]]['output_info_name']] + [
                output_info_dict[chart_tuple[1]]['unit']]
                          for chart_tuple in chart_tuples]
            full_chart_list += chart_list

        chart_dict = {}
        # based on the full chart list, we will create a dict will all
        # necessary information for each chart
        for chart in full_chart_list:
            # we store x,y and z fullname variable
            z_vble = chart[1]
            x_vble = chart[0][0]
            y_vble = chart[0][1]

            # we retrieve the corresponding short name for x and y
            # GET-->first parameter=key, second parameter=value if the key is
            # not found
            x_short = inputs_name_mapping.get(x_vble, x_vble)
            y_short = inputs_name_mapping.get(y_vble, y_vble)

            # we add a slider if necessary
            slider_list = []
            for col in inputs_list:
                if col not in chart[0]:
                    slider = {
                        'full_name': col,
                        'short_name': inputs_name_mapping.get(col, col),
                        'unit': self.ee.dm.get_data(
                            self.ee.dm.get_all_namespaces_from_var_name(col)[0]
                        )['unit'],
                    }
                    slider_list.append(slider)

            chart_name = None
            if slider_list != []:
                z_data_None = []
                col_slider = slider_list[0]['full_name']
                slider_values = cont_plot_df[col_slider].unique()
                for slide_value in slider_values:
                    z_data = list(
                        cont_plot_df.loc[cont_plot_df[col_slider] == slide_value, z_vble])
                    if all(np.isnan(z_data[i]) for i in range(len(z_data))):
                        z_data_None = True
                        break

                if not z_data_None:
                    chart_name = f'{z_vble} contour plot with {slider_list[0]["short_name"]} as slider'
                    chart_data = cont_plot_df.loc[:, [
                                                         'scenario', x_vble, y_vble,slider_list[0]['full_name'], z_vble ]]

            elif slider_list == []:
                chart_name = f'{z_vble} contour plot'
                chart_data = cont_plot_df.loc[:, [
                                                     'scenario', x_vble, y_vble, z_vble]]

            if chart_name is not None:
                chart_dict[chart_name] = {
                    'x': x_vble,
                    'x_short': x_short,
                    'x_unit': self.ee.dm.get_data(
                        self.ee.dm.get_all_namespaces_from_var_name(x_vble)[0]
                    )['unit'],
                    'y': y_vble,
                    'y_short': y_short,
                    'y_unit': self.ee.dm.get_data(
                        self.ee.dm.get_all_namespaces_from_var_name(y_vble)[0]
                    )['unit'],
                    'z': z_vble,
                    'z_unit': chart[3],
                    'z_max': output_df[z_vble].max(skipna=True),
                    'z_min': output_df[z_vble].min(skipna=True),
                    'slider': slider_list,
                    'chart_data': chart_data,
                    'reference_scenario': reference_row,
                }
                output_df_to_return = output_df.copy()
                scenario_column = output_df_to_return.pop('scenario')
                output_df_to_return.insert(0, 'scenario', scenario_column)

        return chart_dict, output_df_to_return

    def get_chart_filter_list(self):
        
        chart_filters = []
        outputs_dict = self.get_sosdisc_outputs()
        inputs_dict = self.get_sosdisc_inputs()
        self.chart_dict, output_df = self.prepare_chart_dict(
            outputs_dict, inputs_dict)
        chart_list = list(self.chart_dict.keys()) 
        if 'reference_scenario' in self.chart_dict[chart_list[0]] :
            if len(self.chart_dict[chart_list[0]]['reference_scenario']):
                chart_list.insert(0,'Reference Scenario')
                
        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'Charts'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        outputs_dict = self.get_sosdisc_outputs()
        inputs_dict = self.get_sosdisc_inputs()
        self.chart_dict, output_df = self.prepare_chart_dict(
            outputs_dict, inputs_dict)
        # chart_dict = self.chart_dict

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'Charts':
                    graphs_list = chart_filter.selected_values

        if 'Reference Scenario' in graphs_list:
            
            fig = go.Figure()
            
            first_chart_info=self.chart_dict[list(self.chart_dict.keys())[0]]
            eval_inputs_df=inputs_dict['design_space'].copy(deep=True)
            eval_in_list=eval_inputs_df['shortest_name'].to_list()
            ref_scen_df=self.get_postprocessing_table(first_chart_info, eval_in_list)
            colores = ['lavender' if tipo=='input' in ref_scen_df['Type'].to_list() else 'floralwhite' for tipo in ref_scen_df['Type'] ]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=ref_scen_df.columns.tolist(),
                        fill_color='midnightblue',
                        align='center',
                        font_color='white',
                    ),
                    cells=dict(
                        values=ref_scen_df.T.values.tolist(),
                        fill_color=[colores],
                        align='center',
                    ),
                )
            )

            table_name = 'Reference Scenario Data'
            number_of_rows = ref_scen_df.shape[0]
            fig.update_layout(
                title_text='Reference Scenario',
                showlegend=False,
                autosize=True,
                height=number_of_rows * 30 + 250,
            )
            
            if len(fig.data):
                    # Create native plotly chart
                new_chart = InstantiatedPlotlyNativeChart(
                    fig=fig, chart_name=table_name, default_legend=False, default_font=True, with_default_annotations=False)
                # new_chart.to_plotly().show()
                instanciated_charts.append(new_chart)
            
        if len(self.chart_dict.keys()) > 0:
            # we create a unique dataframe containing all data that will be
            # used for drawing the graphs

                # we go through the list of charts and draw all of them
            for name, chart_info in self.chart_dict.items():
                if name in graphs_list:
                    
                    if len(chart_info['slider']) == 0:

                        fig = go.Figure()

                        x_data = chart_info['chart_data'][chart_info['x']].to_list(
                        )

                        y_data = chart_info['chart_data'][chart_info['y']].to_list(
                        )
                        z_data = chart_info['chart_data'][chart_info['z']].replace(
                            np.nan, 'None').to_list()


                        x_max = max(x_data)
                        y_max = max(y_data)
                        x_min = min(x_data)
                        y_min = min(y_data)

                        fig.add_trace(
                            go.Contour(
                                x=x_data,
                                y=y_data,
                                z=z_data,
                                colorscale='YlGnBu', reversescale=True,
                                contours=dict(
                                    coloring='heatmap',
                                    showlabels=True,  # show labels on contours
                                    labelfont=dict(  # label font properties
                                        size=10,
                                        # color = 'white',
                                    ),

                                ),
                                colorbar=dict(
                                    title=f'{chart_info["z"]}',
                                    nticks=10,
                                    ticklen=5,
                                    tickwidth=1,
                                    ticksuffix=f'{chart_info["z_unit"]}',
                                    tickangle=0,
                                    tickfont_size=10,
                                ),
                                visible=True,
                                connectgaps=False,
                                hovertemplate='{}'.format(chart_info["x_short"]) + ': %{x}' +

                                            '<br>{}'.format(chart_info["y_short"]) + ': %{y}' +
                                            '<br><b>{}<b>'.format(
                                                chart_info["z"]) + ': <b> %{z}<b>' + '<b> {}<b><br>'.format(
                                    chart_info["z_unit"]),
                                name="",

                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode='markers',
                                marker_symbol=SymbolValidator(
                                ).values[SymbolValidator().values.index('x-thin')],
                                marker=dict(
                                    size=5,
                                    color='dimGray',
                                    symbol='x-thin',
                                    line=dict(
                                        # color='MediumPurple',
                                        width=2,
                                    )
                                ),
                                visible=True,
                                showlegend=False,
                                hoverinfo='skip'
                                )
                            )

                        if len(chart_info['reference_scenario']):

                            x_ref_scen=chart_info['reference_scenario'][chart_info['x']].to_list()
                            y_ref_scen=chart_info['reference_scenario'][chart_info['y']].to_list()
                            z_ref_scen=float(chart_info['reference_scenario'][chart_info['z']].values)
                            legend_letter, factor, z_ref_hover = get_order_of_magnitude(z_ref_scen)
                            


                            # if float(chart_info['reference_scenario'][col_slider])==slide_value:

                            fig.add_trace(
                                go.Scatter(
                                    x=x_ref_scen,
                                    y=y_ref_scen,
                                    mode='markers',
                                    marker_symbol=SymbolValidator(
                                    ).values[SymbolValidator().values.index('star')],
                                    marker_color="#f03b20",
                                    marker_size=10,
                                    visible=True,
                                    showlegend=False,
                                    hovertemplate='{}'.format(chart_info["x_short"]) + ': %{x}' +

                                        '<br>{}'.format(chart_info["y_short"]) + ': %{y}' +
                                        '<br>{} '.format(
                                            chart_info["z"]) + ': <b> {} {}<b>'.format(round(z_ref_hover,5),legend_letter) + '<b> {}<b><br>'.format(chart_info["z_unit"]),
                                    name='Reference Scenario',

                                )
                            )

                        fig.update_layout(
                            autosize=True,
                            xaxis=dict(
                                title=chart_info['x_short'],
                                ticksuffix=chart_info["x_unit"],
                                titlefont_size=12,
                                tickfont_size=10,
                                automargin=True,
                                range=[x_min, x_max],
                            ),
                            yaxis=dict(
                                title=chart_info['y_short'],
                                ticksuffix=chart_info["y_unit"],
                                titlefont_size=12,
                                tickfont_size=10,
                                automargin=True,
                                range=[y_min, y_max],
                            ),
                        )

                        if len(fig.data) > 0:
                            chart_name = f'<b>{name}</b>'
                            new_chart = InstantiatedPlotlyNativeChart(
                                fig=fig, chart_name=chart_name, default_legend=False
                            )
                            instanciated_charts.append(new_chart)

                    if len(chart_info['slider']) == 1:
                        col_slider = chart_info['slider'][0]['full_name']
                        slider_short_name = chart_info['slider'][0]['short_name']
                        slider_unit = chart_info['slider'][0]['unit']
                        slider_values = chart_info['chart_data'][col_slider].unique(
                        )
                        z_max = chart_info['z_max']
                        z_min = chart_info['z_min']

                        fig = go.Figure()

                        for slide_value in slider_values:
                            x_data = chart_info['chart_data'].loc[
                                chart_info['chart_data'][col_slider] == slide_value
                                ][chart_info['x']].to_list()
                            y_data = chart_info['chart_data'].loc[
                                chart_info['chart_data'][col_slider] == slide_value
                                ][chart_info['y']].to_list()
                            z_data = chart_info['chart_data'].loc[
                                chart_info['chart_data'][col_slider] == slide_value
                                ][chart_info['z']].replace(np.nan, 'None').to_list()

                            x_max = max(x_data)
                            x_min = min(x_data)
                            y_min = min(y_data)
                            y_max = max(y_data)

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
                                    colorscale='YlGnBu', reversescale=True,
                                    contours=dict(
                                        coloring='heatmap',
                                        showlabels=True,  # show labels on contours
                                        labelfont=dict(  # label font properties
                                            size=10,
                                            # color = 'white',
                                        ),
                                        start=z_min,
                                        end=z_max,

                                    ),
                                    colorbar=dict(
                                        title=f'{chart_info["z"]}',
                                        nticks=10,
                                        ticks='outside',
                                        ticklen=5,
                                        tickwidth=1,
                                        ticksuffix=f'{chart_info["z_unit"]}',
                                        # showticklabels=True,
                                        tickangle=0,
                                        tickfont_size=10,
                                    ),
                                    visible=visible,
                                    connectgaps=False,
                                    hovertemplate='{}'.format(chart_info["x_short"]) + ': %{x}' +
                                                '<br>{}'.format(chart_info["y_short"]) + ': %{y}' +
                                                '<br><b>{}<b>'.format(
                                                    chart_info["z"]) + ': <b> %{z}<b>' + '<b> {}<b><br>'.format(
                                        chart_info["z_unit"]),
                                    name='{} '.format(
                                        slider_short_name) + ': {} {}'.format(round(slide_value,2),slider_unit),
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=x_data,
                                    y=y_data,
                                    mode='markers',
                                    marker_symbol=SymbolValidator(
                                    ).values[SymbolValidator().values.index('x-thin')],
                                    marker=dict(
                                        size=5,
                                        color='dimGray',
                                        line=dict(
                                            # color='MediumPurple',
                                            width=2,
                                        )
                                    ),
                                    visible=visible,
                                    showlegend=False,
                                    hoverinfo='skip',
                                )
                            )

                            if len(chart_info['reference_scenario']):
                                x_ref_scen = chart_info['reference_scenario'].loc[
                                    chart_info['reference_scenario'][col_slider] == slide_value][
                                    chart_info['x']].to_list()
                                y_ref_scen = chart_info['reference_scenario'].loc[
                                    chart_info['reference_scenario'][col_slider] == slide_value][
                                    chart_info['y']].to_list()
                                
                                z_ref_scen=float(chart_info['reference_scenario'][chart_info['z']].values)
                                legend_letter, factor, z_ref_hover = get_order_of_magnitude(z_ref_scen)

                                # if float(chart_info['reference_scenario'][col_slider])==slide_value:
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_ref_scen,
                                        y=y_ref_scen,
                                        mode='markers',
                                        marker_symbol=SymbolValidator(
                                        ).values[SymbolValidator().values.index('star')],
                                        marker_color="#f03b20",
                                        marker_size=10,
                                        visible=visible,
                                        showlegend=False,
                                        hovertemplate='{}'.format(chart_info["x_short"]) + ': %{x}' +
                                                    '<br>{}'.format(chart_info["y_short"]) + ': %{y}' +
                                                    '<br>{} '.format(
                                                        slider_short_name) + ': {}'.format(float(
                                            chart_info['reference_scenario'][col_slider])) + f'{slider_unit}' +
                                                    '<br><b>{}<b>'.format(
                                                        chart_info["z"]) + ': <b> {} {}<b>'.format(round(z_ref_hover,5),legend_letter) + '<b> {}<b><br>'.format(chart_info["z_unit"]),
                                        name='Reference Scenario',
                                    )
                                )

                        # Create and add slider
                        steps = []
                        if len(chart_info['reference_scenario']):
                            lid = 3
                        else:
                            lid = 2

                        for i in range(int(len(fig.data) / lid)):
                            # for i in range(int(len(fig.data) / 2)):

                            step = dict(
                                method="update",
                                args=[
                                    {"visible": [False] * len(fig.data)},
                                    {"title": f'<b>{name}</b>'},
                                ],
                                # layout attribute
                                label='{} {}'.format(round(slider_values[i],2),slider_unit),
                            )
                            # Toggle i'th trace to 'visible'
                            for k in range(lid):
                                step['args'][0]['visible'][i * lid + k] = True
                            steps.append(step)
                            # for k in range(2):
                            # step['args'][0]['visible'][i * 2 + k] = True

                        sliders = [
                            dict(
                                active=len(steps) - 1,
                                currentvalue={
                                    'visible': True,
                                    "prefix": f'{slider_short_name}: ',
                                },
                                steps=steps,
                                pad=dict(t=50),
                            )
                        ]

                        fig.update_layout(
                            sliders=sliders,
                            autosize=True,
                            hoverlabel_align='left',
                            xaxis=dict(
                                title=f'{chart_info["x_short"]}',
                                ticksuffix=chart_info["x_unit"],
                                titlefont_size=12,
                                tickfont_size=10,
                                automargin=True,
                                range=[x_min, x_max],
                            ),
                            yaxis=dict(
                                title=f'{chart_info["y_short"]}',
                                titlefont_size=12,
                                tickfont_size=10,
                                ticksuffix=chart_info["y_unit"],
                                # tickformat=',.0%',
                                automargin=True,
                                range=[y_min, y_max],
                            ),
                        )

                        # Create native plotly chart
                        last_value = slider_values[-1]
                        if len(fig.data) > 0:
                            chart_name = f'<b>{name}</b>'
                            new_chart = InstantiatedPlotlyNativeChart(
                                fig=fig, chart_name=chart_name, default_legend=False
                            )
                            instanciated_charts.append(new_chart)

            return instanciated_charts

    def get_postprocessing_table(self, ref_scen_dict, eval_in_list):
        
        row_ref_scen=copy.deepcopy(ref_scen_dict['reference_scenario'])
        if len(ref_scen_dict['slider'])==0: 
            row_ref_scen.rename(columns={
                ref_scen_dict['x']:ref_scen_dict['x_short'],
                ref_scen_dict['y']:ref_scen_dict['y_short'],
            }, inplace=True)
        elif len(ref_scen_dict['slider'])==1:
            row_ref_scen.rename(columns={
                ref_scen_dict['x']:ref_scen_dict['x_short'],
                ref_scen_dict['y']:ref_scen_dict['y_short'],
                ref_scen_dict['slider'][0]['full_name']:ref_scen_dict['slider'][0]['short_name'],
            }, inplace=True)
        
        
        df=row_ref_scen.drop(['scenario'], axis='columns').T.reset_index()
        df.columns=['Parameter', 'table_value']
        df['Type']=["input" if param in eval_in_list else "output" for param in df['Parameter']] 
        df['pretty_value']=df['table_value'].apply(lambda x: str(round(get_order_of_magnitude(x)[-1],5)) + ' '+ get_order_of_magnitude(x)[0])
        df['unit']='None'    
        df['unit'].where(df['Parameter']!=ref_scen_dict['x_short'], ref_scen_dict['x_unit'], inplace=True)
        df['unit'].where(df['Parameter']!=ref_scen_dict['y_short'], ref_scen_dict['y_unit'], inplace=True)
        if len(ref_scen_dict['slider'])==1:
            df['unit'].where(df['Parameter']!=ref_scen_dict['slider'][0]['short_name'], ref_scen_dict['slider'][0]['unit'], inplace=True)
        df['unit'].where(df['unit']!='None', ref_scen_dict['z_unit'], inplace=True)
        df.replace(np.nan,'None', inplace=True)
        df['Value']=df['pretty_value'] +  df['unit']
        
        table_fig=df[['Type','Parameter','Value']]
        
        return table_fig

def get_order_of_magnitude(maxvalue):

    if abs(maxvalue) >= 1.0e9:
        maxvalue = maxvalue / 1.0e9
        legend_letter = 'B'
        factor = 1.0e09
    elif 1.0e9 > abs(maxvalue) >= 1.0e6:
        maxvalue = maxvalue / 1.0e6
        legend_letter = 'M'
        factor = 1.0e06
    elif 1.0e06 > abs(maxvalue) >= 1.0e03:
        maxvalue = maxvalue / 1.0e3
        legend_letter = 'k'
        factor = 1.0e03
    else:
        legend_letter = ''
        factor = 1.0

    return legend_letter, factor, maxvalue

