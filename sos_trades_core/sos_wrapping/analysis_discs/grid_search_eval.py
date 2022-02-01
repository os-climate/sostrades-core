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

from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import copy
from copy import deepcopy
import collections

from sos_trades_core.api import get_sos_logger
from numpy import array, ndarray, delete, NaN

from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.tools.scenario.scenario_generator import ScenarioGenerator
from sos_trades_core.sos_wrapping.analysis_discs.doe_eval import DoeEval
from gemseo.algos.doe.doe_factory import DOEFactory

from gemseo.algos.design_space import DesignSpace


class GridSearchEval(DoeEval):
    '''
    Generic Grid Search evaluation class
    '''

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
        Overload setup_sos_disciplines to create a dynamic desc_in
        default descin are the algo name and its options
        In case of a custom_doe, additionnal input is the customed sample ( dataframe)
        In other cases, additionnal inputs are the number of samples and the design space
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
            self.selected_inputs = selected_inputs.tolist()
            self.selected_outputs = selected_outputs.tolist()

            self.set_eval_in_out_lists(selected_inputs, selected_outputs)

            # grid8seqrch can be done only for selected inputs and outputs
            if (len(self.eval_in_list) > 0) and (len(self.eval_out_list) > 0):

                # setting dynamic outputs. One output of type dict per selected
                # output
                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var}_dict': {'type': 'dict'}})

                default_design_space = pd.DataFrame({self.VARIABLES: self.eval_in_list,

                                                     self.LOWER_BOUND: [array([0.0, 0.0]) if self.ee.dm.get_data(var,
                                                                                                                 'type') == 'array' else 0.0
                                                                        for var in self.eval_in_list],
                                                     self.UPPER_BOUND: [array([10.0, 10.0]) if self.ee.dm.get_data(var,
                                                                                                                   'type') == 'array' else 10.0
                                                                        for var in self.eval_in_list],
                                                     self.NB_POINTS: 2
                                                     })

                dynamic_inputs.update(
                    {'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
                                      }})
                if 'design_space' in self._data_in:
                    self._data_in['design_space']['value'] = default_design_space

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        super(GridSearchEval, self).__init__(sos_name, ee, cls_builder)
        self.doe_factory = DOEFactory()
        self.logger = get_sos_logger(f'{self.ee.logger.name}.GridSearch')
        self.eval_input_types = ['float', 'int', 'string']
        self.eval_in_list = []
        self.eval_out_list = []

    def generate_samples_from_doe_factory(self):
        """Generating samples for the Doe using the Doe Factory
        """
        algo_name = 'fullfact'
        ds = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        options = {'levels': ds['nb_points'].to_list()}

        self.design_space = self.create_design_space()

        filled_options = {}
        for algo_option in options:
            if options[algo_option] != 'default':
                filled_options[algo_option] = options[algo_option]

        # if 'levels' in options:
        #     options['levels'] = options['levels'].astype(int).tolist()
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

    def set_design_space(self):
        """
        reads design space (set_design_space)
        """

        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        variables = self.eval_in_list
        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds
        enable_variables = [True for invar in self.eval_in_list]

        activated_elems = [[True, True] if self.ee.dm.get_data(var, 'type') == 'array' else [True] for var in
                           self.eval_in_list]

        dspace_dict_updated = pd.DataFrame({self.VARIABLES: variables,
                                            self.VALUES: values,
                                            self.LOWER_BOUND: lower_bounds,
                                            self.UPPER_BOUND: upper_bounds,
                                            self.ENABLE_VARIABLE_BOOL: enable_variables,
                                            self.LIST_ACTIVATED_ELEM: activated_elems})

        design_space = self.read_from_dataframe(dspace_dict_updated)

        return design_space

    def _fill_possible_values(self, disc):
        '''
            Fill possible values for eval inputs and outputs: tuples with (name, namespace)
            an input variable must be a float, int or string coming from a data_in of a discipline in all the process
            an output variable must be any data from a data_out discipline
        '''
        name_in = []
        name_out = []
        for data_in_key in disc._data_in.keys():
            is_input_types = disc._data_in[data_in_key][self.TYPE] in self.eval_input_types
            in_coupling_numerical = data_in_key in list(SoSCoupling.DESC_IN.keys()
                                                        ) + list(SoSDiscipline.NUM_DESC_IN.keys())
            if is_input_types and not in_coupling_numerical:
                namespaced_data = disc.get_var_full_name(
                    data_in_key, disc._data_in)
                # remove usecase name
                namespaced_data = namespaced_data.split('.', 1)[1]
                name_in.append(namespaced_data)
        for data_out_key in disc._data_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            namespaced_data = disc.get_var_full_name(
                data_out_key, disc._data_out)
            # remove usecase name
            namespaced_data = namespaced_data.split('.', 1)[1]
            name_out.append(namespaced_data)

        return name_in, name_out

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

        default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values_full],
                                             'full_name': possible_in_values_full})
        default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values_full],
                                              'full_name': possible_out_values_full})

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        if eval_input_new_dm is None:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)
