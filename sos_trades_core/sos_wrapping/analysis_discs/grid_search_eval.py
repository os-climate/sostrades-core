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
            self.selected_inputs = selected_inputs.tolist()
            self.selected_outputs = selected_outputs.tolist()

            self.set_eval_in_out_lists(selected_inputs, selected_outputs)

            # grid8seqrch can be done only for selected inputs and outputs
            if (len(self.eval_in_list) > 0) and (len(self.eval_out_list) > 0):

                # setting dynamic outputs. One output of type dict per selected
                # output
                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var.split(self.ee.study_name + ".")[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
                                                                                'namespace': 'ns_doe'}})

                default_design_space = pd.DataFrame({self.VARIABLES: selected_inputs,

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
        # check if the eval_inputs need to be updtated after a subprocess
        # configure
        elif eval_input_new_dm['full_name'].equals(default_in_dataframe['full_name']) == False:
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)
