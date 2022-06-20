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
from copy import deepcopy
import collections

from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.tools.scenario.scenario_generator import ScenarioGenerator


class SoSMorphMatrixEval(SoSEval):
    '''
        SOSMorphMatrixEval class which creates a sub process to evaluate, built from morphological matrix of scenarios
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Morphological Matrix Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-border-all fa-fw',
        'version': '',
    }
    EVAL_INPUTS = 'eval_inputs'
    EVAL_OUTPUTS = 'eval_outputs'
    ACTIVATION_DF = 'activation_morphological_matrix'
    SELECTED_SCENARIOS = 'selected_scenarios'

    DEFAULT_DF_DESCRIPTOR = {'selected_scenario': ('bool', None, True),
                             'scenario_name':  ('string', None, True)}

    DESC_IN = {ACTIVATION_DF: {'type': 'dataframe',
                               'dataframe_descriptor': DEFAULT_DF_DESCRIPTOR,
                               'dataframe_edition_locked': False,
                               'default': DataFrame(),
                               'user_level': 99,
                               'structuring': True},
               SELECTED_SCENARIOS: {'type': 'dataframe',
                                    'dataframe_edition_locked': True,
                                    'default': DataFrame(),
                                    'user_level': 99,
                                    'editable': False},
               EVAL_INPUTS: {'type': 'dataframe',
                             'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                      'name':  ('string', None, False),
                                                      'namespace':  ('string', None, False),
                                                      'input_variable_name': ('string', None, True)},
                             'dataframe_edition_locked': False,
                             'structuring': True},
               EVAL_OUTPUTS: {'type': 'dataframe',
                              'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                       'name':  ('string', None, False),
                                                       'namespace':  ('string', None, False),
                                                       'output_variable_name': ('string', None, True)},
                              'dataframe_edition_locked': False,
                              'structuring': True},
               'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
               'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0}
               }

    def __init__(self, sos_name, ee, cls_builder):
        '''
            Constructor
        '''
        super(SoSMorphMatrixEval, self).__init__(sos_name, ee, cls_builder)
        self.scenario_generator = ScenarioGenerator()
        # allowed input types
        self.eval_input_types = ['float', 'int', 'string']
        # x0 for evaluation
        self.initial_inputs = None
        # eval_input dict of values
        self.eval_input_dict = {}
        self.namespaced_eval_inputs = {}
        self.namespaced_eval_outputs = {}
        self.OLD_MORPHOLOGICAL_MATRIX_DF = None

    def get_eval_input_dict(self, selected_inputs):
        '''
            Build dictionary of values for selected_inputs
        '''
        eval_input_dict = {}
        for input in selected_inputs:
            if input in self._data_in:
                if self.get_sosdisc_inputs(input) is not None:
                    eval_input_dict[input] = self.get_sosdisc_inputs(
                        input)

        return eval_input_dict

    def setup_sos_disciplines(self):
        '''
            Overloaded SoSDiscipline method to build dynamic inputs and outputs
        '''
        if len(self._data_in) > 0:

            eval_inputs, eval_outputs, activation_df = self.get_sosdisc_inputs(
                [self.EVAL_INPUTS, self.EVAL_OUTPUTS, self.ACTIVATION_DF])

            if eval_inputs is not None:
                # check eval_inputs and eval_outputs content
                selected_inputs, selected_outputs = self.check_eval_inputs_output(
                    eval_inputs, eval_outputs)

                if len(selected_inputs) > 0:
                    # build dictionary of values for selected_inputs
                    eval_input_dict = self.get_eval_input_dict(selected_inputs)

                    if selected_inputs != list(self.eval_input_dict.keys()):
                        # build selected eval inputs
                        self.add_dynamic_eval_inputs(
                            selected_inputs, eval_inputs)

                        # update dataframe descriptor
                        df_descriptor = {key: value for key,
                                         value in self.DEFAULT_DF_DESCRIPTOR.items()}
                        df_descriptor.update({key: ('string', None, False)
                                              for key in selected_inputs})
                        self._data_in[self.ACTIVATION_DF].update(
                            {self.DATAFRAME_DESCRIPTOR: df_descriptor})

                    if list(eval_input_dict.values()) != list(self.eval_input_dict.values()):
                        # build morphological matrix with modified eval inputs
                        # values
                        self.eval_input_dict = eval_input_dict
                        morphological_matrix_df = self.generate_morphological_matrix(
                            eval_input_dict)

                        if len(morphological_matrix_df.values.tolist()) > 0:
                            # get activation_df columns except
                            # ['selected_scenario', 'scenario_name']
                            activation_df_eval_inputs = activation_df[[
                                col for col in activation_df.columns if col not in self.DEFAULT_DF_DESCRIPTOR.keys()]]
                            if self._data_in[self.ACTIVATION_DF][self.USER_LEVEL] > 0:
                                self._data_in[self.ACTIVATION_DF][self.USER_LEVEL] = 0
                                self._data_in[self.SELECTED_SCENARIOS][self.USER_LEVEL] = 0

                            if not activation_df_eval_inputs.columns.tolist() == list(eval_input_dict.keys()) or not morphological_matrix_df[list(eval_input_dict.keys())].equals(activation_df_eval_inputs):
                                # store morphological matrix dataframe if
                                # columns or values have been modified
                                # IF EXPLANATION :
                                # if we have an activation df not empty and not none
                                # if thre is more columns in the activation_df than the eval _inputs (to store lines coming from future configure)
                                # if there is less columns that means that we have added a trade variable and the morph matrix is reset
                                # if the activation_df is not the same than the
                                # morphological_df of the last configure
                                if activation_df is not None and len(activation_df) != 0 \
                                        and activation_df_eval_inputs.columns.tolist() >= list(eval_input_dict.keys())\
                                        and (self.OLD_MORPHOLOGICAL_MATRIX_DF is None
                                             or (self.OLD_MORPHOLOGICAL_MATRIX_DF is not None and not activation_df.equals(self.OLD_MORPHOLOGICAL_MATRIX_DF))):
                                    # merge values of the activation_df given
                                    # by the user in the morph matrix computed
                                    # by the discipline in an outer way to get
                                    # all lines not specified by the user +
                                    # lines specified by the user not yet
                                    # configured
                                    morph_matrix_eval_inputs = morphological_matrix_df[[
                                        col for col in morphological_matrix_df.columns if col not in self.DEFAULT_DF_DESCRIPTOR.keys()]]
                                    updated_activation_df = morphological_matrix_df.merge(
                                        activation_df, how='outer', on=morph_matrix_eval_inputs.columns.tolist(), suffixes=('', '_new'))
                                    # once we have merged with specic suffixes (new corresponds to activation_df)
                                    # we get the value of the new activation_df where there is not Nan values with a np.where
                                    # and delete the new column
                                    # Assure the dtype is the same because Nan
                                    # can change the type of the column
                                    for other_key in ['selected_scenario', 'scenario_name']:
                                        updated_activation_df[other_key] = np.where(pd.notnull(
                                            updated_activation_df[f'{other_key}_new']), updated_activation_df[f'{other_key}_new'], updated_activation_df[other_key])
                                        updated_activation_df.drop(
                                            f'{other_key}_new', axis=1, inplace=True)
                                        updated_activation_df = updated_activation_df.astype(
                                            {other_key: morphological_matrix_df[other_key].dtype})
                                else:
                                    # we reset the default with the morph
                                    # matrix
                                    updated_activation_df = morphological_matrix_df

                                self.update_default_value(
                                    self.ACTIVATION_DF, self.IO_TYPE_IN, updated_activation_df)
                                self.OLD_MORPHOLOGICAL_MATRIX_DF = morphological_matrix_df

            if eval_outputs is not None:
                # build selected eval outputs
                self.add_dynamic_eval_outputs(
                    selected_outputs, eval_outputs)

            self.update_selected_scenarios_input()

    def update_selected_scenarios_input(self):
        '''
        Update value of selecetd_scenarios with activation_morphological_matrix
        '''
        activation_df = self.get_sosdisc_inputs(
            self.ACTIVATION_DF)
        if 'selected_scenario' in activation_df.columns:
            updated_selected_scenarios_df = activation_df[activation_df['selected_scenario']][[
                col for col in activation_df.columns if col != 'selected_scenario']]
            self.update_default_value(
                self.SELECTED_SCENARIOS, self.IO_TYPE_IN, updated_selected_scenarios_df)

    def check_eval_inputs_output(self, eval_inputs, eval_outputs):
        '''
            Check eval_inputs/eval_outputs content and return empty selected_inputs/selected_outputs if incomplete or not correctly completed
        '''
        selected_inputs = eval_inputs.loc[eval_inputs['selected_input'],
                                          'input_variable_name'].values.tolist()
        selected_outputs = eval_outputs.loc[eval_outputs['selected_output'],
                                            'output_variable_name'].values.tolist()

        if len(selected_inputs) == 0:
            self.ee.logger.error(
                'Evaluated Inputs: select at least one input')
        elif '' in selected_inputs:
            missing_input_variable_names = eval_inputs.loc[(eval_inputs['selected_input']) & 
                                                           (eval_inputs['input_variable_name'] == ''), 'name'].values.tolist(
            )
            self.ee.logger.error(
                f'Evaluated Inputs: missing input_variable_name for {missing_input_variable_names}')
            selected_inputs = []
        elif len(set(selected_inputs)) < len(selected_inputs):
            duplicates = [item for item,
                          count in collections.Counter(selected_inputs).items() if count > 1]
            self.ee.logger.error(
                f'Evaluated Inputs: input_variable_name cannot be duplicated, set unique values for {duplicates}')
            selected_inputs = []

        if len(selected_outputs) == 0:
            self.ee.logger.error(
                'Evaluated Outputs: select at least one output')
        elif '' in selected_outputs:
            missing_output_variable_names = eval_outputs.loc[(eval_outputs['selected_output']) & 
                                                             (eval_outputs['output_variable_name'] == ''), 'name'].values.tolist(
            )
            self.ee.logger.error(
                f'Evaluated Outputs: missing output_variable_name for {missing_output_variable_names}')
            selected_outputs = []
        elif len(set(selected_outputs)) < len(selected_outputs):
            duplicates = [item for item,
                          count in collections.Counter(eval_outputs).items() if count > 1]
            self.ee.logger.error(
                f'Evaluated Outputs: output_variable_name cannot be duplicated, set unique values for {duplicates}')
            selected_outputs = []

        return selected_inputs, selected_outputs

    def add_dynamic_eval_inputs(self, selected_inputs, eval_inputs):
        '''
            Build local input named input_variable_name for input in selected_inputs
        '''
        dynamic_inputs = {}
        namespaced_eval_inputs = {}
        for input_name in selected_inputs:
            # build namespaced name from namespace, name in eval_inputs and
            # store it
            # Add self.ee.study_name to namespace
            namespace_build_list = [self.ee.study_name]
            namespace_build_list.extend(eval_inputs.loc[eval_inputs['input_variable_name'] == input_name, [
                'namespace', 'name']].values.tolist()[0])
            input_full_name = '.'.join(namespace_build_list)
            namespaced_eval_inputs[input_name] = input_full_name

            if self.ee.dm.check_data_in_dm(input_full_name):
                dynamic_inputs[input_name] = {self.TYPE: f'{self.ee.dm.get_data(input_full_name, self.TYPE)}_list',
                                              self.RANGE: self.ee.dm.get_data(input_full_name, self.RANGE),
                                              self.POSSIBLE_VALUES: self.ee.dm.get_data(input_full_name, self.POSSIBLE_VALUES),
                                              self.DEFAULT: [],
                                              self.STRUCTURING: True}

        if dynamic_inputs is not {}:
            self.add_inputs(dynamic_inputs, clean_inputs=True)
        self.namespaced_eval_inputs = namespaced_eval_inputs

        return dynamic_inputs

    def add_dynamic_eval_outputs(self, selected_outputs, eval_outputs):
        '''
            Build local output named output_variable_name for output in selected_outputs
        '''
        dynamic_outputs = {}
        namespaced_eval_outputs = {}
        for output_name in selected_outputs:
            dynamic_outputs[output_name] = {self.TYPE: 'dict'}

            # build namespaced name from namespace, name in eval_outputs and
            # store it
            # Add self.ee.study_name to namespace
            namespace_build_list = [self.ee.study_name]
            namespace_build_list.extend(eval_outputs.loc[eval_outputs['output_variable_name'] == output_name, [
                'namespace', 'name']].values.tolist()[0])
            output_full_name = '.'.join(namespace_build_list)

            namespaced_eval_outputs[output_name] = output_full_name

        self.add_outputs(dynamic_outputs, clean_outputs=True)
        self.namespaced_eval_outputs = namespaced_eval_outputs

    def generate_morphological_matrix(self, eval_input_dict):
        '''
            Generate morphological matrix of input combination scenarios
        '''
        morphological_matrix = self.scenario_generator.generate_scenarios(
            eval_input_dict)

        # set x0 to configure process to evaluate
        if len(morphological_matrix) > 0:
            self.set_initial_inputs(morphological_matrix)

        morphological_matrix_df = DataFrame.from_dict(
            morphological_matrix, orient='index')
        morphological_matrix_df.insert(
            0, 'scenario_name', morphological_matrix_df.index.values)
        morphological_matrix_df = morphological_matrix_df.reset_index(
            drop=True)
        # all scenarios activated by default
        morphological_matrix_df.insert(
            0, 'selected_scenario', False)

        return morphological_matrix_df

    def set_initial_inputs(self, morphological_matrix):
        '''
            Get initial values (eval input values in first scenario) and set data in dm to configure eval process
        '''
        initial_inputs = list(morphological_matrix.values())[0]

        if initial_inputs != self.initial_inputs:

            for input_name in initial_inputs.keys():
                input_full_name = self.namespaced_eval_inputs[input_name]
                if self.dm.data_dict[self.dm.data_id_map[input_full_name]]['io_type'] == 'in':
                    self.ee.dm.set_data(
                        input_full_name, self.VALUE, initial_inputs[input_name])

            self.initial_inputs = deepcopy(initial_inputs)

    def fill_possible_values(self, disc):
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
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                namespaced_data = disc.get_var_full_name(
                    data_in_key, disc._data_in)
                namespace = namespaced_data.split(f'.{data_in_key}')[0]
                # Delete ee study name from namespace
                namespace = namespace.replace(f'{self.ee.study_name}.', '')
                # Particular case when namespace is equal to the study name
                if namespace == f'{self.ee.study_name}':
                    namespace = ''
                # store tuple (name, namespace)
                name_in.append((data_in_key, namespace))
        for data_out_key in disc._data_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            namespaced_data = disc.get_var_full_name(
                data_out_key, disc._data_out)
            namespace = namespaced_data.split(f'.{data_out_key}')[0]
            # Delete ee study name from namespace
            namespace = namespace.replace(f'{self.ee.study_name}.', '')
            # store tuple (name, namespace)
            name_out.append((data_out_key, namespace))

        return name_in, name_out

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
            Overloaded SoSEval method to store dataframes instead of lists
        '''
        # the sub-process is stored in self.sos_disciplines[0]
        name_in, name_out = self.fill_possible_values(
            self.sos_disciplines[0])
        name_in, name_out = self.find_possible_values(
            self.sos_disciplines[0], name_in, name_out)

        # A variable can be an input of several models get a unique list of
        # inputs
        name_in = list(set(name_in))

        # sort input names and output names in alphanetical order no matter
        # lower or higher cases
        name_in.sort(key=lambda v: (v[0].upper(), v[1].upper()))
        name_out.sort(key=lambda v: (v[0].upper(), v[1].upper()))

        # fill eval_inputs dataframe if name_in contains new inputs
        eval_input_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
        eval_inputs_from_name_in = DataFrame({'selected_input': False,
                                              'name': [name[0] for name in name_in],
                                              'namespace': [name[1] for name in name_in],
                                              'input_variable_name': ''})

        if eval_input_dm is None:

            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             self.VALUE, eval_inputs_from_name_in, check_value=False)

        elif not eval_input_dm.equals(eval_inputs_from_name_in):

            updated_eval_inputs = eval_inputs_from_name_in.merge(eval_input_dm, how='left', on=[
                'name', 'namespace'], suffixes=('', '_new'))

            # once we have merged with specic suffixes (new corresponds to activation_df)
            # we get the value of the new activation_df where there is not Nan values with a np.where
            # and delete the new column
            # Assure the dtype is the same because Nan can change the type of
            # the column
            for other_key in ['selected_input', 'input_variable_name']:
                updated_eval_inputs[other_key] = np.where(pd.notnull(
                    updated_eval_inputs[f'{other_key}_new']), updated_eval_inputs[f'{other_key}_new'], updated_eval_inputs[other_key])
                updated_eval_inputs.drop(
                    f'{other_key}_new', axis=1, inplace=True)
                updated_eval_inputs = updated_eval_inputs.astype(
                    {other_key: eval_inputs_from_name_in[other_key].dtype})

            self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                             self.VALUE, updated_eval_inputs, check_value=False)

        eval_output_dm = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
        eval_outputs_from_name_out = DataFrame({'selected_output': False,
                                                'name': [name[0] for name in name_out],
                                                'namespace': [name[1] for name in name_out],
                                                'output_variable_name': ''})

        # fill eval_outputs dataframe if name_out contains new outputs
        if eval_output_dm is None:

            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             self.VALUE, eval_outputs_from_name_out, check_value=False)

        elif not eval_output_dm.equals(eval_outputs_from_name_out):

            updated_eval_outputs = eval_outputs_from_name_out.merge(eval_output_dm, how='left', on=[
                'name', 'namespace'], suffixes=('', '_new'))
            # once we have merged with specific suffixes (new corresponds to activation_df)
            # we get the value of the new activation_df where there is not Nan values with a np.where
            # and delete the new column
            # Assure the dtype is the same because Nan can change the type of
            # the column
            for other_key in ['selected_output', 'output_variable_name']:
                updated_eval_outputs[other_key] = np.where(pd.notnull(
                    updated_eval_outputs[f'{other_key}_new']), updated_eval_outputs[f'{other_key}_new'], updated_eval_outputs[other_key])
                updated_eval_outputs.drop(
                    f'{other_key}_new', axis=1, inplace=True)
                updated_eval_outputs = updated_eval_outputs.astype(
                    {other_key: eval_outputs_from_name_out[other_key].dtype})
            self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                             self.VALUE, updated_eval_outputs, check_value=False)

    def launch_morphological_matrix_eval(self, activation_df):
        '''
            Launch eval_process run for each activated scenario inputs in morphological matrix
            The evaluator modifies input values and returns output values
        '''
        output_dict = {output: {}
                       for output in self.namespaced_eval_outputs.keys()}

        # filter activation df on selected scenario only
        selected_scenario_df = activation_df[activation_df['selected_scenario'] == True].reset_index(
            drop=True)
        total_selected_scenario = selected_scenario_df.shape[0]
        samples_to_evaluate = []
        scenario_name_matching = {}
        for index, row in selected_scenario_df.iterrows():
            samples_to_evaluate.append(row[list(self.eval_input_dict.keys())].values)
            scenario_name_matching["scenario_" + str(index + 1)] = row['scenario_name']

        evaluation_outputs = self.samples_evaluation(samples_to_evaluate, convert_to_array=False)

        for (scenario_name, evaluated_samples) in evaluation_outputs.items():
            for i, output in enumerate(self.namespaced_eval_outputs.keys()):
                output_dict[output][scenario_name_matching[scenario_name]] = evaluated_samples[1][i]
        return output_dict

    def run(self):
        '''
            Overloaded SoSEval method
        '''
        activation_df = self.get_sosdisc_inputs(self.ACTIVATION_DF)

        # set namespaced inputs and outputs for the evaluator
        self.eval_in_list = [value for key, value in self.namespaced_eval_inputs.items(
        ) if key in self.eval_input_dict.keys()]
        self.eval_out_list = list(self.namespaced_eval_outputs.values())
        # launch eval_process run for each activated scenario inputs
        output_dict = self.launch_morphological_matrix_eval(
            deepcopy(activation_df))
        # store output values
        self.store_sos_outputs_values(output_dict)
        # update status of eval process even if no run has been executed
        self._update_status_recursive(self.STATUS_DONE)

