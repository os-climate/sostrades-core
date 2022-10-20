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
import copy
import re

import platform
from tqdm import tqdm
import time

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.disciplines_wrappers.eval_wrapper import EvalWrapper

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import pandas as pd
from collections import ChainMap

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)

class DoeEval(EvalWrapper):
    '''
    Generic DOE evaluation class
    '''

    # Design space dataframe headers
    VARIABLES = "variable"
    VALUES = "value"
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    POSSIBLE_VALUES = 'possible_values'
    N_SAMPLES = "n_samples"
    DESIGN_SPACE = "design_space"

    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']
    # INPUT_MULTIPLIER_TYPE = []

    # TODO: refactor as DESC_IN = copy.deepcopy(EvalWrapp.DESC_IN).update(DoeWrapp.DESC_IN) or similar,
    #  when DoeWrapp and EvalWrapp are fixed
    DESC_IN = {'sampling_algo': {'type': 'string', 'structuring': True},
               'eval_inputs': {'type': 'dataframe',
                               'dataframe_descriptor': {'selected_input': ('bool', None, True),
                                                        'full_name': ('string', None, False)},
                               'dataframe_edition_locked': False,
                               'structuring': True,
                               'visibility': SoSWrapp.SHARED_VISIBILITY,
                               'namespace': 'ns_doe_eval'},
               'eval_outputs': {'type': 'dataframe',
                                'dataframe_descriptor': {'selected_output': ('bool', None, True),
                                                         'full_name': ('string', None, False)},
                                'dataframe_edition_locked': False,
                                'structuring': True, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                'namespace': 'ns_doe_eval'},
               'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
               'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},
               }

    DESC_OUT = {
        'samples_inputs_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                              'namespace': 'ns_doe_eval'}
    }

    def create_design_space(self):
        """
        create_design_space
        """
        dspace = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        design_space = None
        if dspace is not None:
            design_space = self.set_design_space()

        return design_space

    def set_design_space(self):
        """
        reads design space (set_design_space)
        """

        dspace_df = self.get_sosdisc_inputs(self.DESIGN_SPACE)
        # variables = self.attributes['eval_in_list']

        if 'full_name' in dspace_df:
            variables = dspace_df['full_name'].tolist()
            variables = [f'{self.attributes["study_name"]}.{eval}' for eval in variables]
        else:
            variables = self.attributes['eval_in_list']

        lower_bounds = dspace_df[self.LOWER_BOUND].tolist()
        upper_bounds = dspace_df[self.UPPER_BOUND].tolist()
        values = lower_bounds
        enable_variables = [True for invar in self.attributes['eval_in_list']]
        # This won't work for an array with a dimension greater than 2
        dspace_dict_updated = pd.DataFrame({self.VARIABLES: variables,
                                            self.VALUES: values,
                                            self.LOWER_BOUND: lower_bounds,
                                            self.UPPER_BOUND: upper_bounds,
                                            self.ENABLE_VARIABLE_BOOL: enable_variables,
                                            self.LIST_ACTIVATED_ELEM: self.attributes['activated_elems_dspace_df']})

        design_space = self.read_from_dataframe(dspace_dict_updated)

        return design_space

    def read_from_dataframe(self, df):
        """Parses a DataFrame to read the DesignSpace

        :param df : design space df
        :returns:  the design space
        """
        names = list(df[self.VARIABLES])
        values = list(df[self.VALUES])
        l_bounds = list(df[self.LOWER_BOUND])
        u_bounds = list(df[self.UPPER_BOUND])
        enabled_variable = list(df[self.ENABLE_VARIABLE_BOOL])
        list_activated_elem = list(df[self.LIST_ACTIVATED_ELEM])
        design_space = DesignSpace()
        for dv, val, lb, ub, l_activated, enable_var in zip(names, values, l_bounds, u_bounds, list_activated_elem,
                                                            enabled_variable):

            # check if variable is enabled to add it or not in the design var
            if enable_var:
                self.attributes['dict_desactivated_elem'][dv] = {}

                if [type(val), type(lb), type(ub)] == [str] * 3:
                    val = val
                    lb = lb
                    ub = ub
                name = dv
                if type(val) != list and type(val) != ndarray:
                    size = 1
                    var_type = ['float']
                    l_b = array([lb])
                    u_b = array([ub])
                    value = array([val])
                else:
                    # check if there is any False in l_activated
                    if not all(l_activated):
                        index_false = l_activated.index(False)
                        self.attributes['dict_desactivated_elem'][dv] = {
                            'value': val[index_false], 'position': index_false}

                        val = delete(val, index_false)
                        lb = delete(lb, index_false)
                        ub = delete(ub, index_false)

                    size = len(val)
                    var_type = ['float'] * size
                    l_b = array(lb)
                    u_b = array(ub)
                    value = array(val)
                design_space.add_variable(
                    name, size, var_type, l_b, u_b, value)
        return design_space

    def take_samples(self):
        algo_name = self.get_sosdisc_inputs(self.ALGO)
        if algo_name == 'CustomDOE':
            return super().take_samples()
        else:
            return self.generate_samples_from_doe_factory(algo_name)

    def generate_samples_from_doe_factory(self, algo_name):
        """Generating samples for the Doe using the Doe Factory
        """
        self.design_space = self.create_design_space()
        options = self.get_sosdisc_inputs(self.ALGO_OPTIONS)
        filled_options = {}
        for algo_option in options:
            if options[algo_option] != 'default':
                filled_options[algo_option] = options[algo_option]

        if self.N_SAMPLES not in options:
            LOGGER.warning("N_samples is not defined; pay attention you use fullfact algo "
                                "and that levels are well defined")

        LOGGER.info(filled_options)
        # TODO : logging from module ?

        filled_options[self.DIMENSION] = self.design_space.dimension
        filled_options[self._VARIABLES_NAMES] = self.design_space.variables_names
        filled_options[self._VARIABLES_SIZES] = self.design_space.variables_sizes
        # filled_options['n_processes'] = int(filled_options['n_processes'])
        filled_options['n_processes'] = self.get_sosdisc_inputs(
            'n_processes')
        filled_options['wait_time_between_samples'] = self.get_sosdisc_inputs(
            'wait_time_between_fork')
        algo = self.attributes['doe_factory'].create(algo_name)

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

    def prepare_samples(self):
        #TODO: necessary ?
        samples = []
        for sample in self.samples:
            sample_dict = self.design_space.array_to_dict(sample)
            # FIXME : are conversions needed here?
            # sample_dict = self._convert_array_into_new_type(sample_dict)
            ordered_sample = []
            for in_variable in self.attributes['eval_in_list']:
                ordered_sample.append(sample_dict[in_variable])
            samples.append(ordered_sample)
        return samples

    # def create_samples_from_custom_df(self):
    #     """Generation of the samples in case of a customed DOE
    #     """
    #     self.customed_samples = self.get_sosdisc_inputs('doe_df').copy()
    #     self.check_customed_samples()
    #     samples_custom = []
    #     for index, rows in self.customed_samples.iterrows():
    #         ordered_sample = []
    #         for col in rows:
    #             ordered_sample.append(col)
    #         samples_custom.append(ordered_sample)
    #     return samples_custom

    # def run(self):
    #     '''
    #         Overloaded SoSEval method
    #         The execution of the doe
    #     '''
    #     # upadte default inputs of children with dm values
    #     # TODO: Ask whether it is necessary to update default values.
    #     # self.update_default_inputs(self.attributes['sub_mdo_disciplines'])
    #
    #     dict_sample = {}
    #     dict_output = {}
    #
    #     # We first begin by sample generation
    #     self.samples = self.take_samples()
    #
    #     # Then add the reference scenario (initial point ) to the generated
    #     # samples
    #     self.samples.append(self.attributes['reference_scenario'])
    #     reference_scenario_id = len(self.samples)
    #     eval_in_with_multiplied_var = None
    #     # if self.INPUT_MULTIPLIER_TYPE != []:
    #     #     origin_vars_to_update_dict = self.create_origin_vars_to_update_dict()
    #     #     multipliers_samples = copy.deepcopy(self.samples)
    #     #     self.add_multiplied_var_to_samples(
    #     #         multipliers_samples, origin_vars_to_update_dict)
    #     #     eval_in_with_multiplied_var = self.attributes['eval_in_list'] + \
    #     #         list(origin_vars_to_update_dict.keys())
    #
    #     # evaluation of the samples through a call to samples_evaluation
    #     evaluation_outputs = self.samples_evaluation(
    #         self.samples, convert_to_array=False, completed_eval_in_list=eval_in_with_multiplied_var)
    #
    #     # we loop through the samples evaluated to build dictionnaries needed
    #     # for output generation
    #     reference_scenario = f'scenario_{reference_scenario_id}'
    #
    #     for (scenario_name, evaluated_samples) in evaluation_outputs.items():
    #
    #         # generation of the dictionnary of samples used
    #         dict_one_sample = {}
    #         current_sample = evaluated_samples[0]
    #         scenario_naming = scenario_name if scenario_name != reference_scenario else 'reference'
    #         for idx, f_name in enumerate(self.attributes['eval_in_list']):
    #             dict_one_sample[f_name] = current_sample[idx]
    #         dict_sample[scenario_naming] = dict_one_sample
    #
    #         # generation of the dictionnary of outputs
    #         dict_one_output = {}
    #         current_output = evaluated_samples[1]
    #         for idx, values in enumerate(current_output):
    #             dict_one_output[self.attributes['eval_out_list'][idx]] = values
    #         dict_output[scenario_naming] = dict_one_output
    #
    #     # construction of a dataframe of generated samples
    #     # columns are selected inputs
    #     columns = ['scenario']
    #     columns.extend(self.attributes['selected_inputs'])
    #     samples_all_row = []
    #     for (scenario, scenario_sample) in dict_sample.items():
    #         samples_row = [scenario]
    #         for generated_input in scenario_sample.values():
    #             samples_row.append(generated_input)
    #         samples_all_row.append(samples_row)
    #     samples_dataframe = pd.DataFrame(samples_all_row, columns=columns)
    #
    #     # construction of a dictionnary of dynamic outputs
    #     # The key is the output name and the value a dictionnary of results
    #     # with scenarii as keys
    #     global_dict_output = {key: {} for key in self.attributes['eval_out_list']}
    #     for (scenario, scenario_output) in dict_output.items():
    #         for full_name_out in scenario_output.keys():
    #             global_dict_output[full_name_out][scenario] = scenario_output[full_name_out]
    #
    #     #save data of last execution i.e. reference values #FIXME: do this better in refacto doe
    #     subprocess_ref_outputs = {key:self.attributes['sub_mdo_disciplines'][0].local_data[key]
    #                               for key in self.attributes['sub_mdo_disciplines'][0].output_grammar.get_data_names()}
    #     self.store_sos_outputs_values(subprocess_ref_outputs, full_name_keys=True)
    #     #save doeeval outputs
    #     self.store_sos_outputs_values(
    #         {'samples_inputs_df': samples_dataframe})
    #     for dynamic_output in self.attributes['eval_out_list']:
    #         self.store_sos_outputs_values({
    #             # f'{dynamic_output.split(".")[-1]}_dict':
    #             #     global_dict_output[dynamic_output]})
    #             f'{dynamic_output.split(self.attributes["study_name"] + ".",1)[1]}_dict':
    #                 global_dict_output[dynamic_output]})
