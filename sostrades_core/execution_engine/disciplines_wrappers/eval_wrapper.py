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
import numpy as np

import platform
from tqdm import tqdm
import time

from sostrades_core.tools.base_functions.compute_len import compute_len
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_new_type_into_array, convert_array_into_new_type

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.disciplines_wrappers.abstract_eval_wrapper import AbstractEvalWrapper
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import pandas as pd
from collections import ChainMap
from gemseo.core.parallel_execution import ParallelExecution

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)

class EvalWrapper(AbstractEvalWrapper):
    '''
    Generic Wrapper with SoSEval functions
    '''

    _maturity = 'Fake'
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'
    DESC_IN = {
        'eval_inputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'eval_outputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
        'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0}
    }

    DESC_OUT = {
        'samples_inputs_df': {'type': 'dataframe', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                              'namespace': 'ns_eval'}
    }

    INPUT_MULTIPLIER_TYPE = []

    def __init__(self, sos_name):

        super().__init__(sos_name)
        self.samples = None
        self.input_data_for_disc = None

    def _init_input_data(self):
        #TODO: deepcopy option? [discuss]
        self.input_data_for_disc = self.get_input_data_for_gems(self.attributes['sub_mdo_disciplines'][0])
        # self.input_data_for_disc = [self.get_input_data_for_gems(disc) for disc in self.attributes['sub_mdo_disciplines']]

    def _get_input_data(self, delta_dict):
        #TODO: deepcopy option? [discuss]
        self.input_data_for_disc.update(delta_dict)
        return self.input_data_for_disc

    def _select_output_data(self, raw_data, eval_out_data_names):
        output_data_dict = {key: value for key,value in raw_data.items()
                          if key in eval_out_data_names}
        return output_data_dict

    def samples_evaluation(self, samples, convert_to_array=True, completed_eval_in_list=None):

        '''This function executes a parallel execution of the function sample_evaluation
        over a list a samples. Depending on the numerical parameter n_processes it loops
        on a sequential or parallel way over the list of samples to evaluate
        '''

        self._init_input_data()

        evaluation_output = {}
        n_processes = self.get_sosdisc_inputs('n_processes')
        wait_time_between_samples = self.get_sosdisc_inputs(
            'wait_time_between_fork')
        if platform.system() == 'Windows' or n_processes == 1:
            if n_processes != 1:
                LOGGER.warning(
                    "multiprocessing is not possible on Windows")
                n_processes = 1
            LOGGER.info("running sos eval in sequential")

            for i in tqdm(range(len(samples)), ncols=100, position=0):
                time.sleep(0.1)
                LOGGER.info(f'   Scenario_{str(i + 1)} is running.')
                x = samples[i]
                scenario_name = "scenario_" + str(i + 1)
                evaluation_output[scenario_name] = x, self.evaluation(
                    x, scenario_name, convert_to_array, completed_eval_in_list)
            return evaluation_output

        if n_processes > 1:
            LOGGER.info(
                "Running SOS EVAL in parallel on n_processes = %s", str(n_processes))

            # Create the parallel execution object. The function we want to
            # parallelize is the sample_evaluation
            def sample_evaluator(sample_to_evaluate):
                """Evaluate a sample
                """
                return self.evaluation(sample_to_evaluate, convert_to_array=False, completed_eval_in_list=completed_eval_in_list)

            parallel = ParallelExecution(sample_evaluator, n_processes=n_processes,
                                         wait_time_between_fork=wait_time_between_samples)

            # Define a callback function to store the samples on the fly
            # during the parallel execution
            def store_callback(
                    index,  # type: int
                    outputs,  # type: DOELibraryOutputType
            ):  # type: (...) -> None
                """Store the outputs in dedicated dictionnary:
                - Here the dictionnary key is the sample evaluated and the value is the evaluation output
                Args:
                    index: The sample index.
                    outputs: The outputs of the parallel execution.
                """
                scenario_name = "scenario_" + str(index + 1)
                evaluation_output[scenario_name] = (samples[index], outputs)
                LOGGER.info(
                    f'{scenario_name} has been run. computation progress: {int(((len(evaluation_output)) / len(samples)) * 100)}% done.')
                time.sleep(0.05)

            try:
                # execute all the scenarios (except the reference scenario)  in
                # parallel
                parallel.execute(samples[0:-1], exec_callback=store_callback)
                # execute the reference scenario in a sequential way so that
                # sostrades objects are updated
                scenario_name = "scenario_" + str(len(samples))
                evaluation_output[scenario_name] = samples[-1], self.evaluation(
                    samples[-1], scenario_name, convert_to_array, completed_eval_in_list)
                self.proxy_disciplines[0]._update_status_recursive(
                    self.STATUS_DONE)
                dict_to_return = {}
                for (scenario_name, sample_value) in sorted(evaluation_output.items(),
                                                            key=lambda scenario: int(
                                                                scenario[0].split("scenario_")[1])):
                    dict_to_return[scenario_name] = sample_value
                return dict_to_return

            except:
                self.proxy_disciplines[0]._update_status_recursive(
                    self.STATUS_FAILED) # FIXME: This won't work


    def evaluation(self, x, scenario_name=None, convert_to_array=True, completed_eval_in_list=None):
        '''
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays.
        '''
        # -- need to clear cash to avoir GEMS preventing execution when using disciplinary variables
        # self.attributes['sub_mdo_discipline'].clear_cache() # FIXME: cache management?
        # TODO: x should already be a dictionary.
        values_dict = {}
        eval_in = self.attributes['eval_in_list']
        if completed_eval_in_list is not None:
            eval_in = completed_eval_in_list
        for i, x_id in enumerate(eval_in):
            values_dict[x_id] = x[i]

        # Because we use set_data instead of load_data_from_inputs_dict it isn't possible
        # to run  soseval on a structuring variable. Therefore structuring variables are
        # excluded from eval possible values
        # set values_dict in the data manager to execute the sub process
        # self.attributes['dm'].set_values_from_dict(values_dict)

        # # execute eval process stored in children
        # if len(self.proxy_disciplines) > 1:
        #     # the only child must be a coupling or a single discipline
        #     raise ProxyEvalException(
        #         f'ProxyEval discipline has more than one child discipline')
        # else:
        # input_data_for_disc = self.get_input_data_for_gems(self.attributes['sub_mdo_disciplines'])
        # local_data = self.attributes['sub_mdo_discipline'].execute(input_data_for_disc)
        local_data = self.attributes['sub_mdo_disciplines'][0].execute(self._get_input_data(values_dict))
        out_local_data = self._select_output_data(local_data, self.attributes['eval_out_list'])

        # needed for gradient computation
        # TODO: manage data flow for gradient computation ?
        # self.attributes['dm'].set_values_from_dict(local_data)

        if convert_to_array:
            out_local_data_converted = convert_new_type_into_array(
                out_local_data, self.attributes['reduced_dm'])
            out_values = np.concatenate(
                list(out_local_data_converted.values())).ravel()
        else:
            # out_values = list(out_local_data.values())
            out_values = []
            # get back out_local_data is not enough because some variables
            # could be filtered for unsupported type for gemseo  TODO: is this case relevant??
            for y_id in self.attributes['eval_out_list']:
                y_val = out_local_data[y_id]
                out_values.append(y_val)

        return out_values

    def get_input_data_for_gems(self, disc):
        '''
        Get input_data for linearize sosdiscipline
        '''
        input_data = {}
        input_data_names = disc.input_grammar.get_data_names()
        if len(input_data_names) > 0:
            input_data = self.get_sosdisc_inputs(keys=input_data_names, in_dict=True, full_name_keys=True)
            # for data_name in input_data_names:
            #     input_data[data_name] = self.attributes['dm'].get_value(data_name)

        return input_data

    def apply_muliplier(self, multiplier_name, multiplier_value, var_to_update):
        col_index = multiplier_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[1]
        if any(char.isdigit() for char in col_index):
            col_index = re.findall(r'\d+', col_index)[0]
            cols_list = var_to_update.columns.to_list()
            key = cols_list[int(col_index)]
            var_to_update[key] = multiplier_value * var_to_update[key]
        else:
            if isinstance(var_to_update, dict):
                float_cols_ids_list = [dict_keys for dict_keys in var_to_update if isinstance(
                    var_to_update[dict_keys], float)]
            elif isinstance(var_to_update, pd.DataFrame):
                float_cols_ids_list = [
                    df_keys for df_keys in var_to_update if var_to_update[df_keys].dtype == 'float']
            for key in float_cols_ids_list:
                var_to_update[key] = multiplier_value * var_to_update[key]
        return var_to_update

    # def convert_output_results_toarray(self):
    #     #unused???
    #     '''
    #     COnvert toutput results into array in order to apply FDGradient on it for example
    #     '''
    #     out_values = []
    #     self.eval_out_type = []
    #     self.eval_out_list_size = []
    #     for y_id in self.attributes['eval_out_list']:
    #
    #         y_val = self.dm.get_value(y_id)
    #         self.eval_out_type.append(type(y_val))
    #         # Need a flatten list for the eval computation if val is dict
    #         if type(y_val) in [dict, DataFrame]:
    #             val_dict = {y_id: y_val}
    #             dict_flatten = convert_new_type_into_array(
    #                 val_dict, self.attributes['dm'])
    #             y_val = dict_flatten[y_id].tolist()
    #
    #         else:
    #             y_val = [y_val]
    #         self.eval_out_list_size.append(len(y_val))
    #         out_values.extend(y_val)
    #
    #     return np.array(out_values)

    # def reconstruct_output_results(self, outputs_eval):
    #     '''
    #     Reconstruct the metadata saved earlier to get same object in output
    #     instead of a flatten list
    #     '''
    #     #TODO: check eval_process_disc when implementing sos_gradients.
    #     outeval_final_dict = {}
    #     for j, key_in in enumerate(self.attributes['eval_in_list']):
    #         outeval_dict = {}
    #         old_size = 0
    #         for i, key in enumerate(self.attributes['eval_out_list']):
    #             eval_out_size = compute_len(
    #                 self.eval_process_disc.local_data[key])
    #             output_eval_key = outputs_eval[old_size:old_size +
    #                                            eval_out_size]
    #             old_size = eval_out_size
    #             type_sos = self.attributes['dm'].get_data(key, 'type')
    #             if type_sos in ['dict', 'dataframe']:
    #                 outeval_dict[key] = np.array([
    #                     sublist[j] for sublist in output_eval_key])
    #             else:
    #                 outeval_dict[key] = output_eval_key[0][j]
    #
    #         outeval_dict = convert_array_into_new_type(outeval_dict,self.attributes['dm'])
    #         outeval_base_dict = {f'{key_out} vs {key_in}': value for key_out, value in zip(
    #             self.attributes['eval_out_list'], outeval_dict.values())}
    #         outeval_final_dict.update(outeval_base_dict)
    #
    #     return outeval_final_dict

    # def update_default_inputs(self, disc):
    #     '''
    #     Update default inputs of disc with dm values
    #     '''
    #     input_data = {}
    #     input_data_names = disc.get_input_data_names()
    #     for data_name in input_data_names:
    #         val = self.attributes['dm'].get_value(data_name)
    #         # val = self.get_value(data_name)
    #         if val is not None:
    #             input_data[data_name] = val
    #
    #     # store mdo_chain default inputs
    #     if disc.is_sos_coupling:
    #         disc.mdo_chain.default_inputs.update(input_data)
    #     disc.default_inputs.update(input_data)
    #
    #     # recursive update default inputs of children
    #     for sub_disc in disc.sos_disciplines:
    #         self.update_default_inputs(sub_disc)

    def create_origin_vars_to_update_dict(self):
        origin_vars_to_update_dict = {}
        for select_in in self.attributes['eval_in_list']:
            if self.MULTIPLIER_PARTICULE in select_in:
                var_origin_f_name = self.get_names_from_multiplier(select_in)[
                    0]
                if var_origin_f_name not in origin_vars_to_update_dict:
                    # origin_vars_to_update_dict[var_origin_f_name] = copy.deepcopy(
                    #     self.attributes['dm'].get_data(var_origin_f_name)['value'])
                    origin_vars_to_update_dict[var_origin_f_name] = copy.deepcopy(
                        self.get_sosdisc_inputs(var_origin_f_name, full_name_keys=True))
        return origin_vars_to_update_dict

    def add_multiplied_var_to_samples(self, multipliers_samples, origin_vars_to_update_dict):
        for sample_i in range(len(multipliers_samples)):
            x = multipliers_samples[sample_i]
            vars_to_update_dict = {}
            for multiplier_i, x_id in enumerate(self.attributes['eval_in_list']):
                # for grid search multipliers inputs
                var_name = x_id.split(self.ee.study_name + '.', 1)[-1]
                if self.MULTIPLIER_PARTICULE in var_name:
                    var_origin_f_name = '.'.join(
                        [self.attributes["study_name"], self.get_names_from_multiplier(var_name)[0]])
                    if var_origin_f_name in vars_to_update_dict:
                        var_to_update = vars_to_update_dict[var_origin_f_name]
                    else:
                        var_to_update = copy.deepcopy(
                            origin_vars_to_update_dict[var_origin_f_name])
                    vars_to_update_dict[var_origin_f_name] = self.apply_muliplier(
                        multiplier_name=var_name, multiplier_value=x[multiplier_i] / 100.0, var_to_update=var_to_update)
            for multiplied_var in vars_to_update_dict:
                self.samples[sample_i].append(
                    vars_to_update_dict[multiplied_var])

    def clean_var_name(self, var_name):
        return re.sub(r"[^a-zA-Z0-9]", "_", var_name)

    def get_names_from_multiplier(self, var_name):
        column_name = None
        var_origin_name = var_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[0]
        if '@' in var_name:
            column_name = var_name.split(self.MULTIPLIER_PARTICULE)[
                0].split('@')[1]

        return [var_origin_name, column_name]

    def run(self):
        '''
            Overloaded SoSEval method
            The execution of the doe
        '''
        # upadte default inputs of children with dm values
        # TODO: no longer necessary to update default values, right?
        # self.update_default_inputs(self.attributes['sub_mdo_disciplines'])

        dict_sample = {}
        dict_output = {}

        # We first begin by sample generation
        self.samples = self.take_samples()

        # Then add the reference scenario (initial point ) to the generated
        # samples
        self.samples.append(self.attributes['reference_scenario'])
        reference_scenario_id = len(self.samples)
        eval_in_with_multiplied_var = None
        if self.INPUT_MULTIPLIER_TYPE != []:
            origin_vars_to_update_dict = self.create_origin_vars_to_update_dict()
            multipliers_samples = copy.deepcopy(self.samples)
            self.add_multiplied_var_to_samples(
                multipliers_samples, origin_vars_to_update_dict)
            eval_in_with_multiplied_var = self.attributes['eval_in_list'] + \
                                          list(origin_vars_to_update_dict.keys())

        # evaluation of the samples through a call to samples_evaluation
        evaluation_outputs = self.samples_evaluation(
            self.samples, convert_to_array=False, completed_eval_in_list=eval_in_with_multiplied_var)

        # we loop through the samples evaluated to build dictionaries needed
        # for output generation
        reference_scenario = f'scenario_{reference_scenario_id}'

        for (scenario_name, evaluated_samples) in evaluation_outputs.items():

            # generation of the dictionary of samples used
            dict_one_sample = {}
            current_sample = evaluated_samples[0]
            scenario_naming = scenario_name if scenario_name != reference_scenario else 'reference'
            for idx, f_name in enumerate(self.attributes['eval_in_list']):
                dict_one_sample[f_name] = current_sample[idx]
            dict_sample[scenario_naming] = dict_one_sample

            # generation of the dictionary of outputs
            dict_one_output = {}
            current_output = evaluated_samples[1]
            for idx, values in enumerate(current_output):
                dict_one_output[self.attributes['eval_out_list'][idx]] = values
            dict_output[scenario_naming] = dict_one_output

        # construction of a dataframe of generated samples
        # columns are selected inputs
        columns = ['scenario']
        columns.extend(self.attributes['selected_inputs'])
        samples_all_row = []
        for (scenario, scenario_sample) in dict_sample.items():
            samples_row = [scenario]
            for generated_input in scenario_sample.values():
                samples_row.append(generated_input)
            samples_all_row.append(samples_row)
        samples_dataframe = pd.DataFrame(samples_all_row, columns=columns)

        # construction of a dictionary of dynamic outputs
        # The key is the output name and the value a dictionary of results
        # with scenarii as keys
        global_dict_output = {key: {} for key in self.attributes['eval_out_list']}
        for (scenario, scenario_output) in dict_output.items():
            for full_name_out in scenario_output.keys():
                global_dict_output[full_name_out][scenario] = scenario_output[full_name_out]

        # save data of last execution i.e. reference values #FIXME: do this better in refacto doe
        subprocess_ref_outputs = {key: self.attributes['sub_mdo_disciplines'][0].local_data[key]
                                  for key in self.attributes['sub_mdo_disciplines'][0].output_grammar.get_data_names()}
        self.store_sos_outputs_values(subprocess_ref_outputs, full_name_keys=True)
        # save doeeval outputs
        self.store_sos_outputs_values(
            {'samples_inputs_df': samples_dataframe})

        for dynamic_output in self.attributes['eval_out_list']:

            self.store_sos_outputs_values({
                f'{dynamic_output.split(self.attributes["study_name"] + ".", 1)[1]}_dict':
                    global_dict_output[dynamic_output]})

            #TODO: dirty namespacing

            # self.store_sos_outputs_values({
            #     f'{dynamic_output.split(".")[-1]}_dict':
            #         global_dict_output[dynamic_output]})

    def take_samples(self):
        """Generating samples for the Eval
        """
        self.customed_samples = self.get_sosdisc_inputs('custom_samples_df').copy()
        self.check_customed_samples()
        samples_custom = []
        for index, rows in self.customed_samples.iterrows():
            ordered_sample = []
            for col in rows:
                ordered_sample.append(col)
            samples_custom.append(ordered_sample)
        return samples_custom

    def check_customed_samples(self):
        """ We that the columns of the dataframe are the same  that  the selected inputs
        We also check that they are of the same type
        """
        # TODO: these set operations might be at the root of the "ordering" issue
        if not set(self.attributes['selected_inputs']).issubset(set(self.customed_samples.columns.to_list())):
            missing_eval_in_variables = set.union(set(self.attributes['selected_inputs']), set(
                self.customed_samples.columns.to_list())) - set(self.customed_samples.columns.to_list())
            msg = f'the columns of the custom samples dataframe must include all the the eval_in selected list of variables. Here the following selected eval_in variables {missing_eval_in_variables} are not in the provided sample.'
            # To do: provide also the list of missing eval_in variables:
            LOGGER.error(msg)
            raise ValueError(msg)
        else:
            not_relevant_columns = set(
                self.customed_samples.columns.to_list()) - set(self.attributes['selected_inputs'])
            msg = f'the following columns {not_relevant_columns} of the custom samples dataframe are filtered because they are not in eval_in.'
            LOGGER.warning(msg)
            if len(not_relevant_columns) != 0:
                self.customed_samples.drop(
                    not_relevant_columns, axis=1, inplace=True)
            self.attributes['selected_inputs'].sort()
            self.customed_samples = self.customed_samples[self.attributes['selected_inputs']]
