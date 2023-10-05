'''
Copyright (c) 2023 Capgemini

All rights reserved

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or mother materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND OR ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType


'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import logging

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
import pandas as pd
from collections import ChainMap
from gemseo.core.parallel_execution import ParallelExecution


class MonoInstanceDriverWrapper(DriverEvaluatorWrapper):

    def samples_evaluation(self, samples, convert_to_array=True, completed_eval_in_list=None):
        """
        This function executes a parallel execution of the function sample_evaluation
        over a list a samples. Depending on the numerical parameter n_processes it loops
        on a sequential or parallel way over the list of samples to evaluate
        """

        self._init_input_data()

        evaluation_output = {}
        n_processes = self.get_sosdisc_inputs('n_processes')
        wait_time_between_samples = self.get_sosdisc_inputs(
            'wait_time_between_fork')
        if platform.system() == 'Windows' or n_processes == 1:
            if n_processes != 1:
                self.logger.warning(
                    "multiprocessing is not possible on Windows")
                n_processes = 1
            self.logger.info("running sos eval in sequential")

            for i in tqdm(range(len(samples)), ncols=100, position=0):
                time.sleep(0.1)
                self.logger.info(f'   Scenario_{str(i + 1)} is running.')
                x = samples[i]
                scenario_name = "scenario_" + str(i + 1)
                evaluation_output[scenario_name] = x, self.evaluation(
                    x, scenario_name, convert_to_array, completed_eval_in_list)
            return evaluation_output

        if n_processes > 1:
            self.logger.info(
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
                self.logger.info(
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
                    self.STATUS_FAILED)  # FIXME: This won't work

    def evaluation(self, x, scenario_name=None, convert_to_array=True, completed_eval_in_list=None):
        """
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays.
        """
        # -- need to clear cash to avoir GEMS preventing execution when using disciplinary variables
        # self.attributes['sub_mdo_discipline'].clear_cache() # TODO: cache
        # management?

        if completed_eval_in_list is None:
            eval_in = self.attributes['eval_in_list']
        else:
            eval_in = completed_eval_in_list
        # TODO: get a values_dict to arrive here for a + robust impl. less prone var. name errors and so ?
        values_dict = dict(zip(eval_in, x))

        local_data = self.attributes['sub_mdo_disciplines'][0].execute(
            self._get_input_data(values_dict))
        out_local_data = self._select_output_data(
            local_data, self.attributes['eval_out_list'])

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
            # EEV3 comment: get back out_local_data is not enough because some variables
            # could be filtered for unsupported type for gemseo TODO: is this case relevant in EEV4?
            for y_id in self.attributes['eval_out_list']:
                y_val = out_local_data[y_id]
                out_values.append(y_val)

        return out_values

    def take_samples(self):
        """
        Generating samples for the Eval
        """
        self.custom_samples = self.get_sosdisc_inputs('samples_df').copy()
        self.check_custom_samples()
        return self.custom_samples

    def check_custom_samples(self):
        """ We that the columns of the dataframe are the same  that  the selected inputs
        We also check that they are of the same type
        """
        if not set(self.attributes['selected_inputs']).issubset(set(self.custom_samples.columns.to_list())):
            missing_eval_in_variables = set.union(set(self.attributes['selected_inputs']), set(
                self.custom_samples.columns.to_list())) - set(self.custom_samples.columns.to_list())
            msg = f'the columns of the custom samples dataframe must include all the the eval_in selected list of variables. Here the following selected eval_in variables {missing_eval_in_variables} are not in the provided sample.'
            # To do: provide also the list of missing eval_in variables:
            self.logger.error(msg)
            raise ValueError(msg)
        else:
            not_relevant_columns = set(
                self.custom_samples.columns.to_list()) - set(self.attributes['selected_inputs'])
            msg = f'the following columns {not_relevant_columns} of the custom samples dataframe are filtered because they are not in eval_in.'
            self.logger.warning(msg)
            # if len(not_relevant_columns) != 0:
            #     self.custom_samples.drop(
            #         not_relevant_columns, axis=1, inplace=True)
            # drop irrelevant + reorder
            self.custom_samples = self.custom_samples[self.attributes['selected_inputs']]

    def mono_instance_run(self):
        '''
            Overloaded SoSEval method
            The execution of the doe
        '''
        # upadte default inputs of children with dm values -> should not be necessary in EEV4
        # self.update_default_inputs(self.attributes['sub_mdo_disciplines'])

        dict_sample = {}
        dict_output = {}

        # We first begin by sample generation
        self.samples = self.take_samples()

        # Before, for User-defined samples, Eval received a dataframe and transformed it into list in take_samples
        # above. For DoE sampling generation Eval received a list.
        # Now, for User-defined samples and DoE sampling generation, Eval receives a dataframe in both cases and
        # transforms it in list.
        self.samples = self.samples.values.tolist()

        # Then add the reference scenario (initial point ) to the input samples
        self.samples.append([self.attributes['reference_scenario'][var_to_eval]
                             for var_to_eval in self.attributes['eval_in_list']])

        reference_scenario_id = len(self.samples)
        eval_in_with_multiplied_var = None
        # if self.INPUT_MULTIPLIER_TYPE != []:
        #     origin_vars_to_update_dict = self.create_origin_vars_to_update_dict()
        #     multipliers_samples = copy.deepcopy(self.samples)
        #     self.add_multiplied_var_to_samples(
        #         multipliers_samples, origin_vars_to_update_dict)
        #     eval_in_with_multiplied_var = self.attributes['eval_in_list'] + \
        #         list(origin_vars_to_update_dict.keys())

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

        samples_all_row = []
        out_samples_all_row = []
        for (scenario, scenario_sample) in dict_sample.items():
            samples_row = [scenario]
            out_samples_row = [scenario]
            for generated_input in scenario_sample.values():
                samples_row.append(generated_input)
            for generated_output in dict_output[scenario].values():
                out_samples_row.append(generated_output)
            samples_all_row.append(samples_row)
            out_samples_all_row.append(out_samples_row)
        input_columns = ['scenario']
        input_columns.extend(self.attributes['selected_inputs'])
        samples_input_df = pd.DataFrame(samples_all_row, columns=input_columns)
        output_columns = ['scenario']
        output_columns.extend(self.attributes['selected_outputs'])
        samples_output_df = pd.DataFrame(out_samples_all_row, columns=output_columns)
        # construction of a dictionary of dynamic outputs
        # The key is the output name and the value a dictionary of results
        # with scenarii as keys
        global_dict_output = {key: {}
                              for key in self.attributes['eval_out_list']}
        for (scenario, scenario_output) in dict_output.items():
            for full_name_out in scenario_output.keys():
                global_dict_output[full_name_out][scenario] = scenario_output[full_name_out]

        # save data of last execution i.e. reference values # TODO: do this  better in refacto doe
        subprocess_ref_outputs = {key: self.attributes['sub_mdo_disciplines'][0].local_data[key]
                                  for key in self.attributes['sub_mdo_disciplines'][0].output_grammar.get_data_names()}
        self.store_sos_outputs_values(
            subprocess_ref_outputs, full_name_keys=True)
        # save doeeval outputs
        self.store_sos_outputs_values(
            {'samples_inputs_df': samples_input_df})

        self.store_sos_outputs_values(
            {'samples_outputs_df': samples_output_df})
        for dynamic_output, out_name in zip(self.attributes['eval_out_list'], self.attributes['eval_out_names']):
            self.store_sos_outputs_values({
                out_name: global_dict_output[dynamic_output]})
