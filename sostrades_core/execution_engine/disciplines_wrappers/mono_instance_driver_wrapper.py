'''
Copyright 2023 Capgemini

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
import platform
import time

import numpy as np
import pandas as pd
from gemseo.core.parallel_execution import ParallelExecution
from tqdm import tqdm

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import (
    DriverEvaluatorWrapper,
)
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import (
    SampleGeneratorWrapper,
)
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import (
    convert_new_type_into_array,
)


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
            scenario_nb = len(samples)
            for i in tqdm(range(scenario_nb), ncols=100, position=0):
                time.sleep(0.1)
                scenario_name = samples[i][SampleGeneratorWrapper.SCENARIO_NAME]
                self.logger.info(f'   {scenario_name} is running.')
                x = {key:value for key,value in samples[i].items() if key != SampleGeneratorWrapper.SCENARIO_NAME}
                
                evaluation_output[scenario_name] = x, self.evaluation(
                    x, scenario_name, convert_to_array)
            return evaluation_output

        if n_processes > 1:
            self.logger.info(
                "Running SOS EVAL in parallel on n_processes = %s", str(n_processes))

            # Create the parallel execution object. The function we want to
            # parallelize is the sample_evaluation
            def sample_evaluator(sample_to_evaluate):
                """Evaluate a sample
                """
                return self.evaluation(sample_to_evaluate, convert_to_array=False)

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
                scenario_name = samples[index][SampleGeneratorWrapper.SCENARIO_NAME]
                evaluation_output[scenario_name] = (samples[index], outputs)
                self.logger.info(
                    f'{scenario_name} has been run. computation progress: {int(((len(evaluation_output)) / len(samples)) * 100)}% done.')
                time.sleep(0.05)

            try:
                # execute all the scenarios (except the reference scenario)  in
                # parallel
                #remove the scenario_name key of each sample
                x = [{key:value for key,value in samples[i].items() if key != SampleGeneratorWrapper.SCENARIO_NAME} for i in range(len(samples))]
                
                parallel.execute(x[0:-1], exec_callback=store_callback)
                # execute the reference scenario in a sequential way so that
                # sostrades objects are updated
                scenario_name = samples[-1][SampleGeneratorWrapper.SCENARIO_NAME]
                evaluation_output[scenario_name] = samples[-1], self.evaluation(
                    x[-1], scenario_name, convert_to_array)
                self.proxy_disciplines[0]._update_status_recursive(
                    self.STATUS_DONE)
                dict_to_return = {}
                #return the outputs in the same order of the scenario lists
                for sample in samples:
                    scenario_name = sample[SampleGeneratorWrapper.SCENARIO_NAME]
                    if scenario_name in evaluation_output.keys():
                        dict_to_return[scenario_name] = evaluation_output[scenario_name]
                
                return dict_to_return

            except:
                self.proxy_disciplines[0]._update_status_recursive(
                    self.STATUS_FAILED)  # FIXME: This won't work

    def evaluation(self, x, scenario_name=None, convert_to_array=True):
        """
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays.
        """
        values_dict = x
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

    # def take_samples(self):
    #     """
    #     Generating samples for the Eval
    #     """
    #     self.custom_samples = self.get_sosdisc_inputs('samples_df').copy()
    #     # self.check_custom_samples()
    #     return self.custom_samples

    # TODO: transfert to sample generaor
    # def check_custom_samples(self):
    #     """ We that the columns of the dataframe are the same  that  the selected inputs
    #     We also check that they are of the same type
    #     """
    #     if not set(self.attributes['selected_inputs']).issubset(set(self.custom_samples.columns.to_list())):
    #         missing_eval_in_variables = set.union(set(self.attributes['selected_inputs']), set(
    #             self.custom_samples.columns.to_list())) - set(self.custom_samples.columns.to_list())
    #         msg = f'the columns of the custom samples dataframe must include all the the eval_in selected list of variables. Here the following selected eval_in variables {missing_eval_in_variables} are not in the provided sample.'
    #         # To do: provide also the list of missing eval_in variables:
    #         self.logger.error(msg)
    #         raise ValueError(msg)
    #     else:
    #         not_relevant_columns = set(
    #             self.custom_samples.columns.to_list()) - set(self.attributes['selected_inputs'])
    #         msg = f'the following columns {not_relevant_columns} of the custom samples dataframe are filtered because they are not in eval_in.'
    #         self.logger.warning(msg)
    #         # if len(not_relevant_columns) != 0:
    #         #     self.custom_samples.drop(
    #         #         not_relevant_columns, axis=1, inplace=True)
    #         # drop irrelevant + reorder
    #         self.custom_samples = self.custom_samples[self.attributes['selected_inputs']]

    def run(self):
        '''
            Overloaded SoSEval method
            The execution of the doe
        '''
        # upadte default inputs of children with dm values -> should not be necessary in EEV4
        # self.update_default_inputs(self.attributes['sub_mdo_disciplines'])

        dict_sample = {}
        dict_output = {}

        # We first begin by sample generation
        samples_df = self.get_sosdisc_inputs(SampleGeneratorWrapper.SAMPLES_DF)
        
        input_columns = [f"{self.attributes['driver_name']}.{col}" for col in samples_df.columns 
                         if col != SampleGeneratorWrapper.SCENARIO_NAME and col != SampleGeneratorWrapper.SELECTED_SCENARIO]
        input_columns_short_name = [col for col in samples_df.columns 
                         if col != SampleGeneratorWrapper.SCENARIO_NAME and col != SampleGeneratorWrapper.SELECTED_SCENARIO]
        # get reference scenario
        reference_values = self.get_sosdisc_inputs(input_columns, full_name_keys=True)
        if len(input_columns) == 1:
            reference_values = [reference_values]
        reference_scenario = {input_columns[i]: reference_values[i] for i in range(len(input_columns))}
        reference_scenario[SampleGeneratorWrapper.SCENARIO_NAME] = 'reference_scenario'

        # keep only selected scenario
        samples_df = samples_df[samples_df[SampleGeneratorWrapper.SELECTED_SCENARIO]== True]
        samples_df = samples_df.drop(SampleGeneratorWrapper.SELECTED_SCENARIO, axis='columns')

        #rename the columns with full names
        for key in input_columns_short_name:
            samples_df[f"{self.attributes['driver_name']}.{key}"] = samples_df[key].values
        samples_df = samples_df.drop(input_columns_short_name, axis='columns')

        #build samples dict
        self.samples = []
        scenario_names = set()
        scenario_nb = len(samples_df[SampleGeneratorWrapper.SCENARIO_NAME])
        for i in range(scenario_nb):
            sample = samples_df.iloc[i].to_dict()
            self.samples.append(sample)
            scenario_names.add(sample[SampleGeneratorWrapper.SCENARIO_NAME])
        # add reference_scenario if not added already by a SampleGenerator or user
        if 'reference_scenario' not in scenario_names:
            self.samples.append(reference_scenario)

        # evaluation of the samples through a call to samples_evaluation
        evaluation_outputs = self.samples_evaluation(
            self.samples, convert_to_array=False)

        # we loop through the samples evaluated to build dictionaries needed
        # for output generation
       
        for (scenario_name, evaluated_samples) in evaluation_outputs.items():

            # generation of the dictionary of outputs
            dict_one_output = {}
            current_output = evaluated_samples[1]
            for idx, values in enumerate(current_output):
                dict_one_output[self.attributes['eval_out_list'][idx]] = values
            dict_output[scenario_name] = dict_one_output

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
        
        output_columns = ['scenario_name']
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
        sample_input_df = pd.DataFrame( self.samples)

        # go again with short names into samples_inputs_df
        for key in input_columns_short_name:
            sample_input_df[key] = sample_input_df[f"{self.attributes['driver_name']}.{key}"].values
        sample_input_df = sample_input_df.drop(input_columns, axis='columns')

        self.store_sos_outputs_values({'samples_inputs_df':sample_input_df})

        self.store_sos_outputs_values(
            {'samples_outputs_df': samples_output_df})
        for dynamic_output, out_name in zip(self.attributes['eval_out_list'], self.attributes['eval_out_names']):
            self.store_sos_outputs_values({
                out_name: global_dict_output[dynamic_output]})
