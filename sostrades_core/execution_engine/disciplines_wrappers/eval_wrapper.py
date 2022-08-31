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

class EvalWrapper(SoSWrapp):
    '''
    Generic Wrapper with SoSEval functions
    '''

    def samples_evaluation(self, samples, convert_to_array=True, completed_eval_in_list=None):
        # FIXME: should be moved to mother class EvalWrapper

        '''This function executes a parallel execution of the function sample_evaluation
        over a list a samples. Depending on the numerical parameter n_processes it loops
        on a sequential or parallel way over the list of samples to evaluate
        '''
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
                    self.STATUS_FAILED)


    def evaluation(self, x, scenario_name=None, convert_to_array=True, completed_eval_in_list=None):
        # FIXME: should be moved to mother class EvalWrapper

        '''
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays.
        '''
        # -- need to clear cash to avoir GEMS preventing execution when using disciplinary variables
        # self.attributes['sub_mdo_discipline'].clear_cache() # FIXME: cache management

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
        self.attributes['dm'].set_values_from_dict(values_dict)

        # # execute eval process stored in children
        # if len(self.proxy_disciplines) > 1:
        #     # the only child must be a coupling or a single discipline
        #     raise ProxyEvalException(
        #         f'ProxyEval discipline has more than one child discipline')
        # else:
        input_data_for_disc = self.get_input_data_for_gems(self.attributes['sub_mdo_discipline'])
        local_data = self.attributes['sub_mdo_discipline'].execute(input_data_for_disc)

        out_local_data = {key: value for key,
                          value in local_data.items() if key in self.attributes['eval_out_list']}

        # needed for gradient computation
        self.attributes['dm'].set_values_from_dict(local_data)

        if convert_to_array:
            out_local_data_converted = self._convert_new_type_into_array(
                out_local_data)
            out_values = np.concatenate(
                list(out_local_data_converted.values())).ravel()
        else:
            out_values = []
            # get back out_local_data is not enough because some variables
            # could be filtered for unsupported type for gemseo
            for y_id in self.attributes['eval_out_list']:
                y_val = self.attributes['dm'].get_value(y_id)
                out_values.append(y_val)

        return out_values

    def get_input_data_for_gems(self, disc):
        # FIXME : will need to be done differently without dm
        '''
        Get input_data for linearize sosdiscipline
        '''
        input_data = {}
        input_data_names = disc.input_grammar.get_data_names()
        if len(input_data_names) > 0:

            for data_name in input_data_names:
                input_data[data_name] = self.attributes['dm'].get_value(data_name)

        return input_data