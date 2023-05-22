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

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType


'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import pandas as pd
from collections import ChainMap
from gemseo.core.parallel_execution import ParallelExecution

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)


class DriverEvaluatorWrapper(SoSWrapp):
    """
    DriverEvaluatorWrapper is a type of SoSWrapp that can evaluate one or several subprocesses either with their
    reference inputs or by applying modifications to some of the subprocess variables. It is assumed to have references
    to the GEMSEO objects at the root of each of the subprocesses, stored in self.attributes['sub_mdo_disciplines'].

    1) Structure of Desc_in/Desc_out:
        |_ DESC_IN
            |_ BUILDER_MODE (structuring)
            |_ USECASE_DATA (structuring)                
            |_ SUB_PROCESS_INPUTS (structuring) #TODO V1
    2) Description of DESC parameters:
        |_ DESC_IN
            |_ BUILDER_MODE
            |_ USECASE_DATA
            |_ SUB_PROCESS_INPUTS:               All inputs for driver builder in the form of ProcessBuilderParameterType type
                                                    PROCESS_REPOSITORY:   folder root of the sub processes to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    PROCESS_NAME:         selected process name (in repository) to be nested inside the DoE.
                                                                          If 'None' then it uses the sos_processes python for doe creation.
                                                    USECASE_INFO:         either empty or an available data source of the sub_process
                                                    USECASE_NAME:         children of USECASE_INFO that contains data source name (can be empty)
                                                    USECASE_TYPE:         children of USECASE_INFO that contains data source type (can be empty)
                                                    USECASE_IDENTIFIER:   children of USECASE_INFO that contains data source identifier (can be empty)
                                                    USECASE_DATA:         anonymized dictionary of usecase inputs to be nested in context
                                                                          it is a temporary input: it will be put to None as soon as
                                                                          its content is 'loaded' in the dm. We will have it has editable
                                                It is in dict type (specific 'proc_builder_modale' type to have a specific GUI widget) 

    """

    _maturity = 'Fake'

    BUILDER_MODE = 'builder_mode'
    MONO_INSTANCE = 'mono_instance'
    MULTI_INSTANCE = 'multi_instance'
    REGULAR_BUILD = 'regular_build'
    BUILDER_MODE_POSSIBLE_VALUES = [MULTI_INSTANCE, MONO_INSTANCE]
    SUB_PROCESS_INPUTS = 'sub_process_inputs'
    USECASE_DATA = 'usecase_data'
    GATHER_DEFAULT_SUFFIX = '_dict'


    default_process_builder_parameter_type = ProcessBuilderParameterType(
        None, None, 'Empty')

    DESC_IN = {
        BUILDER_MODE: {SoSWrapp.TYPE: 'string',
                       # SoSWrapp.DEFAULT: MULTI_INSTANCE,
                       SoSWrapp.POSSIBLE_VALUES: BUILDER_MODE_POSSIBLE_VALUES,
                       SoSWrapp.STRUCTURING: True}}

    with_modal = True
    if with_modal:
        DESC_IN.update({SUB_PROCESS_INPUTS: {'type': ProxyDiscipline.PROC_BUILDER_MODAL,
                                             'structuring': True,
                                             'default': default_process_builder_parameter_type.to_data_manager_dict(),
                                             'user_level': 1,
                                             'optional': False}})
    else:
        DESC_IN.update({USECASE_DATA: {'type': 'dict',
                                       'structuring': True,
                                       'default': {},
                                       'user_level': 1,
                                       'optional': False}})

    def __init__(self, sos_name):
        """
        Constructor.

        Arguments:
            sos_name (string): name of the discipline
        """
        super().__init__(sos_name)
        self.custom_samples = None  # input samples dataframe
        # samples to evaluate as list[list[Any]] or ndarray
        self.samples = None
        self.n_subprocs = 0
        self.input_data_for_disc = None
        self.subprocesses_to_eval = None

    def _init_input_data(self):
        """
        Initialise the attribute that stores the input data of every subprocess for this run.
        """
        self.n_subprocs = len(self.attributes['sub_mdo_disciplines'])
        self.input_data_for_disc = [{}] * self.n_subprocs
        # TODO: deepcopy option? [discuss]
        for i_subprocess in self.subprocesses_to_eval or range(self.n_subprocs):
            self.input_data_for_disc[i_subprocess] = self.get_input_data_for_gems(
                self.attributes['sub_mdo_disciplines'][i_subprocess])

    def _get_input_data(self, var_delta_dict, i_subprocess=0):
        """
        Updates the input data to execute a given subprocess by applying changes to the variables whose full names and
        new values are specified in the var_delta_dict (for all other variables use reference subprocess values).

        Arguments:
            var_delta_dict (dict): keys are variable full names and values are variable non-reference values to be applied
                               at subprocess execution
            i_subprocess (int): index of the subprocess to execute, i.e. the subprocess that provides reference inputs
                                and to whom var_delta_dict is applied

        Returns:
            self.input_data_for_disc[i_subprocess] (dict): the input data updated with new values for certain variables
        """
        # TODO: deepcopy option? [discuss]
        self.input_data_for_disc[i_subprocess].update(var_delta_dict)
        return self.input_data_for_disc[i_subprocess]

    def _select_output_data(self, raw_data, eval_out_data_names):
        """
        Filters from raw_data the items that are in eval_out_data_names.

        Arguments:
            raw_data (dict): dictionary of variable full names and values such as the local_data of a subprocess
            eval_out_data_names (list[string]): full names of the variables to keep

        Returns:
             output_data_dict (dict): filtered dictionary
        """
        output_data_dict = {key: value for key, value in raw_data.items()
                            if key in eval_out_data_names}
        return output_data_dict

    def get_input_data_for_gems(self, disc):
        """
        Get reference inputs for a subprocess by querying for the data names in its input grammar.

        Arguments:
            disc (MDODiscipline): discipline at the root of the subprocess.

        Returns:
            input_data (dict): full names and reference values for the subprocess inputs
        """
        input_data = {}
        input_data_names = disc.input_grammar.get_data_names()
        if len(input_data_names) > 0:
            input_data = self.get_sosdisc_inputs(
                keys=input_data_names, in_dict=True, full_name_keys=True)
        return input_data

    def subprocess_evaluation(self, var_delta_dict, i_subprocess, convert_to_array=True):
        """
        Evaluate a subprocess.

        Arguments:
            var_delta_dict (dict): keys are variable full names and values are variable non-reference values to be
                                   applied for the execution of this subprocess. Providing an empty dict will evaluate
                                   reference values unless input_data_for_disc[i_subprocess] attribute has been modified
                                   beforehand.
            i_subprocess (int): index of the subprocess to execute, i.e. the subprocess that provides reference inputs
                                and to whom delta_dict is applied
        """
        local_data = self.attributes['sub_mdo_disciplines'][i_subprocess]\
                         .execute(self._get_input_data(var_delta_dict, i_subprocess))
        # out_local_data = self._select_output_data(local_data, self.attributes['eval_out_list'][i_subprocess])
        # if convert_to_array:
        #     out_local_data_converted = convert_new_type_into_array(
        #         out_local_data, self.attributes['reduced_dm'])
        #     out_values = np.concatenate(
        #         list(out_local_data_converted.values())).ravel()
        # else:
        #     out_values = []
        #     # get back out_local_data is not enough because some variables
        #     # could be filtered for unsupported type for gemseo  TODO: is this case relevant??
        #     for y_id in self.attributes['eval_out_list'][i_subprocess]:
        #         y_val = out_local_data[y_id]
        #         out_values.append(y_val)
        # return out_values
        # return local_data

    def run(self):
        """
        Run overload
        """
        builder_mode = self.get_sosdisc_inputs(self.BUILDER_MODE)
        if builder_mode == self.MONO_INSTANCE:
            self.mono_instance_run()
        elif builder_mode == self.MULTI_INSTANCE:
            self.multi_instance_run()
        else:
            raise NotImplementedError()

    ####################################
    ####################################
    ###### MULTI INSTANCE PROCESS ######
    ####################################
    ####################################

    def multi_instance_run(self):
        """
        Run in the multi instance case.
        """
        # very simple ms only
        self._init_input_data()
        subpr_to_eval = self.subprocesses_to_eval or range(self.n_subprocs)
        gather_names = self.attributes['gather_names']
        gather_out_keys = self.attributes['gather_out_keys']
        # TODO: if an output does not exist in a scenario, it will not be in the dict. Add entry {sc_name: None} ?
        gather_output_dict = {key: {} for key in gather_out_keys}
        # gather_output_dict = {key: {sc: None for sc in self.attributes['scenario_names']} for key in gather_out_keys}

        for i_subprocess in subpr_to_eval:
            self.subprocess_evaluation({}, i_subprocess)
            # save data of execution i.e. scenario values
            subprocess_outputs = {key: self.attributes['sub_mdo_disciplines'][i_subprocess].local_data[key]
                                  for key in self.attributes['sub_mdo_disciplines'][i_subprocess].output_grammar.get_data_names()}
            self.store_sos_outputs_values(
                subprocess_outputs, full_name_keys=True)

            # the keys of gather_names correspond to the full names of the vars to gather
            gathered_in_subprocess = self._select_output_data(subprocess_outputs, gather_names)
            for _gathered_var_name, _gathered_var_value in gathered_in_subprocess.items():
                # the values of gather_names are tuples out_key, scenario_name which allow mapping to global_dict_output
                out_key, scenario_name = gather_names[_gathered_var_name]
                gather_output_dict[out_key][scenario_name] = _gathered_var_value
        self.store_sos_outputs_values(gather_output_dict)

    ###################################
    ###################################
    ###### MONO INSTANCE PROCESS ######
    ###################################
    ###################################

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
            LOGGER.error(msg)
            raise ValueError(msg)
        else:
            not_relevant_columns = set(
                self.custom_samples.columns.to_list()) - set(self.attributes['selected_inputs'])
            msg = f'the following columns {not_relevant_columns} of the custom samples dataframe are filtered because they are not in eval_in.'
            LOGGER.warning(msg)
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
