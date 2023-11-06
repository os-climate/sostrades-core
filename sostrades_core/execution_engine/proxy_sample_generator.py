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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
from copy import copy, deepcopy
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.tools.gather.gather_tool import check_eval_io
from sostrades_core.execution_engine.ns_manager import NS_SEP
import pandas as pd


class ProxySampleGeneratorException(Exception):
    pass


class ProxySampleGenerator(ProxyDiscipline):
    '''
    Class that gather output data from a scatter discipline
    '''

    # ontology information
    _ontology_data = {
        'label': 'Sample Generator',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-outdent fa-fw',
        'version': '',
    }

    MULTIPLIER_PARTICULE = "__MULTIPLIER__" # todo: to delete
    EVAL_INPUTS = SampleGeneratorWrapper.EVAL_INPUTS
    SAMPLES_DF = SampleGeneratorWrapper.SAMPLES_DF
    SAMPLES_DF_DESC = SampleGeneratorWrapper.SAMPLES_DF_DESC_SHARED

    def set_eval_in_possible_values(self, possible_values):
        driver_is_configured = True
        # TODO: might want to refactor this eventually. If so, take into account that this "driver_is_configured" flag
        #  is a quick fix. The proper way is probably as follows: in this method just set the attribute eval_in_possible_values
        #  and handle SampleGenerator configuration status if it has changed. Then in SampleGenerator configuration do the
        #  remaining actions in the code below (set eval_inputs and handle corresponding samples_df columns update).
        if possible_values:
            driver_is_configured = False
            disc_in = self.get_data_in()
            # FIXME: this has to be done during configuration of the sampler. In the driver configuration, only an
            #  attribute should be set. Implement this to fix test 39_02
            if disc_in and self.get_sosdisc_inputs(
                    SampleGeneratorWrapper.SAMPLING_METHOD) == SampleGeneratorWrapper.CARTESIAN_PRODUCT:
                # NB: this if clause only exists because cartesian product has no input variable "eval_inputs"
                # TODO: so it has to disappear when eval_inputs_cp and eval_inputs are homogenized
                driver_is_configured = True
            elif self.EVAL_INPUTS in disc_in:
                driver_is_configured = True
                default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_values],
                                                     'full_name': possible_values})
                eval_input_new_dm = self.get_sosdisc_inputs(self.EVAL_INPUTS)
                eval_inputs_f_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)

                if eval_input_new_dm is None:
                    self.dm.set_data(eval_inputs_f_name,
                                     'value', default_in_dataframe, check_value=False)
                # check if the eval_inputs need to be updated after a subprocess
                # configure
                elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
                    error_msg = check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
                                       is_eval_input=True)
                    if len(error_msg) > 0:
                        for msg in error_msg:
                            self.logger.warning(msg)
                    default_dataframe = deepcopy(default_in_dataframe)
                    already_set_names = eval_input_new_dm['full_name'].tolist()
                    already_set_values = eval_input_new_dm['selected_input'].tolist()
                    for index, name in enumerate(already_set_names):
                        default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = \
                            already_set_values[
                                index]  # this will filter variables that are not inputs of the subprocess
                        if self.MULTIPLIER_PARTICULE in name:
                            default_dataframe = default_dataframe.append(
                                pd.DataFrame({'selected_input': [already_set_values[index]],
                                              'full_name': [name]}), ignore_index=True)
                    self.dm.set_data(eval_inputs_f_name,
                                     'value', default_dataframe, check_value=False)

                selected_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
                selected_inputs = selected_inputs[selected_inputs['selected_input'] == True]['full_name'].tolist()
                all_columns = [SampleGeneratorWrapper.SELECTED_SCENARIO,
                               SampleGeneratorWrapper.SCENARIO_NAME] + selected_inputs
                default_custom_dataframe = pd.DataFrame(
                    [[None for _ in range(len(all_columns))]], columns=all_columns)
                dataframe_descriptor = self.SAMPLES_DF_DESC['dataframe_descriptor'].copy()
                # This reflects 'samples_df' dynamic input has been configured and that
                # eval_inputs have changed
                if self.SAMPLES_DF in disc_in:
                    if disc_in[self.SAMPLES_DF]['value'] is not None:
                        from_samples = list(disc_in[self.SAMPLES_DF]['value'].keys())
                        from_eval_inputs = list(default_custom_dataframe.keys())
                        final_dataframe = pd.DataFrame(
                            None, columns=all_columns)

                        len_df = 1
                        for element in from_eval_inputs:
                            if element in from_samples:
                                len_df = len(disc_in[self.SAMPLES_DF]['value'])

                        for element in from_eval_inputs:
                            if element in from_samples:
                                final_dataframe[element] = disc_in[self.SAMPLES_DF]['value'][element]

                            else:
                                final_dataframe[element] = [None for _ in range(len_df)]
                        disc_in[self.SAMPLES_DF][self.VALUE] = final_dataframe
                    disc_in[self.SAMPLES_DF][self.DATAFRAME_DESCRIPTOR] = dataframe_descriptor
                elif self.get_sosdisc_inputs(SampleGeneratorWrapper.SAMPLING_GENERATION_MODE) == SampleGeneratorWrapper.AT_CONFIGURATION_TIME:
                    driver_is_configured = False
        return driver_is_configured

    # TODO: rewrite these functions for sample proxy migrating class variables etc.
    def setup_sos_disciplines(self):
        # NB: the sample generator might be configuring twice when configured by driver and also in standalone
        super().setup_sos_disciplines()
