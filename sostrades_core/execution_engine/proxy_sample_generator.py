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

    def set_eval_in_possible_values(self, possible_values):
        # TODO: might want to refactor this eventually
        if possible_values:
            disc_in = self.get_data_in()
            if self.EVAL_INPUTS in disc_in:
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

    # TODO: rewrite these functions for sample proxy migrating class variables etc.
    def setup_sos_disciplines(self):
        super().setup_sos_disciplines()