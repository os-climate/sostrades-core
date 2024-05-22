'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/12-2023/11/08 Copyright 2023 Capgemini

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
import logging

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


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

    # MONO_INSTANCE = 'mono_instance'
    # MULTI_INSTANCE = 'multi_instance'
    # REGULAR_BUILD = 'regular_build'
    SUB_PROCESS_INPUTS = 'sub_process_inputs'
    USECASE_DATA = 'usecase_data'

    def __init__(self, sos_name, logger: logging.Logger):
        """
        Constructor.

        Arguments:
            sos_name (string): name of the discipline
            logger (logging.Logger): logger to use
        """
        super().__init__(sos_name=sos_name, logger=logger)
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
        input_data_names = disc.input_grammar.names
        if len(input_data_names) > 0:
            input_data = self.get_sosdisc_inputs(
                keys=input_data_names, in_dict=True, full_name_keys=True)
        return input_data

    def run(self):
        """
        Run overload
        """
        pass
