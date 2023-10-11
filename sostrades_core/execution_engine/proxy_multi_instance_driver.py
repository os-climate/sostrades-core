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

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import copy
import pandas as pd
import numpy as np
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from gemseo.utils.compare_data_manager_tooling import dict_are_equal


class ProxyMultiInstanceDriverException(Exception):
    pass


class ProxyMultiInstanceDriver(ProxyDriverEvaluator):
    '''
    Class for driver on multi instance mode
    '''

    DISPLAY_OPTIONS_POSSIBILITIES = ['hide_under_coupling', 'hide_coupling_in_driver',
                                     'group_scenarios_under_disciplines', 'autogather']

    #            display_options (optional): Dictionary of display_options for multiinstance mode (value True or False) with options :
    #             'autogather' : will create an automatic gather discipline which will gather
    #                         all cls_builder outputs at driver node
    #             'hide_under_coupling' : Hide all disciplines created under the coupling at scenario name node for display purpose
    #             'hide_coupling_in_driver': Hide the coupling (scenario_name node) under the driver for display purpose
    #             'group_scenarios_under_disciplines' : Invert the order of scenario and disciplines for display purpose
    #                                                   Scenarios will be under discipline for the display treeview

    DISPLAY_OPTIONS_DEFAULT = {disp_option: False for disp_option in DISPLAY_OPTIONS_POSSIBILITIES}
    DISPLAY_OPTIONS = 'display_options'

    DESC_IN = {
        # MUST BE REPLACED BY SAMPLES_DF ?
        ProxyDriverEvaluator.SCENARIO_DF: {
            ProxyDriverEvaluator.TYPE: 'dataframe',
            ProxyDriverEvaluator.DEFAULT: pd.DataFrame(
                columns=[ProxyDriverEvaluator.SELECTED_SCENARIO, ProxyDriverEvaluator.SCENARIO_NAME]),
            ProxyDriverEvaluator.DATAFRAME_DESCRIPTOR: {ProxyDriverEvaluator.SELECTED_SCENARIO: ('bool', None, True),
                                                        ProxyDriverEvaluator.SCENARIO_NAME: ('string', None, True)},
            ProxyDriverEvaluator.DYNAMIC_DATAFRAME_COLUMNS: True,
            ProxyDriverEvaluator.DATAFRAME_EDITION_LOCKED: False,
            ProxyDriverEvaluator.EDITABLE: True,
            ProxyDriverEvaluator.STRUCTURING: True
        },  # TODO: manage variable columns for (non-very-simple) multiscenario cases
        # MUST BE REPLACED OR USED FOR GENERIC GATHERING FEATURE
        ProxyDriverEvaluator.EVAL_OUTPUTS: {
            ProxyDriverEvaluator.TYPE: 'dataframe',
            ProxyDriverEvaluator.DEFAULT: pd.DataFrame(columns=['selected_output', 'full_name', 'output_name']),
            ProxyDriverEvaluator.DATAFRAME_DESCRIPTOR: {'selected_output': ('bool', None, True),
                                                        'full_name': ('string', None, False),
                                                        'output_name': ('multiple', None, True)
                                                        },
            ProxyDriverEvaluator.DATAFRAME_EDITION_LOCKED: False,
            ProxyDriverEvaluator.STRUCTURING: True,
            # TODO: run-time coupling is not possible but might want variable in NS_EVAL for config-time coupling ?
            # ProxyDriverEvaluator.VISIBILITY: ProxyDriverEvaluator.SHARED_VISIBILITY,
            # ProxyDriverEvaluator.NAMESPACE: ProxyDriverEvaluator.NS_EVAL
        },
        ProxyDriverEvaluator.INSTANCE_REFERENCE: {
            ProxyDriverEvaluator.TYPE: 'bool',
            ProxyDriverEvaluator.DEFAULT: False,
            ProxyDriverEvaluator.POSSIBLE_VALUES: [True, False],
            ProxyDriverEvaluator.STRUCTURING: True
        },
        ProxyDriverEvaluator.GENERATED_SAMPLES: {ProxyDriverEvaluator.TYPE: 'dataframe',
                                                 ProxyDriverEvaluator.DATAFRAME_DESCRIPTOR: {
                                                     ProxyDriverEvaluator.SELECTED_SCENARIO: ('string', None, False),
                                                     ProxyDriverEvaluator.SCENARIO_NAME: ('string', None, False)},
                                                 ProxyDriverEvaluator.DYNAMIC_DATAFRAME_COLUMNS: True,
                                                 ProxyDriverEvaluator.DATAFRAME_EDITION_LOCKED: True,
                                                 ProxyDriverEvaluator.STRUCTURING: True,
                                                 ProxyDriverEvaluator.UNIT: None,
                                                 ProxyDriverEvaluator.DEFAULT: pd.DataFrame(),
                                                 ProxyDriverEvaluator.USER_LEVEL: 3
                                                 },
        DISPLAY_OPTIONS: {ProxyDriverEvaluator.TYPE: 'dict',
                          ProxyDriverEvaluator.STRUCTURING: True,
                          ProxyDriverEvaluator.DEFAULT: DISPLAY_OPTIONS_DEFAULT,
                          ProxyDriverEvaluator.SUBTYPE: {'dict': 'bool'}
                          }
    }

    DESC_IN.update(ProxyDriverEvaluator.DESC_IN)

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None,
                 process_display_options=None
                 ):
        """
        Constructor

        Arguments:
            sos_name (string): name of the discipline/node
            ee (ExecutionEngine): execution engine of the current process
            cls_builder (List[SoSBuilder]): list of the sub proxy builders
            driver_wrapper_cls (Class): class constructor of the driver wrapper (user-defined wrapper or SoSTrades wrapper or None)
            map_name (string): name of the map associated to the scatter builder in case of multi-instance build
            associated_namespaces(List[string]): list containing ns ids ['name__value'] for namespaces associated to builder
            process_display_options  [dict] still keep the possibility to modify display options through the process for archibuilder
        """
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces, map_name)

        self.display_options = None
        if process_display_options is not None:
            self.display_options = process_display_options

    def setup_sos_disciplines(self):
        disc_in = self.get_data_in()
        self.add_reference_mode(disc_in)
        self.add_gather_outputs(disc_in)

        self.set_generated_samples_values(disc_in)

    def set_generated_samples_values(self, disc_in):
        '''

        Args:
            disc_in: input dictionary with values

        if generated samples modify the selection of scenario in the scenario_df
        OPEN QUESTION : Do we need that ?

        '''
        if self.GENERATED_SAMPLES in disc_in:
            generated_samples = self.get_sosdisc_inputs(
                self.GENERATED_SAMPLES)
            generated_samples_dict = {
                self.GENERATED_SAMPLES: generated_samples}
            scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
            # checking whether generated_samples has changed
            # NB also doing nothing with an empty dataframe, which means sample needs to be regenerated to renew
            # scenario_df on 2nd config. The reason of this choice is that using an optional generated_samples
            # gives problems with structuring variables checks leading
            # to incomplete configuration sometimes
            if not (generated_samples.empty and not self.old_samples_df) \
                    and not dict_are_equal(generated_samples_dict, self.old_samples_df):
                # checking whether the dataframes are already coherent in which case the changes come probably
                # from a load and there is no need to crush the truth
                # values
                if not generated_samples.equals(
                        scenario_df.drop([self.SELECTED_SCENARIO, self.SCENARIO_NAME], axis=1)):
                    # TODO: could overload struct. var. check to spare this deepcopy (only if generated_samples
                    #  remains as a DriverEvaluator input, othrwise another sample change check logic is needed)
                    self.old_samples_df = copy.deepcopy(
                        generated_samples_dict)
                    # we crush old scenario_df and propose a df with
                    # all scenarios imposed by new sample, all
                    # de-activated
                    scenario_df = pd.DataFrame(
                        columns=[self.SELECTED_SCENARIO, self.SCENARIO_NAME])
                    scenario_df = pd.concat(
                        [scenario_df, generated_samples], axis=1)
                    n_scenarios = len(scenario_df.index)
                    # check whether the number of generated scenarios
                    # is not too high to auto-activate them
                    if self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS is None or n_scenarios <= self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS:
                        scenario_df[self.SELECTED_SCENARIO] = True
                    else:
                        self.logger.warning(
                            f'Sampled over {self.MAX_SAMPLE_AUTO_BUILD_SCENARIOS} scenarios, please select which to build. ')
                        scenario_df[self.SELECTED_SCENARIO] = False
                    scenario_name = scenario_df[self.SCENARIO_NAME]
                    for i in scenario_name.index.tolist():
                        scenario_name.iloc[i] = 'scenario_' + \
                                                str(i + 1)
                    self.logger.info(
                        'Generated sample has changed, updating scenarios to select.')
                    self.dm.set_data(self.get_var_full_name(self.SCENARIO_DF, disc_in),
                                     'value', scenario_df, check_value=False)

    def configure_driver(self):
        disc_in = self.get_data_in()
        if self.SCENARIO_DF in disc_in:
            self.configure_tool()
            self.configure_subprocesses_with_driver_input()
            self.set_eval_possible_values(io_type_in=False, strip_first_ns=True)

    def create_mdo_discipline_wrap(self, name, wrapper, wrapping_mode, logger):
        """
        No need to create a MDODisciplineWrap in the multi instance case , the computation is delegated to the coupling discipline above the driver
        """
        pass

    def prepare_build(self):
        """
        Call the tool to build the subprocesses in multi-instance builder mode.
        """
        if self.get_data_in():
            self.build_tool()
            # Tool is building disciplines for the driver on behalf of the driver name
            # no further disciplines needed to be builded by the evaluator
            # then we return an empty list
        return []

    def is_configured(self):
        config_status = super().is_configured()
        disc_in = self.get_data_in()
        # TODO: to be improved with ref. instance refacto
        if self.INSTANCE_REFERENCE in disc_in and self.get_sosdisc_inputs(self.INSTANCE_REFERENCE):
            config_status = config_status and (
                not self.check_if_there_are_reference_variables_changes()) and (
                                    self.sub_proc_import_usecase_status == 'No_SP_UC_Import'
                            )
        return config_status

    def update_reference(self):
        return self.INSTANCE_REFERENCE in self.get_data_in()

    def add_reference_mode(self, disc_in):
        '''
        Add reference mode as dynamic input if we are using instance reference option
        '''
        dynamic_inputs = {}

        if self.INSTANCE_REFERENCE in disc_in:
            instance_reference = self.get_sosdisc_inputs(
                self.INSTANCE_REFERENCE)
            if instance_reference:
                dynamic_inputs.update({self.REFERENCE_MODE:
                                           {self.TYPE: 'string',
                                            # SoSWrapp.DEFAULT: self.LINKED_MODE,
                                            self.POSSIBLE_VALUES: self.REFERENCE_MODE_POSSIBLE_VALUES,
                                            self.STRUCTURING: True}})

        self.add_inputs(dynamic_inputs)

    def add_gather_outputs(self, disc_in):
        '''

        Add gather output variables to dynamic desc_out to deal with gather option (autogather and eval_outputs)

        '''
        dynamic_outputs = {}
        if self.EVAL_OUTPUTS in disc_in:
            _vars_to_gather = self.get_sosdisc_inputs(self.EVAL_OUTPUTS)
            # we fetch the inputs and outputs selected by the user
            vars_to_gather = _vars_to_gather[_vars_to_gather['selected_output'] == True]
            selected_outputs = vars_to_gather['full_name'].values.tolist()
            outputs_names = vars_to_gather['output_name'].values.tolist()
            self._clear_gather_names()
            for out_var, out_name in zip(selected_outputs, outputs_names):
                _out_name = out_name or f'{out_var}{self.GATHER_DEFAULT_SUFFIX}'
                # Val : Possibility to add subtype for dict with output type maybe ?
                dynamic_outputs.update(
                    {_out_name: {self.TYPE: 'dict',
                                 self.VISIBILITY: 'Shared',
                                 self.NAMESPACE: self.NS_EVAL}})
                self._set_gather_names(out_var, _out_name)
                # TODO: Disc1.indicator_dict is shown as indicator_dict on GUI and it is not desired behaviour

        # so that eventual mono-instance outputs get clear
        if self.builder_tool is not None:
            dynamic_output_from_tool = self.builder_tool.get_dynamic_output_from_tool()
            dynamic_outputs.update(dynamic_output_from_tool)

        self.add_outputs(dynamic_outputs)

    def configure_tool(self):
        '''
        Instantiate the tool if it does not and prepare it with data that he needs (the tool know what he needs)
        '''
        if self.builder_tool is None:
            builder_tool_cls = self.ee.factory.create_scatter_tool_builder(
                'scatter_tool', map_name=self.map_name)
            self.builder_tool = builder_tool_cls.instantiate()
            self.builder_tool.associate_tool_to_driver(
                self, cls_builder=self.cls_builder, associated_namespaces=self.associated_namespaces)
        self.scenario_list_valid, self.scenario_list_integrity_msg = self.check_scenario_list_validity()
        if self.scenario_list_valid:
            self.builder_tool.prepare_tool()
        else:
            self.logger.error(self.scenario_list_integrity_msg)

    def build_tool(self):
        '''

        Build the tool if the list of scenario is valid

        '''
        if self.builder_tool is not None and self.scenario_list_valid:
            self.builder_tool.build()

    def check_scenario_list_validity(self):
        # checking for duplicates
        msg = ''
        if self.SCENARIO_DF in self.get_data_in():
            scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
            scenario_names = scenario_df[scenario_df[self.SELECTED_SCENARIO]
                                         == True][self.SCENARIO_NAME].values.tolist()
            set_sc_names = set(scenario_names)
            if len(scenario_names) != len(set_sc_names):
                repeated_elements = [
                    sc for sc in set_sc_names if scenario_names.count(sc) > 1]
                msg = 'Cannot activate several scenarios with the same name (' + \
                      repeated_elements[0]
                for sc in repeated_elements[1:]:
                    msg += ', ' + sc
                msg += ').'
                return False, msg
        # in any other case the list is valid
        return True, msg

    def subprocesses_built(self, scenario_names):
        """
        Check whether the subproxies built are coherent with the input list scenario_names.

        Arguments:
            scenario_names (list[string]): expected names of the subproxies.
        """
        if self.builder_tool:
            proxies_names = self.builder_tool.get_all_built_disciplines_names()
            # return self.builder_tool.has_built and proxies_names
            # TODO: upon overload of is_configured method can refactor quickfix below
            if self.builder_tool.has_built:
                return bool(proxies_names)
            else:
                self.set_configure_status(False)
                return False
        else:
            proxies_names = [disc.sos_name for disc in self.scenarios]
            return proxies_names != [] and set(proxies_names) == set(scenario_names)

    def prepare_variables_to_propagate(self):
        # TODO: code below might need refactoring after reference_scenario
        # configuration fashion is decided upon
        scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
        instance_reference = self.get_sosdisc_inputs(self.INSTANCE_REFERENCE)
        # sce_df = copy.deepcopy(scenario_df)

        if instance_reference:
            # Addition of Reference Scenario
            scenario_df = scenario_df.append(
                {self.SELECTED_SCENARIO: True,
                 self.SCENARIO_NAME: self.REFERENCE_SCENARIO_NAME},
                ignore_index=True)
        # NB assuming that the scenario_df entries are unique otherwise there
        # is some intelligence to be added
        scenario_names = scenario_df[scenario_df[self.SELECTED_SCENARIO]
                                     == True][self.SCENARIO_NAME].values.tolist()
        trade_vars = []
        # check that all the input scenarios have indeed been built
        # (configuration sequence allows the opposite)

        trade_vars = [col for col in scenario_df.columns if col not in
                      [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]
        return scenario_df, instance_reference, trade_vars, scenario_names

    def _clear_gather_names(self):
        """
        Clear attributes gather_names and gather_out_keys used for multi-instance gather capabilities.
        """
        self.gather_names = {}
        self.gather_out_keys = []

    def _set_gather_names(self, var_name, output_out_name):
        """
        Build a dictionary var_full_name : (output_name, scenario_name) to facilitate gather capabilities and gathered
        variable storage. This is done one variable at a time.

        Arguments:
            var_name: full name of variable to gather anonymized wrt scenario name node
            output_out_name: full name of output gather variable anonymized wrt output namespace node
        """

        self.gather_out_keys.append(output_out_name)

        gather_names_for_var = {}
        disc_in = self.get_data_in()
        if self.SCENARIO_DF in disc_in:
            driver_evaluator_ns = self.get_disc_full_name()
            scenario_df = self.get_sosdisc_inputs(self.SCENARIO_DF)
            scenario_names = scenario_df[scenario_df[self.SELECTED_SCENARIO] == True][
                self.SCENARIO_NAME].values.tolist()

            for sc in scenario_names:
                var_full_name = self.ee.ns_manager.compose_ns(
                    [driver_evaluator_ns, sc, var_name])
                gather_names_for_var[var_full_name] = (output_out_name, sc)
        self.gather_names.update(gather_names_for_var)

    def configure_subprocesses_with_driver_input(self):
        """
        This method forces the trade variables values of the subprocesses in function of the driverevaluator input df.
        """

        scenario_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()

        if self.subprocesses_built(scenario_names):
            if instance_reference:
                # propagate non trade variables values from reference scenario to other scenarios
                scenario_names = scenario_names[:-1]
                self.manage_reference_scenario_features(trade_vars, scenario_names)
            else:
                self.turn_other_variables_to_editable(scenario_names)
            # PROPAGATE TRADE VARIABLES VALUES FROM scenario_df
            # check that there are indeed variable changes input, with respect
            # to reference scenario
            if trade_vars:
                driver_evaluator_ns = self.ee.ns_manager.get_local_namespace_value(
                    self)
                scenarios_data_dict = {}
                for sc in scenario_names:
                    # assuming it is unique
                    sc_row = scenario_df[scenario_df[self.SCENARIO_NAME]
                                         == sc].iloc[0]
                    for var in trade_vars:
                        var_full_name = self.ee.ns_manager.compose_ns(
                            [driver_evaluator_ns, sc, var])
                        scenarios_data_dict[var_full_name] = sc_row.loc[var]
                if scenarios_data_dict and self.subprocess_is_configured():
                    # push to dm
                    # TODO: should also alter associated disciplines' reconfig.
                    # flags for structuring ? TO TEST
                    self.ee.dm.set_values_from_dict(scenarios_data_dict)
                    # self.ee.load_study_from_input_dict(scenarios_data_dict)

    '''
    All methods below are here for reference scenario and need to be cleaned 
    
    '''

    def manage_reference_scenario_features(self, trade_vars, scenario_names):
        '''

        Args:
            trade_vars: trade variables
            scenario_names: scenario names

        Header TO DO when dealing with reference scenario

        '''
        # ref_discipline_full_name =
        # ref_discipline.get_disc_full_name() # do provide the sting
        # path of data in flatten
        driver_evaluator_ns = self.get_disc_full_name()
        reference_scenario_ns = self.ee.ns_manager.compose_ns(
            [driver_evaluator_ns, self.REFERENCE_SCENARIO_NAME])
        # ref_discipline_full_name may need to be renamed has it is not
        # true in flatten mode
        ref_discipline_full_name = reference_scenario_ns

        # Manage usecase import
        self.manage_import_inputs_from_sub_process(
            ref_discipline_full_name)
        ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(
            trade_vars)

        scenarios_non_trade_vars_dict = self.transform_dict_from_reference_to_other_scenarios(scenario_names,
                                                                                              ref_dict)

        # Update of original editability state in case modification
        # scenario df
        if (not set(scenario_names) == set(self.old_scenario_names)):
            new_scenarios = set(scenario_names) - set(self.old_scenario_names)
            self.there_are_new_scenarios = True
            for new_scenario in new_scenarios:
                new_scenario_non_trade_vars_dict = {key: value
                                                    for key, value in scenarios_non_trade_vars_dict.items()
                                                    if new_scenario in key}

                new_scenario_editable_dict = self.save_original_editable_attr_from_non_trade_variables(
                    new_scenario_non_trade_vars_dict)
                self.original_editable_dict_non_ref.update(
                    new_scenario_editable_dict)
        self.old_scenario_names = scenario_names

        # Save the original editability state in case reference is
        # un-instantiated.
        self.save_original_editability_state(
            ref_dict, scenarios_non_trade_vars_dict)
        # Modification of read-only or editable depending on
        # LINKED_MODE or COPY_MODE
        self.modify_editable_attribute_according_to_reference_mode(
            scenarios_non_trade_vars_dict)
        # Propagation to other scenarios if necessary
        self.propagate_reference_non_trade_variables(
            ref_changes_dict, ref_dict, scenario_names)

    def turn_other_variables_to_editable(self, scenario_names):
        '''

        Args:
            scenario_names (list) : List of scenario names

        Turn Editable all the variables that needs to be editable in the original_editable_dict
        TO DO : better explain why

        '''
        if self.original_editable_dict_non_ref:
            for sc in scenario_names:
                for key in self.original_editable_dict_non_ref.keys():
                    if sc in key:
                        self.ee.dm.set_data(
                            key, 'editable', self.original_editable_dict_non_ref[key])

    # def set_reference_trade_variables_in_scenario_df(self, sce_df):
    #
    #     var_names = [col for col in sce_df.columns if col not in
    #                  [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]
    #
    #     index_ref_disc = self.get_reference_scenario_index()
    #     for var in var_names:
    #         short_name_var = var.split(".")[-1]
    #         for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
    #             if short_name_var in subdisc.get_data_in():
    #                 value_var = subdisc.get_sosdisc_inputs(short_name_var)
    #                 sce_df.at[sce_df.loc[sce_df[self.SCENARIO_NAME] == 'ReferenceScenario'].index, var] = value_var
    #
    #     return sce_df
    # def set_reference_trade_variables_in_scenario_df(self, sce_df):
    #
    #     var_names = [col for col in sce_df.columns if col not in
    #                  [self.SELECTED_SCENARIO, self.SCENARIO_NAME]]
    #
    #     index_ref_disc = self.get_reference_scenario_index()
    #     # for var in var_names:
    #     #    short_name_var = var.split(".")[-1]
    #     #    for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
    #     #        if short_name_var in subdisc.get_data_in():
    #     #            value_var = subdisc.get_sosdisc_inputs(short_name_var)
    #     #            sce_df.at[sce_df.loc[sce_df[self.SCENARIO_NAME]
    #     #                                 == 'ReferenceScenario'].index, var] = value_var
    #     # TODO
    #     # This is with error in case value_var is a list-like object (numpy array, list, set, tuple etc.)
    #     # https://stackoverflow.com/questions/48000225/must-have-equal-len-keys-and-value-when-setting-with-an-iterable
    #     # Example variable z = array([1., 1.]) of sellar put in trade variables
    #     return sce_df

    # These dicts are of non-trade variables
    def save_original_editability_state(self, ref_dict, non_ref_dict):

        if self.save_editable_attr:
            # self.original_editable_dict_ref = self.save_original_editable_attr_from_non_trade_variables(
            #     ref_dict)
            self.original_editable_dict_non_ref = self.save_original_editable_attr_from_non_trade_variables(
                non_ref_dict)
            # self.original_editability_dict = self.original_editable_dict_ref | self.original_editable_dict_non_ref
            # self.original_editability_dict = {**self.original_editable_dict_ref,
            #                                   **self.original_editable_dict_non_ref}
            self.save_editable_attr = False

    def get_reference_scenario_disciplines(self):
        reference_scenario_root_name = self.ee.ns_manager.compose_ns([self.get_disc_full_name(),
                                                                      self.REFERENCE_SCENARIO_NAME])
        return [disc for disc in self.scenarios if reference_scenario_root_name in disc.get_disc_full_name()]

    # def get_reference_scenario_index(self):
    #     """
    #     """
    #     index_ref = 0
    #     my_root = self.ee.ns_manager.compose_ns(
    #         [self.sos_name, self.REFERENCE_SCENARIO_NAME])
    #
    #     for disc in self.scenarios:
    #         if disc.sos_name == self.REFERENCE_SCENARIO_NAME \
    #                 or my_root in disc.sos_name:  # for flatten_subprocess
    #             # TODO: better implement this 2nd condition ?
    #             break
    #         else:
    #             index_ref += 1
    #     return index_ref

    def check_if_there_are_reference_variables_changes(self):

        scenario_df, instance_reference, trade_vars, scenario_names = self.prepare_variables_to_propagate()

        ref_changes_dict = {}
        if self.subprocesses_built(scenario_names):
            if instance_reference:
                ref_changes_dict, ref_dict = self.get_reference_non_trade_variables_changes(
                    trade_vars)

        return ref_changes_dict

    def get_reference_non_trade_variables_changes(self, trade_vars):
        # Take reference scenario non-trade variables (num and non-num) and its
        # values
        ref_dict = {}
        for ref_discipline in self.get_reference_scenario_disciplines():
            for key in ref_discipline.get_input_data_names():
                if all(key.split(self.REFERENCE_SCENARIO_NAME + '.')[-1] != trade_var for trade_var in trade_vars):
                    ref_dict[key] = ref_discipline.ee.dm.get_value(key)

        # Check if reference values have changed and select only those which
        # have changed

        ref_changes_dict = {}
        for key in ref_dict.keys():
            if key in self.old_ref_dict.keys():
                if isinstance(ref_dict[key], pd.DataFrame):
                    if not ref_dict[key].equals(self.old_ref_dict[key]):
                        ref_changes_dict[key] = ref_dict[key]
                elif isinstance(ref_dict[key], (np.ndarray)):
                    if not (np.array_equal(ref_dict[key], self.old_ref_dict[key])):
                        ref_changes_dict[key] = ref_dict[key]
                elif isinstance(ref_dict[key], (list)):
                    if not (np.array_equal(ref_dict[key], self.old_ref_dict[key])):
                        ref_changes_dict[key] = ref_dict[key]
                else:
                    if ref_dict[key] != self.old_ref_dict[key]:
                        ref_changes_dict[key] = ref_dict[key]
            else:
                ref_changes_dict[key] = ref_dict[key]

        # TODO: replace the above code by a more general function ...
        # ======================================================================
        # ref_changes_dict = {}
        # if self.old_ref_dict == {}:
        #     ref_changes_dict = ref_dict
        # else:
        #     # See Test 01 of test_69_compare_dict_compute_len
        #     compare_dict(ref_dict, self.old_ref_dict, '',
        #                  ref_changes_dict, df_equals=True)
        #     # We cannot use compare_dict as if: maybe we choude add a diff_compare_dict as an adaptation of compare_dict
        # ======================================================================

        return ref_changes_dict, ref_dict

    def propagate_reference_non_trade_variables(self, ref_changes_dict, ref_dict,
                                                scenario_names_to_propagate):

        if ref_changes_dict:
            self.old_ref_dict = copy.deepcopy(ref_dict)

        # ref_discipline = self.scenarios[self.get_reference_scenario_index()]

        # Build other scenarios variables and values dict from reference
        dict_to_propagate = {}
        # Propagate all reference
        if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
            dict_to_propagate = self.transform_dict_from_reference_to_other_scenarios(scenario_names_to_propagate,
                                                                                      ref_dict)
        # Propagate reference changes
        elif self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.COPY_MODE and ref_changes_dict:
            dict_to_propagate = self.transform_dict_from_reference_to_other_scenarios(scenario_names_to_propagate,
                                                                                      ref_changes_dict)
        # Propagate other scenarios variables and values
        if self.there_are_new_scenarios:
            if dict_to_propagate:
                self.ee.dm.set_values_from_dict(dict_to_propagate)
        else:
            if ref_changes_dict and dict_to_propagate:
                self.ee.dm.set_values_from_dict(dict_to_propagate)

    def get_other_evaluators_names_and_mode_under_current_one(self):
        other_evaluators_names_and_mode = []
        for disc in self.scenarios:
            name_and_modes = self.search_evaluator_names_and_modify_mode_iteratively(
                disc)
            if name_and_modes != []:
                other_evaluators_names_and_mode.append(name_and_modes)

        return other_evaluators_names_and_mode

    def search_evaluator_names_and_modify_mode_iteratively(self, disc):

        list = []
        # subdisc_to_check = disc.scenarios if hasattr(disc, 'scenarios') else disc.proxy_disciplines
        for subdisc in disc.proxy_disciplines:
            if subdisc.__class__.__name__ == 'ProxyMultiInstanceDriver':
                if subdisc.get_sosdisc_inputs(self.INSTANCE_REFERENCE) == True:
                    # If upper ProxyDriverEvaluator is in linked mode, all
                    # lower ProxyDriverEvaluator shall be as well.
                    if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
                        subdriver_full_name = self.ee.ns_manager.get_local_namespace_value(
                            subdisc)
                        if 'ReferenceScenario' in subdriver_full_name:
                            self.ee.dm.set_data(
                                subdriver_full_name + '.reference_mode', 'value', self.LINKED_MODE)
                    list = [subdisc.sos_name]
                else:
                    list = [subdisc.sos_name]
            elif subdisc.__class__.__name__ == 'ProxyDiscipline':
                pass
            else:
                name_and_modes = self.search_evaluator_names_and_modify_mode_iteratively(
                    subdisc)
                list.append(name_and_modes)

        return list

    def modify_editable_attribute_according_to_reference_mode(self, scenarios_non_trade_vars_dict):

        other_evaluators_names_and_mode = self.get_other_evaluators_names_and_mode_under_current_one()

        if self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.LINKED_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                self.ee.dm.set_data(key, 'editable', False)
        elif self.get_sosdisc_inputs(self.REFERENCE_MODE) == self.COPY_MODE:
            for key in scenarios_non_trade_vars_dict.keys():
                if other_evaluators_names_and_mode != []:  # This means there are evaluators under current one
                    for element in other_evaluators_names_and_mode:
                        if element[0] in key:  # Ignore variables from inner ProxyDriverEvaluators
                            pass
                        else:
                            if self.original_editable_dict_non_ref[key] == False:
                                pass
                            else:
                                self.ee.dm.set_data(key, 'editable', True)
                else:
                    if self.original_editable_dict_non_ref[key] == False:
                        pass
                    else:
                        self.ee.dm.set_data(key, 'editable', True)

    def save_original_editable_attr_from_non_trade_variables(self, dict):

        dict_out = {}
        for key in dict:
            dict_out[key] = self.dm.get_data(key, 'editable')

        return dict_out

    def transform_dict_from_reference_to_other_scenarios(self, scenario_names, dict_from_ref):

        transformed_to_other_scenarios_dict = {}
        for key in dict_from_ref.keys():
            for sc in scenario_names:
                if self.REFERENCE_SCENARIO_NAME in key and self.sos_name in key:
                    new_key = key.split(self.sos_name, 1)[0] + self.sos_name + '.' + sc + \
                              key.split(self.sos_name,
                                        1)[-1].split(self.REFERENCE_SCENARIO_NAME, 1)[-1]
                elif self.REFERENCE_SCENARIO_NAME in key and not self.sos_name in key:
                    new_key = key.split(self.REFERENCE_SCENARIO_NAME, 1)[
                                  0] + sc + key.split(self.REFERENCE_SCENARIO_NAME, 1)[-1]
                else:
                    new_key = key
                if self.dm.check_data_in_dm(new_key):
                    transformed_to_other_scenarios_dict[new_key] = dict_from_ref[key]

        return transformed_to_other_scenarios_dict

        # # Take non-trade variables values from subdisciplines of reference scenario
        # for subdisc in self.proxy_disciplines[index_ref_disc].proxy_disciplines:
        #     if subdisc.__class__.__name__ == 'ProxyDiscipline':
        #         # For ProxyDiscipline --> Propagation of non-trade variables
        #         self.propagate_non_trade_variables_of_proxy_discipline(subdisc, trade_vars)
        #     elif subdisc.__class__.__name__ == 'ProxyDriverEvaluator':
        #         # For ProxyDriverEvaluator --> Propagation of non-trade variables from ReferenceScenario (recursivity)
        #         subdisc.set_non_trade_variables_from_reference_scenario(trade_vars)
        #     else:
        #         # For ProxyCoupling... --> Propagation of its subdisciplines variables (recursively)
        #         self.propagate_non_trade_variables_of_proxy_coupling(subdisc, trade_vars)

    # def propagate_non_trade_variables_of_proxy_discipline(self, subdiscipline, trade_vars):
    #
    #     non_trade_var_dict_ref_to_propagate = {}
    #     non_trade_var_dict_not_ref_scenario = {}
    #     # Get non-numerical variables full name and values from reference
    #     non_num_var_dict = subdiscipline.get_non_numerical_variables_and_values_dict()
    #
    #     # If non-numerical variables have been set, select non-trade variables from them
    #     if all(value == None for value in non_num_var_dict.values()):
    #         pass
    #     else:
    #         for key in non_num_var_dict:  # Non-numerical variables
    #             if all(key.split('.ReferenceScenario.')[-1] != trade_var for trade_var in
    #                    trade_vars):  # Here non-trade variables are taken from non-numerical values
    #                 non_trade_var_dict_ref_to_propagate[key] = non_num_var_dict[key]
    #
    #     # Adapt non-trade variables and values from reference to full name of other scenarios
    #     if non_trade_var_dict_ref_to_propagate:
    #         for key in non_trade_var_dict_ref_to_propagate.keys():
    #             for sc in self.scenario_names[:-1]:
    #                 if 'ReferenceScenario' in key:
    #                     new_key = key.rsplit('ReferenceScenario', 1)[0] + sc + key.rsplit('ReferenceScenario', 1)[-1]
    #                     # new_key = driver_evaluator_ns + "." + sc + key.split('ReferenceScenario')[-1]
    #                 else:
    #                     new_key = key
    #                 non_trade_var_dict_not_ref_scenario[new_key] = non_trade_var_dict_ref_to_propagate[key]
    #
    #     if non_trade_var_dict_not_ref_scenario:
    #         self.ee.dm.set_values_from_dict(non_trade_var_dict_not_ref_scenario)
    #
    # def propagate_non_trade_variables_of_proxy_coupling(self, subcoupling, trade_vars):
    #     for subsubdisc in subcoupling.proxy_disciplines:
    #         if subsubdisc.__class__.__name__ == 'ProxyDiscipline':
    #             # For ProxyDiscipline --> Propagation of non-trade variables
    #             self.propagate_non_trade_variables_of_proxy_discipline(subsubdisc, trade_vars)
    #         elif subsubdisc.__class__.__name__ == 'ProxyDriverEvaluator':
    #             # For ProxyDriverEvaluator --> Propagation of non-trade variables from ReferenceScenario (recursivity)
    #             subsubdisc.set_non_trade_variables_from_reference_scenario(trade_vars)
    #         else:
    #             # For ProxyCoupling... --> Propagation of its subdisciplines variables (recursively)
    #             self.propagate_non_trade_variables_of_proxy_coupling(subsubdisc, trade_vars)
