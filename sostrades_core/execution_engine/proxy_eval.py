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
import platform
import pandas as pd
import re
import copy

from tqdm import tqdm
import time

from gemseo.core.parallel_execution import ParallelExecution
from sostrades_core.tools.base_functions.compute_len import compute_len

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np
from pandas.core.frame import DataFrame

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_driver import ProxyDisciplineDriver
from sostrades_core.execution_engine.proxy_abstract_eval import ProxyAbstractEval
from numpy import NaN

class ProxyEvalException(Exception):
    pass


class ProxyEval(ProxyAbstractEval):
    '''
        SOSEval class which creates a sub process to evaluate
        with different methods (Gradient,FORM,Sensitivity ANalysis, DOE, ...)
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Eval Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    INPUT_TYPE = ['float', 'array', 'int', 'string']

    def __init__(self, sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces=None):
        '''
        Constructor
        '''
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        self.eval_in_base_list = None
        self.eval_in_list = None
        self.eval_out_base_list = None
        self.eval_out_list = None
        # Needed to reconstruct objects from flatten list
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Eval')
        # self.cls_builder = cls_builder
        # Create the eval process builder associated to SoSEval
        self.eval_process_builder = self._set_eval_process_builder()
        self.eval_process_disc = None
        self.selected_outputs = []
        self.selected_inputs = []

    def set_eval_in_out_lists(self, in_list, out_list, inside_evaluator=False):
        '''
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        '''
        # FIXME: mixing short names and full names
        self.eval_in_base_list = in_list
        self.eval_out_base_list = out_list
        self.eval_in_list = []
        for v_id in in_list:
            full_id_list = self.dm.get_all_namespaces_from_var_name(v_id)
            # full_id_list = [v_id]  #FIXME: quick fix so that eval works with full names
            for full_id in full_id_list:
                if not inside_evaluator:
                    self.eval_in_list.append(full_id)
                else:
                    if full_id.startswith(self.get_disc_full_name()):
                        self.eval_in_list.append(full_id)
        self.eval_out_list = []
        for v_id in out_list:
            full_id_list = self.dm.get_all_namespaces_from_var_name(v_id)
            # full_id_list = [v_id] #FIXME: quick fix so that eval works with full names
            for full_id in full_id_list:
                self.eval_out_list.append(full_id)

        # FIXME: manipulating namespaces manually
        # self.eval_in_base_list = [
        #     element.split(".")[-1] for element in in_list]
        # self.eval_out_base_list = [
        #     element.split(".")[-1] for element in out_list]
        # self.eval_in_list = [
        #     f'{self.ee.study_name}.{element}' for element in in_list]
        # self.eval_out_list = [
        #     f'{self.ee.study_name}.{element}' for element in out_list]

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        # disc_in = disc.get_data_in()
        # poss_in_values = []
        # poss_out_values = []
        # for data_in_key in disc_in.keys():
        #     is_float = disc_in[data_in_key][self.TYPE] == 'float'
        #     # structuring variables are excluded from possible values!!!
        #     is_structuring = disc_in[data_in_key].get(
        #         self.STRUCTURING, False)
        #     in_coupling_numerical = data_in_key in list(
        #         ProxyCoupling.DESC_IN.keys())
        #     full_id = self.dm.get_all_namespaces_from_var_name(data_in_key)[0]
        #     is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
        #                                    ]['io_type'] == 'in'
        #     if is_float and is_in_type and not in_coupling_numerical and not is_structuring:
        #         # Caution ! This won't work for variables with points in name
        #         # as for ac_model
        #         poss_in_values.append(data_in_key)
        # for data_out_key in disc.get_data_out().keys():
        #     # Caution ! This won't work for variables with points in name
        #     # as for ac_model
        #     poss_out_values.append(data_out_key.split(NS_SEP)[-1])
        #
        # return poss_in_values, poss_out_values

        # FIXME: need to accommodate long names and subprocess variables
        # FIXME: need to accommodate long names and subprocess variables
        poss_in_values_full = []
        poss_out_values_full = []
        disc_in = disc.get_data_in()
        for data_in_key in disc_in.keys():
            is_input_type = disc_in[data_in_key][self.TYPE] in self.INPUT_TYPE
            is_structuring = disc_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                ProxyCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                         ]['io_type'] == 'in'
            # is_input_multiplier_type = disc_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
            is_editable = disc_in[data_in_key]['editable']
            is_None = disc_in[data_in_key]['value'] is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                if is_input_type:
                    # poss_in_values_full.append(
                    #     full_id.split(self.ee.study_name + ".", 1)[1])
                    poss_in_values_full.append(full_id)

                # if is_input_multiplier_type and not is_None:
                #     poss_in_values_list = self.set_multipliers_values(
                #         disc, full_id, data_in_key)
                #     for val in poss_in_values_list:
                #         poss_in_values_full.append(val)

        disc_out = disc.get_data_out()
        for data_out_key in disc_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            in_coupling_numerical = data_out_key in list(
                ProxyCoupling.DESC_IN.keys()) or data_out_key == 'residuals_history'
            full_id = disc.get_var_full_name(
                data_out_key, disc_out)
            if not in_coupling_numerical:
                # we remove the study name from the variable full  name for a
                # sake of simplicity
                # poss_out_values_full.append(
                #     full_id.split(self.ee.study_name + ".", 1)[1])
                poss_out_values_full.append(full_id)

        return poss_in_values_full, poss_out_values_full

    def build(self):
        '''
        Method copied from SoSCoupling: build and store disciplines in sos_disciplines
        '''
        # set current_discipline to self to build and store eval process in the
        # children of SoSEval
        old_current_discipline = self.ee.factory.current_discipline
        self.ee.factory.current_discipline = self

        # if we want to build an eval coupling containing eval process,
        # we have to remove SoSEval name in current_ns to build eval coupling
        # at the same node as SoSEval
        if len(self.cls_builder) == 0:  # added condition for proc build
            pass
        elif self.cls_builder[0] != self.eval_process_builder:
            current_ns = self.ee.ns_manager.current_disc_ns
            self.ee.ns_manager.set_current_disc_ns(
                current_ns.split(f'.{self.sos_name}')[0])
            self.build_eval_process()
            # reset current_ns after build
            self.ee.ns_manager.set_current_disc_ns(current_ns)
        else:
            self.build_eval_process()

        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
        # level coupling and we do not have to restore the current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    def build_eval_process(self):
        # build coupling containing eval process if self.cls_builder[0] != self.eval_process_builder
        # or build and store eval process in the children of SoSEval
        self.eval_process_disc = self.eval_process_builder.build()
        # store coupling in the children of SoSEval
        if self.eval_process_disc not in self.proxy_disciplines:
            self.ee.factory.add_discipline(self.eval_process_disc)

    def configure_driver(self):
        # Extract variables for eval analysis
        if len(self.proxy_disciplines) > 0:
            self.set_eval_possible_values()

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # # FIXME: manipulating namespaces manually
        #
        # # the eval process to analyse is stored as the only child of SoSEval
        # # (coupling chain of the eval process or single discipline)
        # analyzed_disc = self.proxy_disciplines[0]
        #
        # possible_in_values, possible_out_values = self.fill_possible_values(
        #     analyzed_disc)
        #
        # possible_in_values, possible_out_values = self.find_possible_values(
        #     analyzed_disc, possible_in_values, possible_out_values)
        #
        # # Take only unique values in the list
        # possible_in_values = list(set(possible_in_values))
        # possible_out_values = list(set(possible_out_values))
        #
        # # Fill the possible_values of eval_inputs
        # self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
        #                  self.POSSIBLE_VALUES, possible_in_values)
        # self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
        #                  self.POSSIBLE_VALUES, possible_out_values)

        analyzed_disc = self.proxy_disciplines[0]

        # possible_in_values, possible_out_values = self.fill_possible_values(
        possible_in_values_full, possible_out_values_full = self.fill_possible_values(analyzed_disc)

        # possible_in_values, possible_out_values = self.find_possible_values(
        #     analyzed_disc, possible_in_values, possible_out_values)
        possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
                                                                      possible_in_values_full, possible_out_values_full)

        # Take only unique values in the list
        # possible_in_values = list(set(possible_in_values))
        # possible_out_values = list(set(possible_out_values))
        possible_in_values = list(set(possible_in_values_full))
        possible_out_values = list(set(possible_out_values_full))
        possible_in_values.sort()
        possible_out_values.sort()

        default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values],
                                             'full_name': possible_in_values})
        default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values],
                                              'full_name': possible_out_values})

        eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
        my_ns_eval_path = self.ee.ns_manager.disc_ns_dict[self]['others_ns']['ns_eval'].get_value(
        )

        if eval_input_new_dm is None:
            self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
                             'value', default_in_dataframe, check_value=False)
        # check if the eval_inputs need to be updated after a subprocess
        # configure
        elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
                               is_eval_input=True)
            default_dataframe = copy.deepcopy(default_in_dataframe)
            already_set_names = eval_input_new_dm['full_name'].tolist()
            already_set_values = eval_input_new_dm['selected_input'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
                    index]
            self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
                             'value', default_dataframe, check_value=False)

        if eval_output_new_dm is None:
            self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
                             'value', default_out_dataframe, check_value=False)
        # check if the eval_inputs need to be updated after a subprocess
        # configure
        elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
            self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
                               is_eval_input=False)
            default_dataframe = copy.deepcopy(default_out_dataframe)
            already_set_names = eval_output_new_dm['full_name'].tolist()
            already_set_values = eval_output_new_dm['selected_output'].tolist()
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
                    index]
            self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
                             'value', default_dataframe, check_value=False)

    def check_eval_io(self, given_list, default_list, is_eval_input):
        """
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        """

        for given_io in given_list:
            if given_io not in default_list:
                if is_eval_input:
                    error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
                                f'input of the subprocess with the correct full name (without study name at the ' \
                                f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
                                f'be created. '

                else:
                    error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
                                f'output of the subprocess with the correct full name (without study name at the ' \
                                f'beginning). Dynamic inputs might  not be created. '

                self.logger.warning(error_msg)

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        # FIXME: this involves recursive back and forths during configuration
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values)

        return possible_in_values, possible_out_values

    def get_x0(self):
        '''
        Get initial values for input values decided in the evaluation
        '''
        x0 = []
        for x_id in self.eval_in_list:
            x_val = self.dm.get_value(x_id)
            x0.append(x_val)
        return x0  # Removed cast to array

    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) == 1:
            disc_builder = self.cls_builder[0]
        else:
            # If eval process is a list of builders or a non executable builder,
            # then we build a coupling containing the eval process

            disc_builder = self.ee.factory.create_builder_coupling(
                self.sos_name+'.subprocess')
            disc_builder.set_builder_info('cls_builder', self.cls_builder)

        return disc_builder

    def set_wrapper_attributes(self, wrapper):
        """ set the attribute attributes of wrapper
        """
        # ProxyDisciplineDriver attributes (sub_mdo_discipline)
        super().set_wrapper_attributes(wrapper)
        eval_attributes = {'eval_in_list': self.eval_in_list,
                           'eval_out_list': self.eval_out_list,
                           'reference_scenario': self.get_x0(),
                           'activated_elems_dspace_df': [[True, True]
                                                         if self.ee.dm.get_data(var, 'type') == 'array' else [True]
                                                         for var in self.eval_in_list],  # TODO: Array dimensions greater than 2??? TEST
                           'study_name': self.ee.study_name,
                           'reduced_dm': self.ee.dm.reduced_dm,  # for conversions
                           'selected_inputs': self.selected_inputs,
                           'selected_outputs': self.selected_outputs,
                           }
        wrapper.attributes.update(eval_attributes)

    # def set_discipline_attributes(self, discipline):
    #     """ set the attribute attributes of gemseo object
    #     """
    #     # TODO : attribute has been added to SoSMDODiscipline __init__, use sos_disciplines rather ?
    #     discipline.disciplines = [self.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline]

    def setup_sos_disciplines(self):
        # TODO: move to wrapper as it was originally?
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        default descin are the algo name and its options
        In case of a CustomDOE', additionnal input is the customed sample ( dataframe)
        In other cases, additionnal inputs are the number of samples and the design space
        """

        dynamic_inputs = {}
        dynamic_outputs = {}
        selected_inputs_has_changed = False
        disc_in = self.get_data_in()

        # if 'eval_inputs' in disc_in:
        if len(disc_in) != 0:

            eval_outputs = self.get_sosdisc_inputs('eval_outputs')
            eval_inputs = self.get_sosdisc_inputs('eval_inputs')

            # we fetch the inputs and outputs selected by the user
            selected_outputs = eval_outputs[eval_outputs['selected_output']
                                            == True]['full_name']
            selected_inputs = eval_inputs[eval_inputs['selected_input']
                                          == True]['full_name']
            if set(selected_inputs.tolist()) != set(self.selected_inputs):
                selected_inputs_has_changed = True
                self.selected_inputs = selected_inputs.tolist()
            self.selected_outputs = selected_outputs.tolist()

            if len(selected_inputs) > 0 and len(selected_outputs) > 0:
                # we set the lists which will be used by the evaluation
                # function of sosEval
                self.set_eval_in_out_lists(selected_inputs, selected_outputs)

                # setting dynamic outputs. One output of type dict per selected
                # output
                #TODO: dirty namespacing
                for out_var in self.eval_out_list:
                    dynamic_outputs.update(
                        {f'{out_var.split(self.ee.study_name + ".", 1)[1]}_dict': {'type': 'dict',
                                                                                   'visibility': 'Shared',
                                                                                   'namespace': 'ns_eval'}})

                default_custom_dataframe = pd.DataFrame(
                    [[NaN for input in range(len(self.selected_inputs))]], columns=self.selected_inputs)
                dataframe_descriptor = {}
                for i, key in enumerate(self.selected_inputs):
                    cle = key
                    var = tuple([self.ee.dm.get_data(
                        self.eval_in_list[i], 'type'), None, True])
                    dataframe_descriptor[cle] = var

                dynamic_inputs.update(
                    {'custom_samples_df': {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
                                           'dataframe_descriptor': dataframe_descriptor,
                                           'dataframe_edition_locked': False}})
                if 'custom_samples_df' in disc_in and selected_inputs_has_changed:
                    disc_in['custom_samples_df']['value'] = default_custom_dataframe
                    disc_in['custom_samples_df']['dataframe_descriptor'] = dataframe_descriptor

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)
        # if (len(self.selected_inputs) > 0) and (
        #     any([self.MULTIPLIER_PARTICULE in val for val in self.selected_inputs])):
        #         generic_multipliers_dynamic_inputs_list = self.create_generic_multipliers_dynamic_input()
        #         for generic_multiplier_dynamic_input in generic_multipliers_dynamic_inputs_list:
        #             dynamic_inputs.update(generic_multiplier_dynamic_input)

        # # The setup of the discipline can begin once the algorithm we want to use to generate
        # # the samples has been set
        # if self.ALGO in disc_in:
        #     algo_name = self.get_sosdisc_inputs(self.ALGO)
        #     if self.previous_algo_name != algo_name:
        #         algo_name_has_changed = True
        #         self.previous_algo_name = algo_name
        #     eval_outputs = self.get_sosdisc_inputs('eval_outputs')
        #     eval_inputs = self.get_sosdisc_inputs('eval_inputs')
        #
        #     # we fetch the inputs and outputs selected by the user
        #     selected_outputs = eval_outputs[eval_outputs['selected_output']
        #                                     == True]['full_name']
        #     selected_inputs = eval_inputs[eval_inputs['selected_input']
        #                                   == True]['full_name']
        #     if set(selected_inputs.tolist()) != set(self.selected_inputs):
        #         selected_inputs_has_changed = True
        #         self.selected_inputs = selected_inputs.tolist()
        #     self.selected_outputs = selected_outputs.tolist()
        #
        #     # doe can be done only for selected inputs and outputs
        #     # if algo_name is not None and len(selected_inputs) > 0 and len(selected_outputs) > 0:
        #     if len(selected_inputs) > 0 and len(selected_outputs) > 0:
        #         # we set the lists which will be used by the evaluation
        #         # function of sosEval
        #         self.set_eval_in_out_lists(selected_inputs, selected_outputs)
        #
        #         # setting dynamic outputs. One output of type dict per selected
        #         # output
        #
        #         for out_var in self.eval_out_list:
        #             dynamic_outputs.update(
        #                 {f'{out_var.split(self.ee.study_name + ".", 1)[1]}_dict': {'type': 'dict',
        #                                                                            'visibility': 'Shared',
        #                                                                            'namespace': 'ns_doe'}})
        #
        #         if algo_name == "CustomDOE":
        #             default_custom_dataframe = pd.DataFrame(
        #                 [[NaN for input in range(len(self.selected_inputs))]], columns=self.selected_inputs)
        #             dataframe_descriptor = {}
        #             for i, key in enumerate(self.selected_inputs):
        #                 cle = key
        #                 var = tuple([self.ee.dm.get_data(
        #                     self.eval_in_list[i], 'type'), None, True])
        #                 dataframe_descriptor[cle] = var
        #
        #             dynamic_inputs.update(
        #                 {'custom_samples_df': {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
        #                                        'dataframe_descriptor': dataframe_descriptor,
        #                                        'dataframe_edition_locked': False}})
        #             if 'custom_samples_df' in disc_in and selected_inputs_has_changed:
        #                 disc_in['custom_samples_df']['value'] = default_custom_dataframe
        #                 disc_in['custom_samples_df']['dataframe_descriptor'] = dataframe_descriptor
        #
        #         else:
        #
        #             default_design_space = pd.DataFrame({'variable': selected_inputs,
        #
        #                                                  'lower_bnd': [[0.0, 0.0] if self.ee.dm.get_data(var,
        #                                                                                                  'type') == 'array' else 0.0
        #                                                                for var in self.eval_in_list],
        #                                                  'upper_bnd': [[10.0, 10.0] if self.ee.dm.get_data(var,
        #                                                                                                    'type') == 'array' else 10.0
        #                                                                for var in self.eval_in_list]
        #                                                  })
        #
        #             dynamic_inputs.update(
        #                 {'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
        #                                   }})
        #             if 'design_space' in disc_in and selected_inputs_has_changed:
        #                 disc_in['design_space']['value'] = default_design_space
        #
        #         default_dict = self.get_algo_default_options(algo_name)
        #         dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
        #                                                 'dataframe_edition_locked': False,
        #                                                 'structuring': True,
        #
        #                                                 'dataframe_descriptor': {
        #                                                     self.VARIABLES: ('string', None, False),
        #                                                     self.VALUES: ('string', None, True)}}})
        #         all_options = list(default_dict.keys())
        #         if 'algo_options' in disc_in and algo_name_has_changed:
        #             disc_in['algo_options']['value'] = default_dict
        #         if 'algo_options' in disc_in and disc_in['algo_options']['value'] is not None and list(
        #                 disc_in['algo_options']['value'].keys()) != all_options:
        #             options_map = ChainMap(
        #                 disc_in['algo_options']['value'], default_dict)
        #             disc_in['algo_options']['value'] = {
        #                 key: options_map[key] for key in all_options}
        #
        #         # if multipliers in eval_in
        #         if (len(self.selected_inputs) > 0) and (
        #         any([self.MULTIPLIER_PARTICULE in val for val in self.selected_inputs])):
        #             generic_multipliers_dynamic_inputs_list = self.create_generic_multipliers_dynamic_input()
        #             for generic_multiplier_dynamic_input in generic_multipliers_dynamic_inputs_list:
        #                 dynamic_inputs.update(generic_multiplier_dynamic_input)