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

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.proxy_eval import ProxyEval
from sostrades_core.execution_engine.disciplines_wrappers.doe_eval import DoeEval
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.disciplines_wrappers.doe_wrapper import DoeWrapper
import pandas as pd
from collections import ChainMap


class ProxyDoeEval(ProxyEval):
    '''
    Generic DOE evaluation class
    '''

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.execution_engine.proxy_doe_eval',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    default_algo_options = {}

    # Design space dataframe headers
    VARIABLES = DoeEval.VARIABLES
    VALUES = DoeEval.VALUES
    UPPER_BOUND = "upper_bnd"
    LOWER_BOUND = "lower_bnd"
    TYPE = "type"
    ENABLE_VARIABLE_BOOL = "enable_variable"
    LIST_ACTIVATED_ELEM = "activated_elem"
    POSSIBLE_VALUES = 'possible_values'
    N_SAMPLES = "n_samples"
    DESIGN_SPACE = "design_space"

    ALGO = "sampling_algo"
    ALGO_OPTIONS = "algo_options"
    USER_GRAD = 'user'

    # To be defined in the heritage
    is_constraints = None
    INEQ_CONSTRAINTS = 'ineq_constraints'
    EQ_CONSTRAINTS = 'eq_constraints'
    # DESC_I/O
    PARALLEL_OPTIONS = 'parallel_options'

    DIMENSION = "dimension"
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    NS_SEP = '.'
    INPUT_TYPE = ['float', 'array', 'int', 'string']
    # INPUT_MULTIPLIER_TYPE = []
    # MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    # We define here the different default algo options in a case of a DOE
    # TODO: Implement a generic get_options functions to retrieve the default
    # options using directly the DoeFactory (todo since EEV3)

    # Default values of algorithms
    default_algo_options = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }

    default_algo_options_lhs = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }

    default_algo_options_fullfact = {
        'n_samples': 'default',
        'alpha': 'orthogonal',
        'eval_jac': False,
        'face': 'faced',
        'iterations': 5,
        'max_time': 0,
        'seed': 1,
        'center_bb': 'default',
        'center_cc': 'default',
        'criterion': 'default',
        'levels': 'default'
    }
    d = {'col1': [1, 2], 'col2': [3, 4]}
    X_pd = pd.DataFrame(data=d)

    default_algo_options_CustomDOE = {
        'n_processes': 1,
        'wait_time_between_samples': 0.0
    }

    default_algo_options_CustomDOE_file = {
        'eval_jac': False,
        'max_time': 0,
        'samples': None,
        'doe_file': 'X_pd.csv',
        'comments': '#',
        'delimiter': ',',
        'skiprows': 0
    }

    algo_dict = {"lhs": default_algo_options_lhs,
                 "fullfact": default_algo_options_fullfact,
                 "CustomDOE": default_algo_options_CustomDOE,
                 }

    def __init__(self, sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces=None):
        '''
        Constructor
        '''
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls,
                         associated_namespaces=associated_namespaces)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.DOE')
        self.doe_factory = DOEFactory()
        self.design_space = None
        self.dict_desactivated_elem = {}
        self.selected_outputs = []
        self.selected_inputs = []
        self.previous_algo_name = ""

    def _get_disc_shared_ns_value(self):
        return self.ee.ns_manager.disc_ns_dict[self]['others_ns']['ns_doe_eval'].get_value()

    def _get_dynamic_inputs_doe(self, disc_in, selected_inputs_has_changed):
        algo_name_has_changed = False
        algo_name = self.get_sosdisc_inputs(self.ALGO)
        if self.previous_algo_name != algo_name:
            algo_name_has_changed = True
            self.previous_algo_name = algo_name
        dynamic_inputs = {}
        if algo_name == 'CustomDOE':
            dynamic_inputs.update(super()._get_dynamic_inputs_doe(
                disc_in, selected_inputs_has_changed))
        else:

            default_design_space = pd.DataFrame({'variable': self.selected_inputs,

                                                 'lower_bnd': [[0.0, 0.0] if self.ee.dm.get_data(var,
                                                                                                 'type') == 'array' else 0.0
                                                               for var in self.eval_in_list],
                                                 'upper_bnd': [[10.0, 10.0] if self.ee.dm.get_data(var,
                                                                                                   'type') == 'array' else 10.0
                                                               for var in self.eval_in_list]
                                                 })

            dynamic_inputs.update({'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
                                                    }})
            if 'design_space' in disc_in and selected_inputs_has_changed:
                disc_in['design_space']['value'] = default_design_space

        #default_dict = self.get_algo_default_options(algo_name)
        default_dict = DoeWrapper.get_algo_default_options(algo_name)
        dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
                                                'dataframe_edition_locked': False,
                                                'structuring': True,

                                                'dataframe_descriptor': {
                                                    self.VARIABLES: ('string', None, False),
                                                    self.VALUES: ('string', None, True)}}})
        all_options = list(default_dict.keys())
        if 'algo_options' in disc_in and algo_name_has_changed:
            disc_in['algo_options']['value'] = default_dict
        if 'algo_options' in disc_in and disc_in['algo_options']['value'] is not None and list(
                disc_in['algo_options']['value'].keys()) != all_options:
            options_map = ChainMap(
                disc_in['algo_options']['value'], default_dict)
            disc_in['algo_options']['value'] = {
                key: options_map[key] for key in all_options}
        return dynamic_inputs

    def setup_sos_disciplines(self):
        # TODO: move to wrapper as it was originally?
        """
        Overload setup_sos_disciplines to create a dynamic desc_in
        default descin are the algo name and its options
        In case of a CustomDOE', additionnal input is the customed sample ( dataframe)
        In other cases, additionnal inputs are the number of samples and the design space
        """
        if self.ALGO in self.get_data_in():
            super().setup_sos_disciplines()

    # def __setup_sos_disciplines(self):
    #     # TODO: move to wrapper as it was originally?
    #     """
    #     Overload setup_sos_disciplines to create a dynamic desc_in
    #     default descin are the algo name and its options
    #     In case of a CustomDOE', additionnal input is the customed sample ( dataframe)
    #     In other cases, additionnal inputs are the number of samples and the design space
    #     """
    #
    #     dynamic_inputs = {}
    #     dynamic_outputs = {}
    #     algo_name_has_changed = False
    #     selected_inputs_has_changed = False
    #     disc_in = self.get_data_in()
    #
    #     # The setup of the discipline can begin once the algorithm we want to use to generate
    #     # the samples has been set
    #     if self.ALGO in disc_in:
    #         algo_name = self.get_sosdisc_inputs(self.ALGO)
    #         if self.previous_algo_name != algo_name:
    #             algo_name_has_changed = True
    #             self.previous_algo_name = algo_name
    #
    #         eval_outputs = self.get_sosdisc_inputs('eval_outputs')
    #         eval_inputs = self.get_sosdisc_inputs('eval_inputs')
    #
    #         # we fetch the inputs and outputs selected by the user
    #         selected_outputs = eval_outputs[eval_outputs['selected_output']
    #                                         == True]['full_name']
    #         selected_inputs = eval_inputs[eval_inputs['selected_input']
    #                                       == True]['full_name']
    #         if set(selected_inputs.tolist()) != set(self.selected_inputs):
    #             selected_inputs_has_changed = True
    #             self.selected_inputs = selected_inputs.tolist()
    #         self.selected_outputs = selected_outputs.tolist()
    #
    #         # doe can be done only for selected inputs and outputs
    #         # if algo_name is not None and len(selected_inputs) > 0 and len(selected_outputs) > 0:
    #         if len(selected_inputs) > 0 and len(selected_outputs) > 0:
    #             # we set the lists which will be used by the evaluation
    #             # function of sosEval
    #             self.set_eval_in_out_lists(selected_inputs, selected_outputs)
    #
    #             # setting dynamic outputs. One output of type dict per selected
    #             # output
    #
    #             for out_var in self.eval_out_list:
    #                 dynamic_outputs.update(
    #                     # {f'{out_var.split(".")[-1]}_dict': {'type': 'dict',
    #                     #                                     'visibility': 'Shared',
    #                     #                                     'namespace': 'ns_doe'}})
    #                     {f'{out_var.split(self.ee.study_name + ".",1)[1]}_dict': {'type': 'dict', 'visibility': 'Shared',
    #                                                                               'namespace': 'ns_doe'}})
    #
    #
    #
    #             # if algo_name == "CustomDOE":
    #             #     default_custom_dataframe = pd.DataFrame(
    #             #         [[NaN for input in range(len(self.selected_inputs))]], columns=self.selected_inputs)
    #             #     dataframe_descriptor = {}
    #             #     for i, key in enumerate(self.selected_inputs):
    #             #         cle = key
    #             #         var = tuple([self.ee.dm.get_data(
    #             #             self.eval_in_list[i], 'type'), None, True])
    #             #         dataframe_descriptor[cle] = var
    #             #
    #             #     dynamic_inputs.update(
    #             #         {'custom_samples_df': {'type': 'dataframe', self.DEFAULT: default_custom_dataframe,
    #             #                                'dataframe_descriptor': dataframe_descriptor,
    #             #                                'dataframe_edition_locked': False}})
    #             #     if 'custom_samples_df' in disc_in and selected_inputs_has_changed:
    #             #         disc_in['custom_samples_df']['value'] = default_custom_dataframe
    #             #         disc_in['custom_samples_df']['dataframe_descriptor'] = dataframe_descriptor
    #             #
    #             # else:
    #             #
    #             #     default_design_space = pd.DataFrame({'variable': selected_inputs,
    #             #
    #             #                                          'lower_bnd': [[0.0, 0.0] if self.ee.dm.get_data(var,
    #             #                                                                                          'type') == 'array' else 0.0
    #             #                                                        for var in self.eval_in_list],
    #             #                                          'upper_bnd': [[10.0, 10.0] if self.ee.dm.get_data(var,
    #             #                                                                                            'type') == 'array' else 10.0
    #             #                                                        for var in self.eval_in_list]
    #             #                                          })
    #             #
    #             #     dynamic_inputs.update(
    #             #         {'design_space': {'type': 'dataframe', self.DEFAULT: default_design_space
    #             #                           }})
    #             #     if 'design_space' in disc_in and selected_inputs_has_changed:
    #             #         disc_in['design_space']['value'] = default_design_space
    #
    #             default_dict = self.get_algo_default_options(algo_name)
    #             dynamic_inputs.update({'algo_options': {'type': 'dict', self.DEFAULT: default_dict,
    #                                                     'dataframe_edition_locked': False,
    #                                                     'structuring': True,
    #
    #                                                     'dataframe_descriptor': {
    #                                                         self.VARIABLES: ('string', None, False),
    #                                                         self.VALUES: ('string', None, True)}}})
    #             all_options = list(default_dict.keys())
    #             if 'algo_options' in disc_in and algo_name_has_changed:
    #                 disc_in['algo_options']['value'] = default_dict
    #             if 'algo_options' in disc_in and disc_in['algo_options']['value'] is not None and list(
    #                     disc_in['algo_options']['value'].keys()) != all_options:
    #                 options_map = ChainMap(
    #                     disc_in['algo_options']['value'], default_dict)
    #                 disc_in['algo_options']['value'] = {
    #                     key: options_map[key] for key in all_options}
    #
    #             # if multipliers in eval_in
    #             #MULTIPLIER
    #             # if (len(self.selected_inputs) > 0) and (any([self.MULTIPLIER_PARTICULE in val for val in self.selected_inputs])):
    #             #     generic_multipliers_dynamic_inputs_list = self.create_generic_multipliers_dynamic_input()
    #             #     for generic_multiplier_dynamic_input in generic_multipliers_dynamic_inputs_list:
    #             #         dynamic_inputs.update(generic_multiplier_dynamic_input)
    #
    #     self.add_inputs(dynamic_inputs)
    #     self.add_outputs(dynamic_outputs)
    #
    # #MULTIPLIER
    # # def create_generic_multipliers_dynamic_input(self):
    # #     dynamic_inputs_list = []
    # #     for selected_in in self.selected_inputs:
    # #         if self.MULTIPLIER_PARTICULE in selected_in:
    # #             multiplier_name = selected_in.split('.')[-1]
    # #             origin_var_name = multiplier_name.split('.')[0].split('@')[0]
    # #             # if
    # #             if len(self.ee.dm.get_all_namespaces_from_var_name(origin_var_name)) > 1:
    # #                 self.logger.exception(
    # #                     'Multiplier name selected already exists!')
    # #             origin_var_fullname = self.ee.dm.get_all_namespaces_from_var_name(origin_var_name)[
    # #                 0]
    # #             origin_var_ns = self.ee.dm.get_data(
    # #                 origin_var_fullname, 'namespace')
    # #             dynamic_inputs_list.append(
    # #                 {
    # #                     f'{multiplier_name}': {
    # #                         'type': 'float',
    # #                         'visibility': 'Shared',
    # #                         'namespace': origin_var_ns,
    # #                         'unit': self.ee.dm.get_data(origin_var_fullname).get('unit', '-'),
    # #                         'default': 100
    # #                     }
    # #                 }
    # #             )
    # #     return dynamic_inputs_list

    def get_algo_default_options(self, algo_name):
        """This algo generate the default options to set for a given doe algorithm
        """

        if algo_name in self.algo_dict.keys():
            return self.algo_dict[algo_name]
        else:
            return self.default_algo_options

    # def fill_possible_values(self, disc):
    #     '''
    #         Fill possible values lists for eval inputs and outputs
    #         an input variable must be a float coming from a data_in of a discipline in all the process
    #         and not a default variable
    #         an output variable must be any data from a data_out discipline
    #     '''
    #     # FIXME: need to accommodate long names and subprocess variables
    #     poss_in_values_full = []
    #     poss_out_values_full = []
    #     disc_in = disc.get_data_in()
    #     for data_in_key in disc_in.keys():
    #         is_input_type = disc_in[data_in_key][self.TYPE] in self.INPUT_TYPE
    #         is_structuring = disc_in[data_in_key].get(
    #             self.STRUCTURING, False)
    #         in_coupling_numerical = data_in_key in list(
    #             ProxyCoupling.DESC_IN.keys())
    #         full_id = disc.get_var_full_name(
    #             data_in_key, disc_in)
    #         is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
    #                                        ]['io_type'] == 'in'
    #         # is_input_multiplier_type = disc_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
    #         is_editable = disc_in[data_in_key]['editable']
    #         is_None = disc_in[data_in_key]['value'] is None
    #         if is_in_type and not in_coupling_numerical and not is_structuring and is_editable:
    #             # Caution ! This won't work for variables with points in name
    #             # as for ac_model
    #             # we remove the study name from the variable full  name for a
    #             # sake of simplicity
    #             if is_input_type:
    #                 poss_in_values_full.append(
    #                     full_id.split(self.ee.study_name + ".", 1)[1])
    #
    #             # if is_input_multiplier_type and not is_None:
    #             #     poss_in_values_list = self.set_multipliers_values(
    #             #         disc, full_id, data_in_key)
    #             #     for val in poss_in_values_list:
    #             #         poss_in_values_full.append(val)
    #
    #     disc_out = disc.get_data_out()
    #     for data_out_key in disc_out.keys():
    #         # Caution ! This won't work for variables with points in name
    #         # as for ac_model
    #         in_coupling_numerical = data_out_key in list(
    #             ProxyCoupling.DESC_IN.keys()) or data_out_key == 'residuals_history'
    #         full_id = disc.get_var_full_name(
    #             data_out_key, disc_out)
    #         if not in_coupling_numerical:
    #             # we remove the study name from the variable full  name for a
    #             # sake of simplicity
    #             poss_out_values_full.append(
    #                 full_id.split(self.ee.study_name + ".", 1)[1])
    #
    #     return poss_in_values_full, poss_out_values_full
    #
    # #MULTIPLIER
    # # def set_multipliers_values(self, disc, full_id, var_name):
    # #     poss_in_values_list = []
    # #     # if local var
    # #     disc_in = disc.get_data_in()
    # #     if 'namespace' not in disc_in[var_name]:
    # #         origin_var_ns = disc_in[var_name]['ns_reference'].value
    # #     else:
    # #         origin_var_ns = disc_in[var_name]['namespace']
    # #
    # #     disc_id = ('.').join(full_id.split('.')[:-1])
    # #     ns_disc_id = ('__').join([origin_var_ns, disc_id])
    # #     if ns_disc_id in disc.ee.ns_manager.all_ns_dict:
    # #         full_id_ns = ('.').join(
    # #             [disc.ee.ns_manager.all_ns_dict[ns_disc_id].value, var_name])
    # #     else:
    # #         full_id_ns = full_id
    # #
    # #     if disc_in[var_name][self.TYPE] == 'float':
    # #         multiplier_fullname = f'{full_id_ns}{self.MULTIPLIER_PARTICULE}'.split(
    # #             self.ee.study_name + ".", 1)[1]
    # #         poss_in_values_list.append(multiplier_fullname)
    # #
    # #     else:
    # #         df_var = disc_in[var_name]['value']
    # #         # if df_var is dict : transform dict to df
    # #         if disc_in[var_name][self.TYPE] == 'dict':
    # #             dict_var = disc_in[var_name]['value']
    # #             df_var = pd.DataFrame(
    # #                 dict_var, index=list(dict_var.keys()))
    # #         # check & create float columns list from df
    # #         columns = df_var.columns
    # #         float_cols_list = [col_name for col_name in columns if (
    # #             df_var[col_name].dtype == 'float' and not all(df_var[col_name].isna()))]
    # #         # if df with float columns
    # #         if len(float_cols_list) > 0:
    # #             for col_name in float_cols_list:
    # #                 col_name_clean = self.clean_var_name(col_name)
    # #                 multiplier_fullname = f'{full_id_ns}@{col_name_clean}{self.MULTIPLIER_PARTICULE}'.split(
    # #                     self.ee.study_name + ".", 1)[1]
    # #                 poss_in_values_list.append(multiplier_fullname)
    # #             # if df with more than one float column, create multiplier for all
    # #             # columns also
    # #             if len(float_cols_list) > 1:
    # #                 multiplier_fullname = f'{full_id_ns}@allcolumns{self.MULTIPLIER_PARTICULE}'.split(
    # #                     self.ee.study_name + ".", 1)[1]
    # #                 poss_in_values_list.append(multiplier_fullname)
    # #     return poss_in_values_list

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        super().set_eval_possible_values()
        # # the eval process to analyse is stored as the only child of SoSEval
        # # (coupling chain of the eval process or single discipline)
        # analyzed_disc = self.proxy_disciplines[0]
        #
        # possible_in_values_full, possible_out_values_full = self.fill_possible_values(
        #     analyzed_disc)
        #
        # possible_in_values_full, possible_out_values_full = self.find_possible_values(
        #     analyzed_disc, possible_in_values_full, possible_out_values_full)
        #
        # # Take only unique values in the list
        # possible_in_values_full = list(set(possible_in_values_full))
        # possible_out_values_full = list(set(possible_out_values_full))
        #
        # # Fill the possible_values of eval_inputs
        #
        # possible_in_values_full.sort()
        # possible_out_values_full.sort()
        #
        # default_in_dataframe = pd.DataFrame({'selected_input': [False for invar in possible_in_values_full],
        #                                      'full_name': possible_in_values_full})
        # default_out_dataframe = pd.DataFrame({'selected_output': [False for invar in possible_out_values_full],
        #                                       'full_name': possible_out_values_full})
        #
        # eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
        # eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
        # my_ns_doe_eval_path = self.ee.ns_manager.disc_ns_dict[self]['others_ns']['ns_doe_eval'].get_value(
        # )
        # if eval_input_new_dm is None:
        #     self.dm.set_data(f'{my_ns_doe_eval_path}.eval_inputs',
        #                      'value', default_in_dataframe, check_value=False)
        #
        # # check if the eval_inputs need to be updtated after a subprocess
        # # configure
        # elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
        #     self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
        #                        is_eval_input=True)
        #     default_dataframe = copy.deepcopy(default_in_dataframe)
        #     already_set_names = eval_input_new_dm['full_name'].tolist()
        #     already_set_values = eval_input_new_dm['selected_input'].tolist()
        #     for index, name in enumerate(already_set_names):
        #         default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
        #             index]
        #     self.dm.set_data(f'{my_ns_doe_eval_path}.eval_inputs',
        #                      'value', default_dataframe, check_value=False)
        #
        # if eval_output_new_dm is None:
        #     self.dm.set_data(f'{my_ns_doe_eval_path}.eval_outputs',
        #                      'value', default_out_dataframe, check_value=False)
        #     # check if the eval_inputs need to be updtated after a subprocess
        #     # configure
        # elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
        #     self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
        #                        is_eval_input=False)
        #     default_dataframe = copy.deepcopy(default_out_dataframe)
        #     already_set_names = eval_output_new_dm['full_name'].tolist()
        #     already_set_values = eval_output_new_dm['selected_output'].tolist()
        #     for index, name in enumerate(already_set_names):
        #         default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
        #             index]
        #     self.dm.set_data(f'{my_ns_doe_eval_path}.eval_outputs',
        #                      'value', default_dataframe, check_value=False)

        # filling possible values for sampling algorithm name
        self.dm.set_data(f'{self.get_disc_full_name()}.sampling_algo',
                         self.POSSIBLE_VALUES, self.custom_order_possible_algorithms(self.doe_factory.algorithms))

    def custom_order_possible_algorithms(self, algo_list):
        """ This algo sorts the possible algorithms list so that most used algorithms
        which are fullfact,lhs and CustomDOE appears at the top of the list
        The remaing algorithms are sorted in an alphabetical order
        """
        sorted_algorithms = algo_list[:]
        sorted_algorithms.remove('CustomDOE')
        sorted_algorithms.remove("fullfact")
        sorted_algorithms.remove("lhs")
        sorted_algorithms.sort()
        sorted_algorithms.insert(0, "lhs")
        sorted_algorithms.insert(0, 'CustomDOE')
        sorted_algorithms.insert(0, "fullfact")
        return sorted_algorithms

    # def check_eval_io(self, given_list, default_list, is_eval_input):
    #     """
    #     Set the evaluation variable list (in and out) present in the DM
    #     which fits with the eval_in_base_list filled in the usecase or by the user
    #     """
    #
    #     for given_io in given_list:
    #         if given_io not in default_list:
    #             if is_eval_input:
    #                 error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
    #                             f'input of the subprocess with the correct full name (without study name at the ' \
    #                             f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
    #                             f'be created. '
    #
    #             else:
    #                 error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
    #                             f'output of the subprocess with the correct full name (without study name at the ' \
    #                             f'beginning). Dynamic inputs might  not be created. '
    #
    #             self.logger.warning(error_msg)

    def set_wrapper_attributes(self, wrapper):
        # ProxyEval attributes
        super().set_wrapper_attributes(wrapper)
        # specific to ProxyDoeEval
        doeeval_attributes = {'dict_desactivated_elem': self.dict_desactivated_elem,
                              'doe_factory': self.doe_factory
                              }
        wrapper.attributes.update(doeeval_attributes)
