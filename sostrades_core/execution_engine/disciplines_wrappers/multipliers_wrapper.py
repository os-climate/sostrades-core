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
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
from sostrades_core.execution_engine.sample_generators.cartesian_product_sample_generator import CartesianProductSampleGenerator
import pandas as pd
import numpy as np
from collections import ChainMap
from gemseo.api import get_available_doe_algorithms

# get module logger not sos logger
import logging
LOGGER = logging.getLogger(__name__)


class MultipliersWrapper(SoSWrapp):
    '''

    '''

    _ontology_data = {
        'label': 'Multipliers wrapper',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'Multipliers wrapper that implements the __MULTIPLIERS__ capability to be used with grid search.',
        'icon': 'fas fa-grid-4 fa-fw',
        'version': ''
    }
    #TODO: add and refer class variables
    EVAL_INPUTS = 'eval_inputs'
    EVAL_INPUTS_CP = 'eval_inputs_cp'
    DISC_SHARED_NS = 'ns_sampling'

    INPUT_MULTIPLIER_TYPE = ['dict', 'dataframe', 'float']
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    DESC_IN = {EVAL_INPUTS:  {SoSWrapp.TYPE: 'dataframe',
                              SoSWrapp.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                              'full_name': ('string', None, False)},
                              SoSWrapp.DATAFRAME_EDITION_LOCKED: False,
                              SoSWrapp.STRUCTURING: True,
                              SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                              SoSWrapp.NAMESPACE: 'ns_sampling'}
               }

    def __init__(self, sos_name):
        super().__init__(sos_name)
        self.eval_ns = None
        self.eval_disc = None
        self.vars_with_multiplier = {}

    def setup_sos_disciplines(self):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of multipliers disc
        '''
        disc_in = self.get_data_in()
        dynamic_inputs = {}
        dynamic_outputs = {}

        self.add_multipliers(dynamic_inputs, disc_in)
        # self.apply_multipliers(dynamic_inputs, dynamic_outputs, disc_in)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def add_multipliers(self, dynamic_inputs, disc_in):
        if self.EVAL_INPUTS in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            self.eval_disc = None
            self.eval_ns = self.get_var_full_name(self.EVAL_INPUTS, disc_in).rsplit('.'+self.EVAL_INPUTS, 1)[0] # pylint: disable-msg=E1121
            disc_list = self.dm.get_disciplines_with_name(self.eval_ns)
            if disc_list:
                self.eval_disc = disc_list[0]
            if eval_inputs is not None and not eval_inputs.empty and self.eval_disc is not None:
                is_multiplier = eval_inputs['full_name'].str.contains(self.MULTIPLIER_PARTICULE)

                eval_inputs_base = eval_inputs[~is_multiplier]
                eval_inputs_mult = eval_inputs[is_multiplier]

                # find all possible multiplier values
                analyzed_disc = self.eval_disc
                possible_in_values_full, possible_out_values_full = [], []
                possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
                                                                                              possible_in_values_full,
                                                                                              possible_out_values_full)
                # Take only unique values in the list
                possible_in_values = list(set(possible_in_values_full))
                # these sorts are just for aesthetics
                possible_in_values.sort()

                selected_mult = []
                for var in possible_in_values:
                    row = eval_inputs_mult[eval_inputs_mult['full_name'] == var]
                    if not row.empty:
                        selected_mult.append(row['selected_input'].iloc[0] or False)
                    else:
                        selected_mult.append(False)

                multipliers_df = pd.DataFrame({'selected_input': selected_mult,
                                               'full_name': possible_in_values})

                eval_inputs = pd.concat([eval_inputs_base, multipliers_df], ignore_index=True)

                self.dm.set_data(f'{self.eval_ns}.{self.EVAL_INPUTS}',
                                 'value', eval_inputs, check_value=False)


        # dynamic_inputs.update({self.EVAL_INPUTS_CP: {self.TYPE: 'dataframe',
        #                                              self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
        #                                                                          'full_name': ('string', None, True),
        #                                                                          'list_of_values': ('list', None, True)},
        #                                              self.DATAFRAME_EDITION_LOCKED: False,
        #                                              self.STRUCTURING: True,
        #                                              self.VISIBILITY: self.SHARED_VISIBILITY,
        #                                              self.NAMESPACE: 'ns_sampling',
        #                                              self.DEFAULT: default_in_eval_input_cp}})
    #
    #
    # def set_eval_possible_values(self):
    #     '''
    #         Once all disciplines have been run through,
    #         set the possible values for eval_inputs and eval_outputs in the DM
    #     '''
    #     analyzed_disc = self.proxy_disciplines[0]
    #     possible_in_values_full, possible_out_values_full = self.fill_possible_values(
    #         analyzed_disc)
    #     possible_in_values_full, possible_out_values_full = self.find_possible_values(analyzed_disc,
    #                                                                                   possible_in_values_full,
    #                                                                                   possible_out_values_full)
    #
    #     # Take only unique values in the list
    #     possible_in_values = list(set(possible_in_values_full))
    #     possible_out_values = list(set(possible_out_values_full))
    #
    #     # these sorts are just for aesthetics
    #     possible_in_values.sort()
    #     possible_out_values.sort()
    #
    #     default_in_dataframe = pd.DataFrame({'selected_input': [False for _ in possible_in_values],
    #                                          'full_name': possible_in_values})
    #     default_out_dataframe = pd.DataFrame({'selected_output': [False for _ in possible_out_values],
    #                                           'full_name': possible_out_values})
    #
    #     eval_input_new_dm = self.get_sosdisc_inputs('eval_inputs')
    #     eval_output_new_dm = self.get_sosdisc_inputs('eval_outputs')
    #     my_ns_eval_path = self._get_disc_shared_ns_value()
    #
    #     if eval_input_new_dm is None:
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
    #                          'value', default_in_dataframe, check_value=False)
    #     # check if the eval_inputs need to be updated after a subprocess
    #     # configure
    #     elif set(eval_input_new_dm['full_name'].tolist()) != (set(default_in_dataframe['full_name'].tolist())):
    #         self.check_eval_io(eval_input_new_dm['full_name'].tolist(), default_in_dataframe['full_name'].tolist(),
    #                            is_eval_input=True)
    #         default_dataframe = copy.deepcopy(default_in_dataframe)
    #         already_set_names = eval_input_new_dm['full_name'].tolist()
    #         already_set_values = eval_input_new_dm['selected_input'].tolist()
    #         for index, name in enumerate(already_set_names):
    #             default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_input'] = already_set_values[
    #                 index]
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_inputs',
    #                          'value', default_dataframe, check_value=False)
    #
    #     if eval_output_new_dm is None:
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
    #                          'value', default_out_dataframe, check_value=False)
    #     # check if the eval_inputs need to be updated after a subprocess
    #     # configure
    #     elif set(eval_output_new_dm['full_name'].tolist()) != (set(default_out_dataframe['full_name'].tolist())):
    #         self.check_eval_io(eval_output_new_dm['full_name'].tolist(), default_out_dataframe['full_name'].tolist(),
    #                            is_eval_input=False)
    #         default_dataframe = copy.deepcopy(default_out_dataframe)
    #         already_set_names = eval_output_new_dm['full_name'].tolist()
    #         already_set_values = eval_output_new_dm['selected_output'].tolist()
    #         for index, name in enumerate(already_set_names):
    #             default_dataframe.loc[default_dataframe['full_name'] == name, 'selected_output'] = already_set_values[
    #                 index]
    #         self.dm.set_data(f'{my_ns_eval_path}.eval_outputs',
    #                          'value', default_dataframe, check_value=False)

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values_full = []
        poss_out_values_full = []
        disc_in = disc.get_data_in()
        for data_in_key in disc_in.keys():
            is_structuring = disc_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                ProxyCoupling.DESC_IN.keys())
            full_id = disc.get_var_full_name(
                data_in_key, disc_in)
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            is_input_multiplier_type = disc_in[data_in_key][self.TYPE] in self.INPUT_MULTIPLIER_TYPE
            is_editable = disc_in[data_in_key]['editable']
            value = disc_in[data_in_key]['value']
            is_none = value is None
            if is_in_type and not in_coupling_numerical and not is_structuring and is_editable and is_input_multiplier_type:
                self.vars_with_multiplier[full_id] = copy.deepcopy(value)
                if not is_none:
                    poss_in_values_list = self.set_multipliers_values(
                        disc, full_id, data_in_key)
                    for val in poss_in_values_list:
                        poss_in_values_full.append(val)
        return poss_in_values_full, poss_out_values_full

    # def find_possible_values(self, disc, possible_in_values, possible_out_values):
    #     return ProxyDriverEvaluator.find_possible_values(self, disc, possible_in_values, possible_out_values)

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        # TODO: copy-pasted code should b refactored (see above)
        # configuration ? (<-> config. graph)
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values)
        return possible_in_values, possible_out_values

    def set_multipliers_values(self, disc, full_id, var_name):
        poss_in_values_list = []
        disc_in = disc.get_data_in()
        # if local var
        if 'namespace' not in disc_in[var_name]:
            origin_var_ns = disc_in[var_name]['ns_reference'].value
        else:
            origin_var_ns = disc_in[var_name]['namespace']

        disc_id = ('.').join(full_id.split('.')[:-1])
        ns_disc_id = ('__').join([origin_var_ns, disc_id])
        if ns_disc_id in disc.ee.ns_manager.all_ns_dict:
            full_id_ns = ('.').join(
                [disc.ee.ns_manager.all_ns_dict[ns_disc_id].value, var_name]
            )
        else:
            full_id_ns = full_id

        if disc_in[var_name][self.TYPE] == 'float':
            multiplier_fullname = f'{full_id_ns}{self.MULTIPLIER_PARTICULE}'.split(
                self.eval_ns + ".", 1
            )[1]
            poss_in_values_list.append(multiplier_fullname)

        else:
            df_var = disc_in[var_name]['value']
            # if df_var is dict : transform dict to df
            if disc_in[var_name][self.TYPE] == 'dict':
                dict_var = disc_in[var_name]['value']
                df_var = pd.DataFrame(dict_var, index=list(dict_var.keys()))
            # check & create float columns list from df
            columns = df_var.columns
            float_cols_list = [
                col_name
                for col_name in columns
                if (
                    df_var[col_name].dtype == 'float'
                    and not all(df_var[col_name].isna())
                )
            ]
            # if df with float columns
            if len(float_cols_list) > 0:
                for col_name in float_cols_list:
                    col_name_clean = self.clean_var_name(col_name)
                    multiplier_fullname = f'{full_id_ns}@{col_name_clean}{self.MULTIPLIER_PARTICULE}'.split(
                        self.eval_ns + ".", 1
                    )[
                        1
                    ]
                    poss_in_values_list.append(multiplier_fullname)
                # if df with more than one float column, create multiplier for all
                # columns also
                if len(float_cols_list) > 1:
                    multiplier_fullname = (
                        f'{full_id_ns}@allcolumns{self.MULTIPLIER_PARTICULE}'.split(
                            self.eval_ns + ".", 1
                        )[1]
                    )
                    poss_in_values_list.append(multiplier_fullname)
        return poss_in_values_list

    def clean_var_name(self, var_name):
        return re.sub(r"[^a-zA-Z0-9]", "_", var_name)

    def run(self):
        pass

    def is_configured(self):
        return not self.check_for_multiplier_changes()

    def check_for_multiplier_changes(self):
        if self.eval_disc is not None:
            subprocess_inputs = self.eval_disc.get_data_io_with_full_name(self.IO_TYPE_IN, as_namespaced_tuple=False)
            new_vars_with_multiplier = {key: value for key, value in subprocess_inputs.items() if key in self.vars_with_multiplier}
            # purge attribute without updating it as configuration process is in charge of update
            self.vars_with_multiplier = {key: value for key, value in self.vars_with_multiplier.items() if key in new_vars_with_multiplier}
            return not dict_are_equal(self.vars_with_multiplier, new_vars_with_multiplier)
        else:
            return False