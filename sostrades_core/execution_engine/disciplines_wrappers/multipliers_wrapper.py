'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/03 Copyright 2023 Capgemini

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



from sostrades_core.execution_engine.disciplines_wrappers.sample_generator_wrapper import SampleGeneratorWrapper
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import logging

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
import pandas as pd


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
    # TODO: add and refer class variables
    EVAL_INPUTS = 'eval_inputs'
    EVAL_INPUTS_CP = 'eval_inputs'
    DISC_SHARED_NS = SampleGeneratorWrapper.NS_SAMPLING

    INPUT_MULTIPLIER_TYPE = ['dict', 'dataframe', 'float']
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name=sos_name, logger=logger)
        self.eval_ns = None
        self.eval_disc = None
        self.vars_with_multiplier = {}  # variables in the subprocess that accept a multiplier
        self.multiplier_variables = {}  # dict containing io names for runtime

    def setup_sos_disciplines(self):
        '''
        Overload of setup_sos_disciplines to specify the specific dynamic inputs of multipliers disc
        '''
        disc_in = self.get_data_in()
        # TODO: dm calls or dynamic inputs ?

        dynamic_inputs = {}
        dynamic_outputs = {}

        self.setup_multipliers(dynamic_inputs)
        self.add_multipliers(disc_in)
        self.add_dynamic_io_for_selected_multipliers(disc_in, dynamic_inputs, dynamic_outputs)

        # self.apply_multipliers(disc_in)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def add_dynamic_io_for_selected_multipliers(self, disc_in, dynamic_inputs, dynamic_outputs):
        self.multiplier_variables = {}
        if self.EVAL_INPUTS in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            if eval_inputs is not None and not eval_inputs.empty:
                for idx, row in eval_inputs.iterrows():
                    var_name = row['full_name']
                    selected = row['selected_input']
                    if selected and self.MULTIPLIER_PARTICULE in var_name:
                        if '@' in var_name:
                            orig_var_name = var_name.rsplit('@', 1)[0]
                        else:
                            orig_var_name = var_name.rsplit(self.MULTIPLIER_PARTICULE, 1)[0]
                        orig_var_f_name = f'{self.eval_ns}.{orig_var_name}'
                        # add the multiplier to the input
                        dynamic_inputs.update({var_name: {self.TYPE: 'float',
                                                          self.VISIBILITY: self.SHARED_VISIBILITY,
                                                          self.NAMESPACE: self.DISC_SHARED_NS,
                                                          self.UNIT: '%',
                                                          self.DEFAULT: 100}})
                        self.multiplier_variables.update({var_name: orig_var_name})
                        # add the original variable to the output
                        self.dm.set_data(orig_var_f_name, self.VISIBILITY, self.SHARED_VISIBILITY)
                        data_dict = copy.copy(self.dm.get_data(orig_var_f_name))
                        dynamic_inputs.update({orig_var_name: {self.TYPE: data_dict[self.TYPE],
                                                               self.VISIBILITY: self.SHARED_VISIBILITY,
                                                               self.NAMESPACE: self.DISC_SHARED_NS,
                                                               self.DATAFRAME_DESCRIPTOR: data_dict[
                                                                   self.DATAFRAME_DESCRIPTOR]}})

                        dynamic_outputs.update({orig_var_name: {self.TYPE: data_dict[self.TYPE],
                                                                self.VISIBILITY: self.SHARED_VISIBILITY,
                                                                self.NAMESPACE: self.DISC_SHARED_NS,
                                                                self.DATAFRAME_DESCRIPTOR: data_dict[
                                                                    self.DATAFRAME_DESCRIPTOR]}})

    def apply_multipliers(self, disc_in):
        update_dm = False
        if self.EVAL_INPUTS_CP in disc_in:
            eval_inputs_cp = self.get_sosdisc_inputs(self.EVAL_INPUTS_CP)
            if eval_inputs_cp is not None and not eval_inputs_cp.empty and self.eval_disc is not None:
                for idx, row in eval_inputs_cp.iterrows():
                    var_name = row['full_name']
                    if self.MULTIPLIER_PARTICULE in var_name:
                        if '@' in var_name:
                            new_var_name = var_name.rsplit('@', 1)[0]
                        else:
                            new_var_name = var_name.rsplit(self.MULTIPLIER_PARTICULE, 1)[0]
                        ref_value = self.dm.get_value(f'{self.eval_ns}.{new_var_name}')
                        new_list_of_values = [self.apply_multiplier(var_name, multiplier_value, ref_value)
                                              for multiplier_value in row['list_of_values']]
                        eval_inputs_cp.iloc[idx]['full_name'] = new_var_name
                        eval_inputs_cp.iloc[idx]['list_of_values'] = new_list_of_values
                        update_dm = True

            if update_dm:
                full_name = self.get_var_full_name(self.EVAL_INPUTS_CP, disc_in)
                self.dm.set_data(full_name, 'value', eval_inputs_cp, check_value=False)

    def find_eval_disc(self, disc):
        fe = disc.get_father_executor()
        if isinstance(fe, ProxyDriverEvaluator):
            return fe
        else:
            return self.find_eval_disc(fe)

    def setup_multipliers(self, dynamic_inputs):
        self.eval_disc = self.find_eval_disc(self)
        self.eval_ns = self.eval_disc.get_disc_full_name()
        self.eval_disc.get_data_in()
        # self.add_disc_to_config_dependency_disciplines(self.eval_disc) # creates a cycle...
        # for disc in self.get_father_executor().proxy_disciplines:
        #     if not hasattr(disc.mdo_discipline_wrapp, 'wrapper') \
        #        or not isinstance(disc.mdo_discipline_wrapp.wrapper, MultipliersWrapper):
        #         self.add_disc_to_config_dependency_disciplines(disc)

        dynamic_inputs.update({self.EVAL_INPUTS: {self.TYPE: 'dataframe',
                                                  self.DATAFRAME_DESCRIPTOR: {'selected_input': ('bool', None, True),
                                                                              'full_name': ('string', None, False)},
                                                  self.DATAFRAME_EDITION_LOCKED: False,
                                                  self.STRUCTURING: True,
                                                  self.VISIBILITY: self.SHARED_VISIBILITY,
                                                  self.NAMESPACE: self.DISC_SHARED_NS}
                               })
        # self.add_inputs(dynamic_inputs)

    def add_multipliers(self, disc_in):

        if self.EVAL_INPUTS in disc_in:
            eval_inputs = self.get_sosdisc_inputs(self.EVAL_INPUTS)
            if eval_inputs is not None and not eval_inputs.empty:
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
                full_name = self.get_var_full_name(self.EVAL_INPUTS, disc_in)
                self.dm.set_data(full_name,
                                 'value', eval_inputs, check_value=False)

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values_full = []
        poss_out_values_full = []
        if hasattr(disc.mdo_discipline_wrapp, 'wrapper') and \
                isinstance(disc.mdo_discipline_wrapp.wrapper, MultipliersWrapper):
            pass
        else:
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
            to find possible values for eval_inputs and gather_outputs
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

    def is_configured(self):
        return not self.check_for_multiplier_changes()

    def check_for_multiplier_changes(self):
        if self.eval_disc is not None:
            new_vars_with_multiplier = {}
            for key in self.vars_with_multiplier:
                if key in self.dm.data_id_map:
                    new_vars_with_multiplier[key] = self.dm.get_value(key)
                else:
                    del self.vars_with_multiplier[key]
            # purge entries without updating values of existent ones as configuration process is in charge of update
            return not dict_are_equal(self.vars_with_multiplier, new_vars_with_multiplier)
        else:
            return False

    def apply_multiplier(self, multiplier_name, multiplier_value, var_to_update):
        # if dict or dataframe to be multiplied
        if '@' in multiplier_name:
            var_updated = copy.deepcopy(var_to_update)
            col_name_clean = multiplier_name.split(self.MULTIPLIER_PARTICULE)[0].split(
                '@'
            )[1]
            if col_name_clean == 'allcolumns':
                if isinstance(var_to_update, dict):
                    float_cols_ids_list = [
                        dict_keys
                        for dict_keys in var_to_update
                        if isinstance(var_to_update[dict_keys], float)
                    ]
                elif isinstance(var_to_update, pd.DataFrame):
                    float_cols_ids_list = [
                        df_keys
                        for df_keys in var_to_update
                        if var_to_update[df_keys].dtype == 'float'
                    ]
                for key in float_cols_ids_list:
                    var_updated[key] = multiplier_value * var_to_update[key]
            else:
                keys_clean = [self.clean_var_name(
                    var) for var in var_to_update.keys()]
                col_index = keys_clean.index(col_name_clean)
                col_name = var_to_update.keys()[col_index]
                var_updated[col_name] = multiplier_value * \
                                        var_to_update[col_name]
        # if float to be multiplied
        else:
            var_updated = multiplier_value * var_to_update
        return var_updated

    def run(self):
        outputs = dict.fromkeys(set(self.multiplier_variables.values()))
        for orig_var in outputs:
            outputs[orig_var] = self.get_sosdisc_inputs(orig_var)
        for multiplier_var, orig_var in self.multiplier_variables.items():
            outputs[orig_var] = self.apply_multiplier(multiplier_name=multiplier_var,
                                                      multiplier_value=self.get_sosdisc_inputs(multiplier_var),
                                                      var_to_update=outputs[orig_var])
        self.store_sos_outputs_values(outputs)
