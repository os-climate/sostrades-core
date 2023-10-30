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

from copy import copy
import pandas as pd
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline


class ValueBlockDiscipline(GatherDiscipline):
    """
    Generic Value Block Discipline getting children outputs as inputs and gathering them as outputs
    """

    # ontology information
    _ontology_data = {
        'label': 'Value Block Discipline',
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
    _maturity = 'Research'

    DESC_IN = {}
    
    def build_dynamic_io(self):
        dynamic_inputs = {}
        dynamic_outputs = {}
        self.gather_suffix = '_gather'

        children_list = self.config_dependency_disciplines
        for child in children_list:
            # child_name = child.sos_name.replace(f'{self.sos_name}.', '')
            # child_name = child.get_disc_full_name().split(
            #     f'{self.sos_name}.')[-1]
            for output, output_dict in child.get_data_io_dict(self.IO_TYPE_OUT).items():

                data_in_dict = {
                    key: value for key, value in output_dict.items() if key in self.NEEDED_DATA_KEYS}

                # if input is local : then put it to shared visibility and add the local namespace from child to the gather discipline as shared namespace
                # if input is shared : copy the namespace and rename it (at least two namespaces with same name but different value since it is a gather)
                # then add it as shared namespace for the gather discipline
                output_namespace = copy(data_in_dict[self.NS_REFERENCE])
                if data_in_dict[self.VISIBILITY] == self.LOCAL_VISIBILITY:
                    data_in_dict[self.VISIBILITY] = self.SHARED_VISIBILITY
                else:
                    output_namespace.name = output_namespace.value.split('.', 1)[-1]

                output_namespace_name = output_namespace.name

                short_alias = '.'.join([substr for substr in output_namespace_name.split('.') if
                                        substr not in self.get_disc_display_name().split('.')])
                self.add_new_shared_ns(output_namespace)
                data_in_dict[self.NAMESPACE] = output_namespace_name

                dynamic_inputs[(output, short_alias)] = data_in_dict
                if output.endswith(self.gather_suffix):
                    output_name = output
                else:
                    output_name = f'{output}{self.gather_suffix}'
                dynamic_outputs[output_name] = data_in_dict.copy()
                # if datafram then we store all the dataframes in one
                if dynamic_outputs[output_name][self.TYPE] != 'dataframe':
                    dynamic_outputs[output_name][self.TYPE] = 'dict'
                dynamic_outputs[output_name][self.VISIBILITY] = self.LOCAL_VISIBILITY
                del dynamic_outputs[output_name][self.NS_REFERENCE]
                del dynamic_outputs[output_name][self.NAMESPACE]
        return dynamic_inputs, dynamic_outputs
    
    def run(self):
        
        input_dict = self.get_sosdisc_inputs()
        output_dict = {}
        output_keys = self.get_sosdisc_outputs().keys()
        for out_key in output_keys:
            if out_key.endswith(self.gather_suffix):
                output_df_list = []
                output_dict[out_key] = {}
                var_key = out_key.replace(self.gather_suffix, '')
                for input_key in input_dict:
                    if isinstance(input_key, tuple) and input_key[0] == out_key:
                        # Then input_dict[input_key] is a dict
                        for input_input_key in input_dict[input_key]:
                            output_dict[out_key][input_input_key] = input_dict[input_key][input_input_key]
                    if isinstance(input_key, tuple) and input_key[0] == var_key:
                        if isinstance(input_dict[input_key], pd.DataFrame):
                            # create the dataframe list before concat
                            df_copy = input_dict[input_key].copy()
                            key_column = 'key'
                            df_copy = self.add_key_column_to_df(df_copy, key_column, input_key[1])

                            output_df_list.append(df_copy)
                        else:
                            output_dict[out_key][input_key[1]] = input_dict[input_key]
                if output_df_list != []:
                    # concat the list of dataframes to get the full dataframe
                    output_dict[out_key] = pd.concat(output_df_list, ignore_index=True)
        self.store_sos_outputs_values(output_dict)