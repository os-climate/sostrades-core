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
from sostrades_core.sos_wrapping.valueblock_discipline import ValueBlockDiscipline
from sostrades_core.tools.sumdfdict.toolboxsum import toolboxsum
from numpy import int32 as np_int32, float64 as np_float64, int64 as np_int64


class SumValueBlockDiscipline(ValueBlockDiscipline):
    """
    Generic Sum Value Block Discipline to sum outputs of its children
    """

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.sum_valueblock_discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-calculator fa-fw',
        'version': '',
    }
    STANDARD_TYPES = [int, float, np_int32, np_int64, np_float64, bool]

    def build_dynamic_io(self):
        """
        The sum is stored in the same name as the inputs found in the children_list
        """
        dynamic_inputs, dynamic_outputs = super().build_dynamic_io()
        self.input_to_sum = {}

        children_list = self.config_dependency_disciplines
        for child in children_list:
            for output, output_dict in child.get_data_io_dict(self.IO_TYPE_OUT).items():
                data_out_dict = {
                    key: value
                    for key, value in output_dict.items()
                    if key in self.NEEDED_DATA_KEYS
                }
                dynamic_outputs[output] = data_out_dict.copy()
                self.input_to_sum[output] = output_dict['type']

        return dynamic_inputs, dynamic_outputs

    def run(self):

        super().run()

        input_dict = self.get_sosdisc_inputs()
        output_dict = {}
        toolbox = toolboxsum()
        for input_to_sum, type_input in self.input_to_sum.items():
            if not input_to_sum.endswith('_gather'):
                # we sum only the same variable: condition endswith and only
                # the direct children condition len(split)==2
                sub_input_dict = {
                    key: value
                    for key, value in input_dict.items()
                    if isinstance(key, tuple) and key[0] == input_to_sum
                }

                if len(list(sub_input_dict.keys())) >= 2:

                    if type_input in ['float', 'int', 'array']:
                        output_dict[input_to_sum] = sum(
                            sub_input_dict.values())
                    elif type_input == 'dict':
                        sum_dict = {}

                        ref_dict = list(sub_input_dict.values())[0]

                        for ref_key, ref_value in ref_dict.items():
                            check_key = [
                                False
                                for sub_dict in sub_input_dict.values()
                                if ref_key not in list(sub_dict.keys())
                            ]

                            if (
                                isinstance(ref_value, tuple(
                                    self.STANDARD_TYPES))
                                and check_key == []
                            ):
                                values_to_sum = [
                                    sub_input_dict[input_key][ref_key]
                                    for input_key in sub_input_dict
                                ]
                                # check_types
                                if all(
                                    isinstance(sub, type(values_to_sum[0]))
                                    for sub in values_to_sum[1:]
                                ):
                                    sum_dict[ref_key] = sum(values_to_sum)
                            #     else:
                            #         self.logger.info(
                            #             f'Can not sum values {values_to_sum} for {ref_key} between these dictionaries {list(sub_input_dict.keys())} ')
                            # else:
                            #     self.logger.info(
                            # f'Can not sum {ref_key} between these
                            # dictionaries {list(sub_input_dict.keys())} ')
                        output_dict[input_to_sum] = sum_dict

                    elif type_input == 'dataframe':

                        if input_to_sum != 'percentage_resource':
                            sum_df, percent = toolbox.compute_sum_df(
                                [
                                    sub_input_dict[input_key]
                                    for input_key in sub_input_dict
                                ],
                                not_sum=['years', 'Quarters'],
                            )
                            output_dict[input_to_sum] = sum_df

                            if 'percentage_resource' in self.input_to_sum.keys():
                                tmp_list = list(sub_input_dict.keys())
                                percent.columns = ['years'] + [
                                    tmp.split(".", 1)[0] for tmp in tmp_list
                                ]
                                output_dict['percentage_resource'] = percent

                elif len(sub_input_dict) == 1:
                    output_dict[input_to_sum] = list(
                        sub_input_dict.values())[0]
                else:
                    raise Exception(
                        f'No input values are present to sum the variable {input_to_sum} in discipline {self.sos_name} '
                    )
        self.store_sos_outputs_values(output_dict)

    def modify_df(self, df_to_modify, col_list, df_dict_to_merge):
        '''Function to take some columns from the df_gather to give them to df_ac_name'''

        for key, values_df in df_dict_to_merge.items():
            if len(key.split('.')) == 1:
                for col in col_list:
                    if col in values_df:
                        index = df_to_modify.keys().to_list().index(
                            f'{col}') + 1
                        df_to_modify.insert(
                            index - 1, f'{col}_{key}', values_df[col], allow_duplicates=True)

        return df_to_modify
