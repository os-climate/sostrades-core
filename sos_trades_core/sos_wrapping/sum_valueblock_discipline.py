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
from sos_trades_core.sos_wrapping.valueblock_discipline import ValueBlockDiscipline
from sos_trades_core.tools.sumdfdict.toolboxsum import toolboxsum


class SumValueBlockDiscipline(ValueBlockDiscipline):
    """
    Generic Sum Value Block Discipline to sum outputs of its children
    """

    def build_dynamic_io(self):
        '''
        The sum is stored in the same name as the inputs found in the children_list 
        '''
        dynamic_inputs, dynamic_outputs = ValueBlockDiscipline.build_dynamic_io(
            self)
        self.input_to_sum = {}

        for child in self.children_list:
            for output, output_dict in child.get_data_io_dict(self.IO_TYPE_OUT).items():
                data_out_dict = {
                    key: value for key, value in output_dict.items() if key in self.NEEDED_DATA_KEYS}
                dynamic_outputs[output] = data_out_dict.copy()
                self.input_to_sum[output] = output_dict['type']
        self.add_outputs(dynamic_outputs)

        return dynamic_inputs, dynamic_outputs

    def run(self):

        ValueBlockDiscipline.run(self)

        input_dict = self.get_sosdisc_inputs()
        output_dict = {}
        toolbox = toolboxsum()
        for input_to_sum, type_input in self.input_to_sum.items():
            if not input_to_sum.endswith('_gather'):
                # we sum only the same variable: condition endswith and only
                # the direct children condition len(split)==2
                sub_input_dict = {key: value for key, value in input_dict.items(
                ) if key.endswith(f'.{input_to_sum}') and len(key.split('.')) == 2}

                if len(list(sub_input_dict.keys())) >= 2:

                    if type_input in ['float', 'int', 'array']:
                        output_dict[input_to_sum] = sum(
                            sub_input_dict.values())
                    elif type_input == 'dict':
                        sum_dict = {}

                        ref_dict = list(sub_input_dict.values())[0]

                        for ref_key, ref_value in ref_dict.items():
                            check_key = [False for sub_dict in sub_input_dict.values()
                                         if ref_key not in list(sub_dict.keys())]

                            if isinstance(ref_value, tuple(self.STANDARD_TYPES)) and check_key == []:
                                values_to_sum = [sub_input_dict[input_key][ref_key]
                                                 for input_key in sub_input_dict]
                                # check_types
                                if all(isinstance(sub, type(values_to_sum[0])) for sub in values_to_sum[1:]):
                                    sum_dict[ref_key] = sum(values_to_sum)
                                else:
                                    self.logger.info(
                                        f'Can not sum values {values_to_sum} for {ref_key} between these dictionaries {list(sub_input_dict.keys())} ')
                            else:
                                self.logger.info(
                                    f'Can not sum {ref_key} between these dictionaries {list(sub_input_dict.keys())} ')
                        output_dict[input_to_sum] = sum_dict

                    elif type_input == 'dataframe':

                        if input_to_sum != 'percentage_resource':
                            sum_df, percent = toolbox.compute_sum_df(
                                [sub_input_dict[input_key] for input_key in sub_input_dict], not_sum=['years', 'Quarters'])
                            output_dict[input_to_sum] = sum_df

                            if 'percentage_resource' in self.input_to_sum.keys():
                                tmp_list = list(sub_input_dict.keys())
                                percent.columns = [
                                    'years'] + [tmp.split(".", 1)[0] for tmp in tmp_list]
                                output_dict['percentage_resource'] = percent

                elif len(sub_input_dict) == 1:
                    output_dict[input_to_sum] = list(
                        sub_input_dict.values())[0]
                else:
                    raise Exception(
                        f'No input values are present to sum the variable {input_to_sum} in discipline {self.sos_name} ')
        self.store_sos_outputs_values(output_dict)
