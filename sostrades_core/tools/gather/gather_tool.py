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

import pandas as pd


def gather_selected_outputs(eval_outputs, gather_suffix):
    """
    get selected output from the eval_output variable 
    :param eval_outputs: dataframe containing the outputs with the columns:
                            - selected_output
                            - full_name
                            - output_name
    :param gather_suffix: string to add after the output if the name is not defined 
                            so that it as a different name from the input
    :return: final_out_names. final_in_names is a dict of the name of the selected_output as key and 
                the output name as value
    """
    final_out_names = {}
    if eval_outputs is not None:
        selected_outputs = eval_outputs[eval_outputs['selected_output'] == True]['full_name'].tolist()
        if 'output_name' in eval_outputs.columns:
            eval_out_names = eval_outputs[eval_outputs['selected_output']== True]['output_name'].tolist()
        else:
            eval_out_names = [None for _ in selected_outputs]

        
        for out_var, out_name in zip(selected_outputs, eval_out_names):
            _out_name = out_name or f'{out_var}{gather_suffix}'
            final_out_names[out_var] = _out_name
    return final_out_names

def get_eval_output(possible_out_values, eval_output_dm):
    error_msg = ''
    default_dataframe = None
    if possible_out_values:
        possible_out_values = list(possible_out_values)
        possible_out_values.sort()
        default_dataframe = pd.DataFrame({'selected_output': [False for _ in possible_out_values],
                                              'full_name': possible_out_values,
                                               'output_name': [None for _ in possible_out_values]})


        # check if the eval_inputs need to be updated after a subprocess configure
        if eval_output_dm is not None and (set(eval_output_dm['full_name'].tolist()) != (set(default_dataframe['full_name'].tolist())) \
         or eval_output_dm['selected_output'].tolist() != (default_dataframe['selected_output'].tolist()) \
         or eval_output_dm['output_name'].tolist() != (default_dataframe['output_name'].tolist())):
            error_msg = check_eval_io(eval_output_dm['full_name'].tolist(),
                                   default_dataframe['full_name'].tolist(),
                                   is_eval_input=False)
            already_set_names = eval_output_dm['full_name'].tolist()
            already_set_values = eval_output_dm['selected_output'].tolist()
            if 'output_name' in eval_output_dm.columns:
                # TODO: maybe better to repair tests than to accept default, in particular for data integrity check
                already_set_out_names = eval_output_dm['output_name'].tolist()
            else:
                already_set_out_names = [None for _ in already_set_names]
            for index, name in enumerate(already_set_names):
                default_dataframe.loc[default_dataframe['full_name'] == name,
                ['selected_output', 'output_name']] = \
                    (already_set_values[index], already_set_out_names[index])
    return default_dataframe, error_msg

            

def check_eval_io(given_list, default_list, is_eval_input):
    """
    Set the evaluation variable list (in and out) present in the DM
    which fits with the eval_in_base_list filled in the usecase or by the user
    """
    error_msg = ''
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'
    for given_io in given_list:
        if given_io not in default_list and not MULTIPLIER_PARTICULE in given_io:
            if is_eval_input:
                error_msg = f'The input {given_io} in eval_inputs is not among possible values. Check if it is an ' \
                            f'input of the subprocess with the correct full name (without study name at the ' \
                            f'beginning) and within allowed types (int, array, float). Dynamic inputs might  not ' \
                            f'be created. should be in {default_list} '

            else:
                error_msg = f'The output {given_io} in eval_outputs is not among possible values. Check if it is an ' \
                            f'output of the subprocess with the correct full name (without study name at the ' \
                            f'beginning). Dynamic inputs might  not be created. should be in {default_list}'

    return error_msg