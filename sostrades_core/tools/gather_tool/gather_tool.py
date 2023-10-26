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

            
   