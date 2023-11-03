'''
Copyright 2023 Capgemini

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

            
   