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
'''
Created on 27 july 2022

@author: NG9430A
'''
import numpy as np


def get_inputs_for_path(input_parameter, PATH, parent_path_admissible=False):
    filtered_input_parameter = None
    if 'PATH' in input_parameter:
        while filtered_input_parameter is None and len(PATH) > 0:
            if PATH in input_parameter['PATH'].unique():
                filtered_input_parameter = input_parameter.loc[
                    input_parameter['PATH'] == PATH
                ]
            else:
                if parent_path_admissible:
                    PATH = get_parent_path(PATH)
                else:
                    raise Exception(
                        f'the path {PATH} is not found in PATH input_parameter column'
                    )
    else:
        raise Exception("Can not find parent path")
    if filtered_input_parameter is None:
        raise Exception("the column 'PATH' is not found as an input_parameter column")
    else:
        return filtered_input_parameter.reset_index(drop=True)


def get_parent_path(PATH):
    path_list = PATH.split('.')
    path_list.pop()
    if len(path_list) > 0:
        return '.'.join(path_list)
    else:
        return path_list
