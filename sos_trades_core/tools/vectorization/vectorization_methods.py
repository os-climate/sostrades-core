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
import pandas as pd


def get_parent_path(PATH):
    path_list = PATH.split('.')
    path_list.pop()
    if len(path_list) > 0:
        return '.'.join(path_list)
    else:
        return path_list


def merge_df_dict_with_path(df_dict):
    df_with_path = pd.DataFrame({})
    for key, val in df_dict.items():
        val['PATH'] = key
        df_with_path = df_with_path.append(val, ignore_index=True)

    return df_with_path


def compute_parent_path_sum(df, path, based_on, columns_not_to_sum):
    all_columns = df.columns.to_list()
    col_to_sum = [val for val in all_columns if val not in columns_not_to_sum]
    filtered_df = df.loc[df['PATH'].str.startswith(path)]
    df_with_col_sum = filtered_df.groupby(based_on)[col_to_sum].sum().reset_index()
    df_with_col_not_sum = filtered_df[columns_not_to_sum]
    df_merged = pd.merge(
        df_with_col_sum, df_with_col_not_sum, on=based_on, how='left'
    ).drop_duplicates()
    df_merged['PATH'] = path
    df_merged = df_merged[all_columns]
    df = df.append(df_merged)
    return df


# check compute sum of val for each possible parent paths except for columns_not_to_sum (with sum based on)
def check_compute_parent_path_sum(df, columns_not_to_sum, based_on):
    path_list = df['PATH'].unique().tolist()
    for path in path_list:
        path = get_parent_path(path)
        if path in path_list or len(path) == 0:
            pass
        else:
            path_list.append(path)
            df = compute_parent_path_sum(df, path, based_on, columns_not_to_sum)
    df = df.sort_values(["PATH"] + [based_on])
    return df.reset_index(drop=True)


# return input_parameter filtered on PATH parameter
# if parent_path_admissible, return iput_parameter filtered on parent path if existing
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
