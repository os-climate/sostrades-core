'''
Copyright 2022 Airbus SAS
Modifications on 2024/02/28 Copyright 2024 Capgemini

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
from logging import Logger

import pandas as pd
import numpy as np
from copy import deepcopy

# from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

BREAKDOWN_COLUMN = 'PATH'


def get_parent_path(PATH):
    path_list = PATH.split('.')
    path_list.pop()
    if len(path_list) > 0:
        return '.'.join(path_list)
    else:
        return path_list


def merge_df_dict_with_path(
        df_dict: dict, path_name: str = BREAKDOWN_COLUMN
) -> pd.DataFrame:
    """Method to merge a dictionary of dataframe into a single dataframe.
    A new column for the aircraft_breakdown is created and the dictionary key is used as the value for the resulting dataframe

    Args:
        df_dict (dict): dictionary of dataframe. All dataframe must have identical columns

    Returns:
        pd.DataFrame: merged dataframe with a new column BREAKDOWN_COLUMN with the dict key as value
    """
    df_with_path = pd.DataFrame({})
    for key, df in df_dict.items():
        # dict of dict
        df_copy = df.copy(deep=True)
        df_copy.insert(0, path_name, key)

        df_with_path = pd.concat([df_with_path, df_copy], ignore_index=True)

    return df_with_path.fillna(0.0)


def compute_parent_path_sum(df, path, based_on, columns_not_to_sum):
    all_columns = df.columns.to_list()
    col_to_sum = [val for val in all_columns if val not in columns_not_to_sum]
    filtered_df = df.loc[df[BREAKDOWN_COLUMN].str.startswith(path)]
    df_with_col_sum = filtered_df.groupby(based_on)[col_to_sum].sum().reset_index()
    df_with_col_not_sum = filtered_df[columns_not_to_sum]
    df_merged = pd.merge(
        df_with_col_sum, df_with_col_not_sum, on=based_on, how='left'
    ).drop_duplicates()
    df_merged[BREAKDOWN_COLUMN] = path
    df_merged = df_merged[all_columns]
    df = df.append(df_merged)
    return df


# check compute sum of val for each possible parent paths except for columns_not_to_sum (with sum based on)
def check_compute_parent_path_sum(df, columns_not_to_sum, based_on):
    path_list = df[BREAKDOWN_COLUMN].unique().tolist()
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
def get_inputs_for_path(
        input_parameter: pd.DataFrame,
        PATH: str,
        parent_path_admissible: bool = False,
        unique_value: bool = False,
        allow_empty_dataframe: bool = False,
        parameter_name: str = '',
):
    search_path = PATH
    filtered_input_parameter = None
    if BREAKDOWN_COLUMN in input_parameter:
        while filtered_input_parameter is None and len(PATH) > 0:
            if PATH in input_parameter[BREAKDOWN_COLUMN].unique():
                filtered_input_parameter = input_parameter.loc[
                    input_parameter[BREAKDOWN_COLUMN] == PATH
                    ]
            else:

                if parent_path_admissible:
                    PATH = get_parent_path(PATH)
                elif allow_empty_dataframe:
                    filtered_input_parameter = pd.DataFrame(
                        columns=input_parameter.columns
                    )
                else:
                    raise Exception(
                        (
                            f'the path {PATH} is not found for parameter {parameter_name}'
                            f' but is required for the calculation. Please update it.'
                        )
                    )
    else:
        raise Exception(
            f"The column {BREAKDOWN_COLUMN} is not found as an input_parameter column in parameter {parameter_name}"
        )
    if filtered_input_parameter is None:
        raise Exception(
            f"Can not find parent path of {search_path} for parameter {parameter_name}"
        )
    else:
        filtered_input_parameter = filtered_input_parameter.drop(
            columns=['PATH']
        ).reset_index(drop=True)
        # check if result dataframe is only one column
        if filtered_input_parameter.shape[1] == 1:
            # in that case, return only the column values as a list
            values_list = filtered_input_parameter.iloc[:, 0].values.tolist()
            if unique_value:
                return values_list[0]
            else:
                return values_list
        else:
            return filtered_input_parameter


def change_column_values_delimiter(
        data_df: pd.DataFrame,
        column_name: list = [BREAKDOWN_COLUMN],
        old_delimiter: str = '.',
        new_delimiter: str = ' ↦ ',
):
    for col in column_name:
        for g in data_df[col].unique():
            data_df.loc[data_df[col] == g, col] = f'{new_delimiter}'.join(
                g.split(old_delimiter)
            )
    return data_df


def check_aircraft_breakdown_in_inputs(
        inputs_dict: dict,
        parameters_dict: dict,
        aircraft_breakdown_tree: list,
        logger: Logger,
        sos_discipline: SoSWrapp,
):
    aircraft_breakdown_list = flatten(nested_dict=aircraft_breakdown_tree)
    for param_name, conf_dict in parameters_dict.items():
        parameter = inputs_dict.get(param_name, None)
        if parameter is not None:
            if BREAKDOWN_COLUMN in parameter:
                for aircraft_breakdown in aircraft_breakdown_list:
                    if aircraft_breakdown not in parameter[BREAKDOWN_COLUMN].values:
                        logger.warning(
                            f'Aircraft Breakdown {aircraft_breakdown} is missing for parameter {param_name} for discipline {sos_discipline.get_disc_full_name()}'
                        )
            else:
                logger.warning(
                    f'Parameter {param_name} does not have the column {BREAKDOWN_COLUMN}, impossible to check if all values are present for discipline {sos_discipline.get_disc_full_name()}'
                )

        else:
            logger.warning(
                f'Parameter {param_name} is not in input dictionary for discipline {sos_discipline.get_disc_full_name()}'
            )


def flatten(
        nested_dict: dict, flatten_list: list = None, parent_key: str = None
) -> list:
    if flatten_list is None:
        flatten_list = []
    for key, sub_dict in sorted(nested_dict.items()):
        if parent_key is None:
            full_key = key
        else:
            full_key = f'{parent_key}.{key}'
        flatten_list.append(full_key)
        if sub_dict:
            flatten(sub_dict, flatten_list, parent_key=full_key)
    return flatten_list


def generate_breakdown_by_level(aircraft_breakdown_tree: list) -> dict:
    breakdown_by_level = {'calculate': [], 'sum': {}}
    aircraft_breakdown_list = flatten(nested_dict=aircraft_breakdown_tree)
    key_with_children = {}
    for key in aircraft_breakdown_list:
        sub_keys_list = [
            k
            for k in aircraft_breakdown_list
            if key in k and len(k.split('.')) == len(key.split('.')) + 1
        ]
        key_with_children[key] = sub_keys_list

    breakdown_by_level['calculate'] = [
        key for key, children_list in key_with_children.items() if children_list == []
    ]
    unsorted_sum_dict = {
        key: children_list
        for key, children_list in key_with_children.items()
        if children_list != []
    }

    breakdown_by_level['sum'] = {
        key: unsorted_sum_dict[key]
        for key in sorted(
            unsorted_sum_dict.keys(), key=lambda x: len(x.split('.')), reverse=True
        )
    }

    return breakdown_by_level


def compute_sum_df(
        df_dict: dict,
        parent_key: str,
        children_keys: list,
        columns_not_to_sum: list = [],
):
    """
    Method to compute sum of dict of dataframes
    not_sum : column name to not sum
    """
    # infer not summable columns in dataframe

    df_dict = deepcopy(df_dict)

    not_summable_nested_list = [
        df.convert_dtypes()
        .select_dtypes(exclude=[np.number, 'datetime'])
        .columns.to_list()
        for df in df_dict.values()
        if not df.empty
    ]
    not_summable = [l for sublist in not_summable_nested_list for l in sublist]
    if len(not_summable):
        if columns_not_to_sum is None:
            columns_not_to_sum = not_summable
        else:
            columns_not_to_sum.extend(not_summable)
            columns_not_to_sum = list(set(columns_not_to_sum))

    df_sum = None
    for key in children_keys:
        # check coherency between parent_key and children keys
        if parent_key not in key:
            raise Exception(
                f'Breakdown {key} is not a children of {parent_key}. The sum cannot be applied.'
            )
        col_split = key.split('.')
        if len(col_split) != len(parent_key.split('.')) + 1:
            raise Exception(
                f'Breakdown {key} is not a direct children of {parent_key}. The sum cannot be applied.'
            )
        if key not in df_dict.keys():
            raise Exception(
                f'Children key {key} is not in specified dict of DataFrame to sum.'
            )
        if df_dict[key].empty:
            df_dict.pop(key)
        else:
            df_to_sum = df_dict[key]
            in_columns = [
                col for col in df_to_sum.columns if col not in columns_not_to_sum
            ]
            filtered_df_to_sum = df_to_sum[in_columns]
            if df_sum is None:
                df_sum = filtered_df_to_sum.copy(deep=True)
            else:
                df_sum = df_sum.add(filtered_df_to_sum, fill_value=0.0)

    # restore column not to sum in result sum df
    if columns_not_to_sum != []:
        if children_keys[0] in df_dict:
            df_restore_base = df_dict[children_keys[0]]
        else:
            df_restore_base = list(df_dict.values())[0]
        restore_columns = [
            col for col in df_restore_base.columns if col in columns_not_to_sum
        ]
        df_restore = df_restore_base[restore_columns]
        df_sum = df_restore.join(df_sum)

    return df_sum
