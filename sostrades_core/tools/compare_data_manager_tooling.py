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
from numpy import ndarray

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Tooling methods to compare data manager content
"""
from collections.abc import MutableMapping
from pandas.core.frame import DataFrame
from pandas.testing import assert_frame_equal
from contextlib import suppress
from sostrades_core.execution_engine.namespace import Namespace
from pandas.core.indexes.base import Index


def dict_are_equal(d1, d2):
    '''
    Use compare_dict method to return True/False if d1 and d2 are/aren't equals
    '''
    return compare_dict(d1, d2, tree='', error=None, df_equals=True)


def compare_dict(d1, d2, tree, error, df_equals=False):
    '''
    Compare all elements in two dicts d1, d2 (the dicts should have the same structure)
    If Error is None, return False as soon as two different elements are identified, return True if all elements are equal
    If error is a dict, store differences in error
    '''
    for key in d1.keys():
        try:
            # if d1[key] or d2[key] is None
            if bool(d1.get(key) is None) != bool(d2.get(key) is None):
                if error is None:
                    return False
                else:
                    error.update(
                        {tree: f'{d1.get(key)} and {d2.get(key)} have different types'})
            # if type is dict
            elif isinstance(d1.get(key), dict):
                dict_equal = compare_dict(d1.get(key), d2.get(key),
                             '.'.join([tree, str(key)]), error)
                if dict_equal is False:
                    return False
            # if type is dataframe
            elif isinstance(d1.get(key), DataFrame):
                dataframe_equal = compare_dataframes(d1.get(key), d2.get(key), '.'.join([tree, str(key)]), error, df_equals)
                if dataframe_equal is False:
                    return False
            # if type is tuple
            elif isinstance(d1.get(key), tuple):
                tuple_equal = parse_tuple_to_compare(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error)
                if tuple_equal is False:
                    return False
            # if type is list or array
            elif isinstance(d1.get(key), (list, Index, ndarray)):
                list_equal = parse_list_to_compare(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error, df_equals)
                if list_equal is False:
                    return False
            else:
                elem_equal = compare_elements(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error)
                if elem_equal is False:
                    return False

        # return False or store exception
        except Exception as e:
            if error is None:
                return False
            else:
                error.update(
                    {
                        tree: f'\nProblem parsing a dictionnary:\n Dict structure differ on {key}:\n Trying to compare {type(d1.get(key))} with {type(d2.get(key))}'})
    
    # check keys equality
    if len(set(d2.keys()).difference(set(d1.keys()))) != 0:
        if error is None: 
            return False
        else:
            error.update(
                {tree: f'd2 keys {set(d2.keys()).difference(set(d1.keys()))} don\'t exist in d1 keys'})

    # if no difference has been detected, return True
    if error is None:
        return True


def parse_list_to_compare(list_1, list_2, tree, error, df_equals=False):
    if len(list_1) != len(list_2):
        if error is None:
            return False
        else:
            error.update(
                {tree: f'\nProblem parsing a list:\n The length of the compared lists differs'})
    else:
        try:
            current = 0
            while current < len(list_1):
                if isinstance(list_1[current], MutableMapping):
                    dict_equal = compare_dict(list_1[current], list_2[current], tree, error)
                    if dict_equal is False:
                        return False
                elif isinstance(list_1[current], DataFrame):
                    dataframe_equal = compare_dataframes(list_1[current], list_2[current], tree, error, df_equals)
                    if dataframe_equal is False:
                        return False
                elif isinstance(list_1[current], tuple):
                    tuple_equal = parse_tuple_to_compare(list_1[current], list_2[current], tree, error)
                    if tuple_equal is False:
                        return False
                elif isinstance(list_1[current], (list, Index, ndarray)):
                    list_equal = parse_list_to_compare(
                        list_1[current], list_2[current], tree, error)
                    if list_equal is False:
                        return False
                else:
                    elem_equal = compare_elements(list_1[current], list_2[current], tree, error)
                    if elem_equal is False:
                        return False
                current += 1

        except Exception as e:
            if error is None:
                return False
            else:
                error.update(
                    {tree: f'\nProblem parsing a list:\n List structure differ in {tree}'})


def parse_tuple_to_compare(tuple1, tuple2, tree, error):
    try:
        for tup in tuple1:
            if isinstance(tup, dict):
                index = tuple1.index(tup)
                dict_equal = compare_dict(tup, tuple2[index], tree, error)
                if dict_equal is False:
                    return False
            elif isinstance(tup, list):
                index = tuple1.index(tup)
                list_equal = parse_list_to_compare(tup, tuple2[index], tree, error)
                if list_equal is False:
                    return False
            else:
                index = tuple1.index(tup)
                elem_equal = compare_elements(tup, tuple2[index], tree, error)
                if elem_equal is False:
                    return False
    except Exception as e:
        if error is None:
            return False
        else:
            error.update(
                {tree: f'\nProblem parsing a tuple:\n tuple structure differ in {tree}'})



def compare_dataframes(df1, df2, tree, error, df_equals):
    """
    Compare two dataframes and raise an exception if not equal
    """
    if df_equals:
        if not df1.equals(df2):
            if error is None:
                return False
            else:
                error.update(
                    {tree: f'Dataframes {df1} and {df2} don\'t match'})
    else:
        try:
            # check if array in dataframe
            list_col_containing_arrays = [col for col in df1.columns if any(isinstance(x, (ndarray, list)) for x in df1[col])]
            if list_col_containing_arrays == []:
                assert_frame_equal(df1, df2, rtol=1e-3)
            else:
                # if at least one column of dataframe has array or list, compare these columns using method of list comparison, compare other columns 
                for col in list_col_containing_arrays:
                    parse_list_to_compare(df1[col].values, df2[col].values, tree, error)
                df1_wo_array_columns = df1.drop(list_col_containing_arrays, axis=1)
                df2_wo_array_columns = df2.drop(list_col_containing_arrays, axis=1)
                assert_frame_equal(df1_wo_array_columns, df2_wo_array_columns, rtol=1e-3)
                        
        except Exception as e:
            if error is None:
                return False
            else:
                error.update(
                    {tree: f'{e} don\'t match'})


def compare_elements(el1, el2, tree, error):
    """
    Compare two elements and raise an exception if not equal
    Caution: add force the type of (el1 == el2) to standard bool to avoid a numpy.bool case
    """
    if bool(el1 == el2) is False:
        if error is None:
            return False
        else:
            error.update({tree: f'{el1} and {el2} don\'t match'})


def delete_keys_from_dict(dictionary):
    '''
        Delete key refering to a namespace object in the dictionary/sub-lists/sub-dict
        :params: dictionary, dictionary where to delete the sos namespace ref
        :type: Dict
    '''
    for key in dictionary.keys():
        with suppress(KeyError):
            if isinstance(dictionary[key], Namespace):
                dictionary.update({key: None})
            if key in ['model_origin', 'disciplines_dependencies']:
                dictionary.update({key: None})
            if isinstance(key, str) and 'delta_' in key:
                dictionary.update({key: None})
    for value in dictionary.values():
        if isinstance(value, MutableMapping):
            delete_keys_from_dict(value)
        elif isinstance(value, list):
            parse_list(value)


def parse_list(list_1):
    '''
        Parse list to find other list or dict to delete
        namespace references
        :params: list_1, list to parse to delete namespace ref in sub-dictionary
        :type: List
    '''
    current = 0
    while current < len(list_1):
        if isinstance(list_1[current], MutableMapping):
            delete_keys_from_dict(list_1[current])
        elif isinstance(list_1[current], list):
            parse_list(list_1[current])
        current += 1
