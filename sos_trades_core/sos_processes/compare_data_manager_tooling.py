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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Tooling methods to compare data manager content
"""
from collections.abc import MutableMapping
from pandas.core.frame import DataFrame
from pandas.testing import assert_frame_equal
from contextlib import suppress
from sos_trades_core.execution_engine.namespace import Namespace


def compare_dict(d1, d2, tree, error, df_equals=False, print_error=False):
    '''
    Compare all elements in two dicts.
    The dicts should have the same structure
    '''
    for key in d1.keys():
        try:

            if bool(d1.get(key) is None) != bool(d2.get(key) is None):
                error.update(
                    {tree: f'{d1.get(key)} and {d2.get(key)} have different types'})
            elif isinstance(d1.get(key), dict):
                compare_dict(d1.get(key), d2.get(key),
                             '.'.join([tree, str(key)]), error)
            elif isinstance(d1.get(key), DataFrame):
                if df_equals:
                    if not d1.get(key).equals(d2.get(key)):
                        error.update(
                            {tree: f'Dataframes {d1.get(key)} and {d2.get(key)} don\'t match'})
                else:
                    try:
                        assert_frame_equal(d1.get(key), d2.get(
                            key), rtol=1e-3)
                    except Exception as e:
                        error.update(
                            {'.'.join([tree, str(key)]): f'{e} don\'t match'})
            elif isinstance(d1.get(key), list):
                parse_list_to_compare(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error)
            elif isinstance(d1.get(key), tuple):
                parse_tuple_to_compare(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error)
            else:
                compare_elements(d1.get(key), d2.get(
                    key), '.'.join([tree, str(key)]), error)
            if print_error:
                print(key, error)
        except Exception as e:
            error.update(
                {tree: f'\nProblem parsing a dictionnary:\n Dict structure differ on {key}:\n Trying to compare {type(d1.get(key))} with {type(d2.get(key))}'})
    if len(set(d2.keys()).difference(set(d1.keys()))) != 0:
        error.update(
            {tree: f'd2 keys {set(d2.keys()).difference(set(d1.keys()))} don\'t exist in d1 keys'})


def parse_list_to_compare(list_1, list_2, tree, error):
    try:
        current = 0
        while current < len(list_1):
            if isinstance(list_1[current], MutableMapping):
                compare_dict(list_1[current], list_2[current], tree, error)
            elif isinstance(list_1[current], list):
                parse_list_to_compare(
                    list_1[current], list_2[current], tree, error)
            else:
                compare_elements(list_1[current], list_2[current], tree, error)
            current += 1
    except Exception as e:
        error.update(
            {tree: f'\nProblem parsing a list:\n List structure differ in {tree}'})


def parse_tuple_to_compare(tuple1, tuple2, tree, error):
    try:
        for tup in tuple1:
            if isinstance(tup, dict):
                index = tuple1.index(tup)
                compare_dict(tup, tuple2[index], tree, error)
            elif isinstance(tup, list):
                index = tuple1.index(tup)
                parse_list_to_compare(tup, tuple2[index], tree, error)
            else:
                index = tuple1.index(tup)
                compare_elements(tup, tuple2[index], tree, error)
    except Exception as e:
        error.update(
            {tree: f'\nProblem parsing a tuple:\n tuple structure differ in {tree}'})


def compare_elements(el1, el2, tree, error):
    '''
    compare two elements and raise an exception if not equal
    '''
    if (el1 == el2) is False:
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
