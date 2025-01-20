'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/08/01 Copyright 2023 Capgemini

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
import logging
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Union
from uuid import uuid4

import pandas as pd
from gemseo.caches.simple_cache import SimpleCache
from pandas import concat

from sostrades_core.datasets.dataset_manager import DatasetsManager
from sostrades_core.datasets.dataset_mapping import DatasetsMapping
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.compare_data_manager_tooling import compare_dataframes, dict_are_equal
from sostrades_core.tools.ontology_variables.ontology_variable_key import create_data_key
from sostrades_core.tools.tree.serializer import DataSerializer
from sostrades_core.tools.tree.treeview import TreeView

TYPE = ProxyDiscipline.TYPE
VALUE = ProxyDiscipline.VALUE
RANGE = ProxyDiscipline.RANGE
ORIGIN = ProxyDiscipline.ORIGIN
DEFAULT = ProxyDiscipline.DEFAULT
OPTIONAL = ProxyDiscipline.OPTIONAL
COUPLING = ProxyDiscipline.COUPLING
EDITABLE = ProxyDiscipline.EDITABLE
IO_TYPE = ProxyDiscipline.IO_TYPE
UNIT = ProxyDiscipline.UNIT
IO_TYPE_IN = ProxyDiscipline.IO_TYPE_IN
IO_TYPE_OUT = ProxyDiscipline.IO_TYPE_OUT
COMPOSED_OF = ProxyDiscipline.COMPOSED_OF
NS_REFERENCE = ProxyDiscipline.NS_REFERENCE
POSSIBLE_VALUES = ProxyDiscipline.POSSIBLE_VALUES
INTERNAL_VISIBILITY = ProxyDiscipline.INTERNAL_VISIBILITY
DISCIPLINES_DEPENDENCIES = ProxyDiscipline.DISCIPLINES_DEPENDENCIES
VAR_NAME = ProxyDiscipline.VAR_NAME
DATAFRAME_DESCRIPTOR = ProxyDiscipline.DATAFRAME_DESCRIPTOR
DATAFRAME_EDITION_LOCKED = ProxyDiscipline.DATAFRAME_EDITION_LOCKED
TYPE_METADATA = ProxyDiscipline.TYPE_METADATA
DISPLAY_NAME = ProxyDiscipline.DISPLAY_NAME


@dataclass()
class ParameterChange:
    """
    Dataclass used to log parameter changes when configuring the data in a study.
    """
    parameter_id: str
    variable_type: str
    old_value: Any
    new_value: Any
    connector_id: Union[str, None]
    dataset_id: Union[str, None]
    dataset_parameter_id: Union[str, None]
    date: datetime
    dataset_data_path: str
    variable_key: str


class DataManagerException(Exception):
    pass


class DataManager:
    """
    Specification: DataManager class collects inputs/outputs and disciplines
    """
    VALUE = VALUE
    DISC_REF = 'reference'
    STATUS = 'status'

    def __init__(self, name,
                 logger: logging.Logger,
                 root_dir=None,
                 rw_object=None,
                 study_filename=None,
                 ns_manager=None):
        '''
        Constructor
        '''
        self.no_change = True
        self.name = name
        self.rw_object = rw_object
        self.root_dir = root_dir
        self.study_filename = study_filename
        self.ns_manager = ns_manager
        self.data_dict = None
        self.data_id_map = None
        self.disciplines_dict = None
        self.disciplines_id_map = None
        self.gemseo_disciplines_id_map = None
        self.cache_map = None
        self.treeview = None
        self.reduced_dm = None
        self.reset()
        self.data_check_integrity = False
        self.logger = logger

        self.dataset_manager = DatasetsManager(logger=logger.getChild("DatasetsManager"))

    @staticmethod
    def get_an_uuid():
        ''' generate a random UUID to make data_dict keys unique '''
        return str(uuid4())

    def generate_hashed_uid(self, string_list):
        '''
        Generate a hashed uid based on string list containing disc infos (full disc name, class name and full data i/o)
        '''
        h = sha256()
        for string in string_list:
            h.update(string.encode())
        return h.digest()

    def reinit_cache_map(self):
        self.cache_map = {}
        self.gemseo_disciplines_id_map = {}

    def fill_cache_map(self, disc_info_list, disc):
        '''
        Fill the cache_map for the dump/load cache method
        '''
        hashed_uid = self.generate_hashed_uid(disc_info_list)
        self.cache_map[hashed_uid] = disc.cache
        # store disc in DM map
        if hashed_uid in self.gemseo_disciplines_id_map:
            raise DataManagerException(
                'The hashed_uid to fill the cache_map must be unique')
        else:
            self.gemseo_disciplines_id_map[hashed_uid] = disc

    def delete_hashed_id_in_cache_map(self, hashed_uid):

        if hashed_uid in self.cache_map:
            del self.cache_map[hashed_uid]
            del self.gemseo_disciplines_id_map[hashed_uid]

    def fill_gemseo_caches_with_unanonymized_cache_map(self, cache_map):
        '''
        Fill gemseo discipline cache from cache_map using gemseo_disciplines_id_map
        '''
        # update cache of all gemseo disciplines with loaded cache_map
        for disc_id, serialized_cache in cache_map.items():
            if disc_id in self.gemseo_disciplines_id_map:
                disc_cache = self.cache_map[disc_id]
                self.fill_cache_with_serialized_cache(
                    disc_cache, serialized_cache)

    #                 for disc in self.gemseo_disciplines_id_map[disc_id]:
    #                     disc.cache = disc_cache

    def fill_cache_with_serialized_cache(self, empty_cache, serialized_cache):
        '''
        Fill empty cache from prepare execution with the serialized cache (dict) found in the pickle
        '''

        if isinstance(empty_cache, SimpleCache):

            cached_inputs = serialized_cache[1]['inputs']
            cached_outputs = serialized_cache[1]['outputs']
            empty_cache.cache_outputs(cached_inputs,
                                      cached_outputs)
            #
            if 'jacobian' in serialized_cache[1]:
                cached_jac = serialized_cache[1]['jacobian']
                empty_cache.cache_jacobian(cached_inputs, cached_jac)
        else:
            self.logger.error(
                f'Only simple cache dump/load is handled for now but discipline {empty_cache.name} has a cache as {empty_cache.__class__}')

    def reset(self):
        self.data_dict = {}
        self.data_id_map = {}
        self.disciplines_dict = {}
        self.disciplines_id_map = {}
        self.no_check_default_variables = []

    def get_data(self, var_f_name, attr=None):
        ''' Get attr value of var_f_name or all data_dict value of var_f_name (if attr=None)
        '''
        if attr is None:
            return self.data_dict[self.get_data_id(var_f_name)]
        else:
            return self.data_dict[self.get_data_id(var_f_name)].get(attr)

    def delete_complex_in_df_and_arrays(self):

        for dataa in self.data_dict.values():
            if dataa['type'] == 'dataframe' and dataa[self.VALUE] is not None:
                for col in dataa[self.VALUE].columns:
                    dataa[self.VALUE][col] = dataa[self.VALUE][col].values.real
            elif dataa['type'] == 'array':
                try:
                    dataa[self.VALUE] = dataa[self.VALUE].real
                except:
                    pass

    def check_data_in_dm(self, var_f_name):
        '''
        Check if the data is in the DM with its full name

        :params: var_f_name, full variable name to check
        :type : string

        :returns: boolean True or False
        '''
        data_in_dm = False

        if var_f_name in self.data_id_map:
            if self.get_data_id(var_f_name) in self.data_dict:
                data_in_dm = True

        return data_in_dm

    def set_data(self, var_f_name, attr, val, check_value=True):
        ''' Set attr value of var_f_name in data_dict
        '''
        if self.get_data_id(var_f_name) in self.data_dict:
            if check_value:
                if self.data_dict[self.get_data_id(var_f_name)][attr] != val:
                    self.data_dict[self.get_data_id(var_f_name)][attr] = val
                    self.no_change = False
            else:
                self.data_dict[self.get_data_id(var_f_name)][attr] = val
        else:
            msg = f"Try to update metadata of variable {var_f_name} that does"
            msg += " not exists as I/O of any discipline"
            raise KeyError(msg)

    def get_io_data_of_disciplines(self, disciplines):
        ''' get i/o value and metadata of provided disciplines
        '''
        data = {}
        data[VALUE] = {}
        data[TYPE_METADATA] = {}
        data["local_data"] = {}
        for d in disciplines:
            # input values and metadata
            var_list = d.get_data_io_dict_keys(io_type=IO_TYPE_IN)
            for v in var_list:
                fname = d.get_var_full_name(v, d.get_data_in())
                data[VALUE][fname] = self.get_data(fname, VALUE)
                data[TYPE_METADATA][fname] = self.get_data(
                    fname, TYPE_METADATA)
            # output values and metadata
            var_list = d.get_data_io_dict_keys(io_type=IO_TYPE_OUT)
            for v in var_list:
                fname = d.get_var_full_name(v, d.get_data_out())
                data[VALUE][fname] = self.get_data(fname, VALUE)
                data[TYPE_METADATA][fname] = self.get_data(
                    fname, TYPE_METADATA)
            # local data update
            data["local_data"].update(
                d.discipline_wrapp.discipline.io.data)
        return data

    def get_value(self, var_f_name):
        ''' Get value of var_f_name from data_dict
        '''
        return self.get_data(var_f_name, ProxyDiscipline.VALUE)

    def get_discipline(self, disc_id):
        ''' Get discipline with disc_id from disciplines_dict
        '''
        if disc_id in self.disciplines_dict:
            return self.disciplines_dict[disc_id][self.DISC_REF]
        else:
            return None

    def get_disciplines_with_name(self, disc_f_name):
        ''' Get discipline with disc_id from disciplines_dict
        '''
        disc_id_list = self.get_discipline_ids_list(disc_f_name)

        disc_list = [self.disciplines_dict[disc_id][self.DISC_REF] for disc_id in disc_id_list]

        return disc_list

    def get_discipline_names_with_starting_name(self, starting_name):
        '''
        Get all disciplines that starts with starting_name in the datamanager
        '''
        disc_list = [disc_name for disc_name in self.disciplines_id_map if disc_name.startswith(starting_name)]

        return disc_list

    def get_all_namespaces_from_var_name(self, var_name):
        ''' Get all namespaces containing var_name in data_dict
        '''
        namespace_list = [key for key in self.data_id_map.keys() if key.endswith(f'.{var_name}')]

        return namespace_list

    def get_all_var_name_with_ns_key(self, var_name):
        ''' Get all namespaces containing var_name in data_dict plus their namespace key as a dict
        '''
        namespace_list = [key for key in self.data_id_map.keys() if key.endswith('.' + var_name)]
        if len(namespace_list) > 0:
            ns_dict_obj = self.get_data_dict_attr('ns_reference')
            return {ns: ns_dict_obj[ns].name for ns in namespace_list}
        else:
            return {}

    def get_data_id(self, var_f_name):
        ''' Get data id with var_f_name
        '''
        return self.data_id_map[var_f_name]

    def get_discipline_ids_list(self, disc_f_name):
        ''' Get discipline id list with disc_f_name
        '''
        return self.disciplines_id_map[disc_f_name]

    def generate_data_id_map(self):
        ''' Generate data_id_map with data_dict
        '''
        self.data_id_map = {}
        data_dict = copy(self.data_dict)
        for var_id in data_dict.keys():
            var_f_name = self.get_var_full_name(var_id)
            self.data_id_map[var_f_name] = var_id

    def generate_disciplines_id_map(self):
        ''' Generate disciplines_id_map with disciplines_dict
        '''
        self.disciplines_id_map = {}
        for disc_id in self.disciplines_dict.keys():
            disc_f_name = self.get_disc_full_name(disc_id)
            self.add_disc_id_to_disc_id_map(disc_f_name, disc_id)

    def set_values_from_dict(self, values_dict, full_ns_keys=True):
        ''' Set values in data_dict from dict with namespaced keys
            if full_ns_keys (not uuid), try to get its uuid correspondency through get_data_id function
        '''
        keys_to_map = self.data_id_map.keys() if full_ns_keys else self.data_id_map.values()
        for key, value in values_dict.items():
            if key not in keys_to_map:
                raise ValueError(f'{key} does not exist in data manager')
            k = self.get_data_id(key) if full_ns_keys else key
            # if self.data_dict[k][ProxyDiscipline.VISIBILITY] == INTERNAL_VISIBILITY:
            #     raise Exception(f'It is not possible to update the variable {k} which has a visibility Internal')
            self.data_dict[k][VALUE] = value

    def fill_data_dict_from_dict(self, values_dict: dict[str:Any],
                                 already_set_data: set[str], parameter_changes: list[ParameterChange],
                                 in_vars: bool, init_coupling_vars: bool, out_vars: bool) -> None:
        '''
        Set values in data_dict from dict with namespaced keys

        :param values_dict: Dictionnary {name : values}
        :type values_dict: dict[str:Any]

        :param parameter_changes: list of effective parameter changes filled in place
        :type parameter_changes: list[ParameterChange]

        :param already_set_data: set of data already set
        :type already_set_data: set[str]

        :param in_vars: set in vars
        :type in_vars: bool

        :param init_coupling_vars: init coupling vars
        :type init_coupling_vars: bool

        :param out_vars: set out vars
        :type out_vars: bool
        '''
        # keys of data stored in dumped study file are namespaced, convert them
        # to uuids
        convert_values_dict = self.convert_data_dict_with_ids(values_dict)

        # convert data_dict with uuids
        for key, dm_data in self.data_dict.items():
            if key in convert_values_dict:
                # Only inject key which are set as input
                # Discipline configuration only take care of input variables
                # Variables are only set f
                is_in_var = dm_data[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_IN
                if in_vars and is_in_var and key not in already_set_data:
                    new_value = convert_values_dict[key][VALUE]
                    self.apply_parameter_change(key, new_value, parameter_changes)
                    already_set_data.add(key)
                # check if the key is an output variable
                is_output_var = dm_data[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_OUT
                # check if this is a strongly coupled input necessary to
                # initialize a MDA
                is_init_coupling_var = dm_data[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_IN and dm_data[
                    ProxyDiscipline.COUPLING]
                if ((out_vars and is_output_var) or (
                        init_coupling_vars and is_init_coupling_var)) and key not in already_set_data:
                    new_value = convert_values_dict[key][VALUE]
                    self.apply_parameter_change(key, new_value, parameter_changes)
                    already_set_data.add(key)

    def fill_data_dict_from_datasets(self, datasets_mapping: DatasetsMapping,
                                     already_set_data: set[str], parameter_changes: list[ParameterChange],
                                     in_vars: bool, init_coupling_vars: bool, out_vars: bool
                                     ) -> None:
        '''
        Set values in data_dict from datasets

        :param: datasets_mapping, adatset list and mapping with study namespaces
        :type: DatasetsMapping

        :param already_set_data: set of data already set
        :type already_set_data: set[str]

        :param parameter_changes: list of effective parameter changes filled in place
        :type parameter_changes: list[ParameterChange]

        :param in_vars: set in vars
        :type in_vars: bool

        :param init_coupling_vars: init coupling vars
        :type init_coupling_vars: bool

        :param out_vars: set out vars
        :type out_vars: bool
        '''
        # do a map between namespace and data from data_dict not already fetched
        # to have a list of data by namespace
        namespaced_data_dict = {}

        for key, data_value in self.data_dict.items():
            # check if the key is an output variable
            is_output_var = data_value[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_OUT
            # check if this is a strongly coupled input necessary to
            # initialize a MDA
            is_init_coupling_var = data_value[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_IN and data_value[
                ProxyDiscipline.COUPLING]
            # get all input values not already set
            is_in_var = data_value[ProxyDiscipline.IO_TYPE] == ProxyDiscipline.IO_TYPE_IN
            if (in_vars and is_in_var or (out_vars and is_output_var) or (
                    init_coupling_vars and is_init_coupling_var)) and key not in already_set_data:
                data_ns = data_value[NS_REFERENCE].value
                data_name = data_value[VAR_NAME]
                data_type = data_value[TYPE]

                # create a dict with namespace, datas with keys (to fill dm after) and types (to convert from dataset)
                namespaced_data_dict[data_ns] = namespaced_data_dict.get(data_ns, {DatasetsMapping.KEY: {}, TYPE: {}})
                namespaced_data_dict[data_ns][DatasetsMapping.KEY][data_name] = key
                namespaced_data_dict[data_ns][TYPE][data_name] = data_type

        # iterate on each namespace to retrieve data in this namespace
        for namespace, data_dict in namespaced_data_dict.items():

            datasets_info = datasets_mapping.get_datasets_info_from_namespace(namespace, self.name)
            # retrieve the list of dataset associated to the namespace from the mapping
            if len(datasets_info) > 0:
                # get data values into the dataset into the right format
                updated_data = self.dataset_manager.fetch_data_from_datasets(
                    datasets_info=datasets_info, data_dict=data_dict[TYPE])
                self.logger.warning(f"Retrieved the namespace {namespace} with values: {list(updated_data.keys())}")
                # update data values in dm
                for data_name, new_data in updated_data.items():
                    key = data_dict[DatasetsMapping.KEY][data_name]
                    new_value = new_data[DatasetsManager.VALUE]
                    connector_id = new_data[DatasetsManager.DATASET_INFO].connector_id
                    dataset_id = new_data[DatasetsManager.DATASET_INFO].dataset_info_id
                    dataset_data_name = datasets_info[new_data[DatasetsManager.DATASET_INFO]].get(data_name, data_name)
                    dataset_data_path = self.dataset_manager.get_path_to_dataset_data(
                        new_data[DatasetsManager.DATASET_INFO],
                        dataset_data_name,
                        data_dict[TYPE][data_name])
                    io_type = self.data_dict[key][ProxyDiscipline.IO_TYPE]
                    discipline_name = self.disciplines_dict.get(self.data_dict[key].get('model_origin')).get(
                        'model_name_full_path')
                    variable_key = create_data_key(discipline_name, io_type, data_name)
                    # dataset_parameter_id = new_data[DatasetsManager.DATASET_INFO].dataset_parameter_id
                    self.apply_parameter_change(key, new_value, parameter_changes,
                                                connector_id=connector_id,
                                                dataset_id=dataset_id,
                                                dataset_parameter_id=None,
                                                dataset_data_path=dataset_data_path,
                                                variable_key=variable_key)
                    already_set_data.add(key)
            else:
                self.logger.warning(f"the namespace {namespace} is not referenced in the datasets mapping of the study")

    def apply_parameter_change(self,
                               key: str,
                               new_value: Any,
                               parameter_changes: list[ParameterChange],
                               connector_id: (str, None) = None,
                               dataset_id: (str, None) = None,
                               dataset_parameter_id: (str, None) = None,
                               dataset_data_path: (str, None) = None,
                               variable_key: str = None,
                               update_parameter_changes: bool = True) -> None:
        """
        Applies and logs an input value change on variable with uuid key. It appends to parameter_changes a
        ParameterChange object with deepcopies of the old and new values and the date, if and only if the old and new
        values are different. Updates the data manager value in place.
        :param key: uuid of the variable
        :param new_value: value to update in the dm
        :param parameter_changes: ongoing list of parameter changes
        :param connector_id: dataset connector id if updating from dataset or None otherwise
        :param dataset_id: dataset id if updating from dataset or None otherwise
        :param dataset_parameter_id: id of the parameter in the dataset if updating from dataset or None otherwise  # todo: WIP!
        :param dataset_data_path: path to the parameter in the dataset if updating from dataset or None otherwise
        :param variable_key: ontology key
        :return: None, inplace update of the data manager value for variable
        """
        dm_data = self.data_dict[key]
        if update_parameter_changes:
            old_value = dm_data[VALUE]
            if not dict_are_equal({VALUE: old_value}, {VALUE: new_value}):
                parameter_changes.append(ParameterChange(parameter_id=self.get_var_full_name(key),
                                                         variable_type=dm_data[TYPE],
                                                         old_value=deepcopy(old_value),
                                                         new_value=deepcopy(new_value),
                                                         connector_id=connector_id,
                                                         dataset_id=dataset_id,
                                                         dataset_parameter_id=dataset_parameter_id,
                                                         date=datetime.now(),
                                                         dataset_data_path=dataset_data_path,
                                                         variable_key=variable_key))

        dm_data[VALUE] = new_value

    def export_data_in_datasets(self, datasets_mapping: DatasetsMapping) -> list[ParameterChange]:
        '''
        Set values in datasets from data_dict

        :param: datasets_mapping, adatset list and mapping with study namespaces
        :type: DatasetsMapping

        returns: list of exported data in witch datasets
        '''
        # do a map between namespace and data from data_dict not already fetched
        # to have a list of data by namespace
        exported_parameters = []
        namespaced_data_dict = {}

        for key, data_value in self.data_dict.items():
            data_ns = data_value[NS_REFERENCE].value
            data_name = data_value[VAR_NAME]
            data_type = data_value[TYPE]
            data_value = data_value[VALUE]

            # create a dict with namespace, datas with keys (to fill dataset after), types (to convert in dataset), value (to fill dataset after)
            namespaced_data_dict[data_ns] = namespaced_data_dict.get(data_ns,
                                                                     {DatasetsMapping.KEY: {}, DatasetsMapping.TYPE: {},
                                                                      DatasetsMapping.VALUE: {}})
            namespaced_data_dict[data_ns][DatasetsMapping.KEY][data_name] = key
            namespaced_data_dict[data_ns][DatasetsMapping.TYPE][data_name] = data_type
            namespaced_data_dict[data_ns][DatasetsMapping.VALUE][data_name] = data_value

        # iterate on each datasets to export data in each dataset
        dataset_parameters_mapping, duplicates = datasets_mapping.get_datasets_namespace_mapping_for_study(self.name, namespaces_dict=namespaced_data_dict)
        if len(duplicates) > 0:
            for duplicate, namespace in duplicates.items():
                self.logger.warning(f"The dataset variable {duplicate} has multiple source parameters. \
                                    The last value written in dataset was taken from {namespace}")
        for dataset, mapping_dict in dataset_parameters_mapping.items():
            # retrieve the list of dataset associated to the namespace from the mapping
            if len(mapping_dict[DatasetsMapping.VALUE].keys()) > 0:
                # write data values into the dataset into the right format
                updated_data = self.dataset_manager.write_data_in_dataset(
                    dataset_info=dataset,
                    data_dict=mapping_dict[DatasetsMapping.VALUE],
                    data_type_dict=mapping_dict[DatasetsMapping.TYPE]
                )

                # save which data have been exported
                for data_dataset_name in updated_data.keys():
                    key = mapping_dict[DatasetsMapping.KEY][data_dataset_name]
                    var_type = mapping_dict[DatasetsMapping.TYPE][data_dataset_name]
                    connector_id = dataset.connector_id
                    dataset_id = dataset.dataset_info_id
                    dataset_data_path = self.dataset_manager.get_path_to_dataset_data(dataset,
                                                                                      data_dataset_name,
                                                                                      var_type)
                    # build ontology key
                    io_type = self.data_dict[key][ProxyDiscipline.IO_TYPE]
                    discipline_name = self.disciplines_dict.get(self.data_dict[key].get('model_origin')).get(
                        'model_name_full_path')
                    variable_key = create_data_key(discipline_name, io_type, self.data_dict[key].get('var_name'))

                    exported_parameters.append(ParameterChange(parameter_id=self.get_var_full_name(key),
                                                         variable_type=var_type,
                                                         old_value=deepcopy(mapping_dict[DatasetsMapping.VALUE][data_dataset_name]),
                                                         new_value=None,
                                                         connector_id=connector_id,
                                                         dataset_id=dataset_id,
                                                         dataset_parameter_id=key,
                                                               date=datetime.now(),
                                                               dataset_data_path=dataset_data_path,
                                                               variable_key=variable_key))

        return exported_parameters

    def convert_data_dict_with_full_name(self):
        ''' Return data_dict with namespaced keys
        '''
        return self.convert_dict_with_maps(self.data_dict, self.data_id_map, keys='full_names')

    def convert_data_dict_with_display_name(self, exec_display=False):
        '''
        Return data_dict with display namespaced keys if exec_display is False.
        FOr exec display use standard full names
        '''
        display_data_dict = {}
        for data_id, data_info in self.data_dict.items():
            if exec_display:
                display_name = self.get_var_full_name(data_id)
            else:
                display_name = self.get_var_display_name(data_id)

            data_info[DISPLAY_NAME] = display_name
            display_data_dict[self.get_var_full_name(data_id)] = data_info

        return display_data_dict

    def get_data_dict_values(self, excepted=[]):
        '''
        Return a dictionaries with all full named keys in the dm and the value of each key from the dm
        '''
        return self.get_data_dict_attr(self.VALUE, excepted)

    def get_data_dict_attr(self, attr, excepted=[]):
        '''
        Return a dictionaries with all full named keys in the dm and the value of each key from the dm
        '''
        data_dict = self.convert_data_dict_with_full_name()
        exception_list = []
        if 'numerical' in excepted:
            exception_list = list(ProxyDiscipline.NUM_DESC_IN.keys())

        data_dict_values = {key: value.get(attr, None) for key, value in data_dict.items() if
                            key.split('.')[-1] not in exception_list}

        return data_dict_values

    def get_data_dict_list_attr(self, list_attr, excepted=[]):
        """
         Return a dictionary of dictionary with all full named keys in the dm and the value of each key from the dm
         output : dict[key][attr] for each attr in list_attr
        """
        data_dict_values_dict = {}
        data_dict_values_list = [self.get_data_dict_attr(
            attr, excepted) for attr in list_attr]

        for key in data_dict_values_list[0].keys():
            data_dict_values_dict[key] = {}
            for index, attr in enumerate(list_attr):
                data_dict_values_dict[key][attr] = data_dict_values_list[index][key]

        return data_dict_values_dict

    def convert_data_dict_with_ids(self, dict_to_convert):
        ''' Return data_dict with ids keys
        '''
        return self.convert_dict_with_maps(dict_to_convert,
                                           self.data_id_map, keys='ids')

    def convert_disciplines_dict_with_full_name(self):
        ''' Return disciplines dict with namespaced keys
        '''

        converted_dict = {}

        for key, val in self.disciplines_id_map.items():
            if key not in converted_dict:
                converted_dict[key] = []

            if isinstance(val, list):
                for val_element in val:
                    if val_element in self.disciplines_dict:
                        converted_dict[key].append(
                            self.disciplines_dict[val_element])
            else:
                if val in self.disciplines_dict:
                    converted_dict[key].append(self.disciplines_dict[val])
        return converted_dict

    def create_reduced_dm(self):
        self.reduced_dm = self.get_data_dict_list_attr(
            [ProxyDiscipline.TYPE, ProxyDiscipline.SUBTYPE, ProxyDiscipline.TYPE_METADATA, ProxyDiscipline.NUMERICAL,
             ProxyDiscipline.DF_EXCLUDED_COLUMNS, ProxyDiscipline.VAR_NAME, ProxyDiscipline.COUPLING])

    def convert_dict_with_maps(self, dict_to_convert, map_full_names_ids, keys='full_names'):
        ''' Convert dict keys with ids to full_names or full_names to ids
            keys: 'full_names' or 'ids'
        '''
        converted_dict = {}
        if keys == 'full_names':
            for key, val in map_full_names_ids.items():
                if isinstance(val, list):
                    for val_element in val:
                        if val_element in dict_to_convert:
                            # The last val_element overwrites the others ...
                            converted_dict[key] = dict_to_convert[val_element]

                else:
                    if val in dict_to_convert:
                        converted_dict[key] = dict_to_convert[val]

        elif keys == 'ids':
            for key, val in map_full_names_ids.items():
                if key in dict_to_convert:
                    if isinstance(val, list):
                        for val_element in val:
                            converted_dict[val_element] = dict_to_convert[key]
                    else:
                        converted_dict[val] = dict_to_convert[key]

        return converted_dict

    def variable_exists_in_dm(self, var_f_name):
        return var_f_name in self.data_id_map.keys()

    def update_with_discipline_dict(self, disc_id, disc_dict):
        ''' Store and update the discipline data into the DM dictionary
        '''
        self.logger.debug(
            f'store and update the discipline data into the DM dictionary {list(disc_dict.keys())[:10]} ...')

        def _dm_update(var_name, io_type, var_f_name):
            if var_f_name in self.data_id_map.keys():
                # If data already exists in DM
                var_id = self.get_data_id(var_f_name)
                # IO_TYPE_IN HANDLING
                if io_type == IO_TYPE_IN:
                    if self.data_dict[var_id][IO_TYPE] != IO_TYPE_OUT:
                        # Before linking check if var data in the DM and in the
                        # disc are coherent
                        self.check_var_data_input_mismatch(
                            var_name, var_f_name, disc_id, self.data_dict[var_id], disc_dict[var_name])
                        disc_dict[var_name] = self.data_dict[var_id]
                        # # reference the parameter from the data manager to the
                        # # discipline...
                        # # ...but keep original namespace in disc_dict!
                        # if 'namespace' in disc_dict[var_name].keys():
                        #     original_namespace = disc_dict[var_name]['namespace']
                        # if 'ns_reference' in disc_dict[var_name].keys():
                        #     original_ns_reference = disc_dict[var_name]['ns_reference']
                        # disc_dict[var_name] = deepcopy(self.data_dict[var_id])
                        # disc_dict[var_name]['var_name'] = var_name
                        # if 'namespace' in disc_dict[var_name].keys():
                        #     disc_dict[var_name]['namespace'] = original_namespace
                        # if 'ns_reference' in disc_dict[var_name].keys():
                        #     disc_dict[var_name]['ns_reference'] = original_ns_reference
                    # else data already exist as OUTPUT and has priority!
                    # => do nothing
                else:
                    # io_type == IO_TYPE_OUT
                    if self.data_dict[var_id][IO_TYPE] == IO_TYPE_OUT:

                        # if already existing data was an OUTPUT
                        if disc_id in self.data_dict[var_id][DISCIPLINES_DEPENDENCIES]:
                            if self.data_dict[var_id][ORIGIN] == disc_id:
                                # if same discipline: just an update
                                self.data_dict[var_id].update(
                                    disc_dict[var_name])
                        else:
                            if self.get_disc_full_name(self.data_dict[var_id][ORIGIN]) is None:
                                # Quick fix to solve cases when
                                self.logger.warning(
                                    f"Discipline with id {self.data_dict[var_id][ORIGIN]} does not exist in the DM.")
                            else:
                                # Data cannot be the OUTPUT of several
                                # disciplines
                                raise ValueError(
                                    f'Model key: {self.get_var_full_name(var_id)} of discipline {self.get_disc_full_name(disc_id)} already exists from disc {self.get_disc_full_name(self.data_dict[var_id][ORIGIN])}')
                    else:
                        # Overwrite OUTPUT information over INPUT information
                        # (OUTPUT has priority)
                        disc_dict[var_name][ORIGIN] = disc_id
                        disc_dict[var_name][DISCIPLINES_DEPENDENCIES] = self.data_dict[var_id][DISCIPLINES_DEPENDENCIES]
                        self.no_change = False
                        # Fix introduced to avoid a particular problem where the input value which is properly set is
                        # overwritten by a null output.
                        # The case was encountered when loading the pickle of an optim study and on an output variable
                        # from a discipline which shared a node with another
                        # one ('techno invest level').
                        if self.data_dict[var_id][VALUE] is not None:
                            disc_dict[var_name][VALUE] = self.data_dict[var_id][VALUE]
                        self.data_dict[var_id] = disc_dict[var_name]
                if disc_id not in self.data_dict[var_id][DISCIPLINES_DEPENDENCIES]:
                    self.data_dict[var_id][DISCIPLINES_DEPENDENCIES].append(
                        disc_id)
                    self.no_change = False
            else:
                # data does not exist yet, add it DM
                disc_dict[var_name][ORIGIN] = disc_id
                disc_dict[var_name][DISCIPLINES_DEPENDENCIES] = [disc_id]
                var_id = self.get_an_uuid()
                self.no_change = False
                self.data_dict[var_id] = disc_dict[var_name]
                self.data_id_map[var_f_name] = var_id
            # END update method

        for var_name in disc_dict.keys():
            io_type = disc_dict[var_name][IO_TYPE]
            ns_reference = disc_dict[var_name][NS_REFERENCE]
            complete_var_name = disc_dict[var_name][VAR_NAME]
            var_f_name = self.ns_manager.compose_ns([ns_reference.value,
                                                     complete_var_name])
            _dm_update(var_name, io_type, var_f_name)

    def update_disciplines_dict(self, disc_id, reference, disc_f_name):
        ''' Store and update discipline into disciplines_dicts
        '''
        self.logger.debug(
            f'store the discipline instances as references for {disc_id}')

        if disc_id is not None:
            # Update already existing discipline with reference
            self.disciplines_dict[disc_id].update(reference)

        else:
            # Create new discipline with unique id in disciplines_dict
            disc_id = self.get_an_uuid()
            self.disciplines_dict[disc_id] = reference
            self.add_disc_id_to_disc_id_map(disc_f_name, disc_id)

        return disc_id

    def add_disc_id_to_disc_id_map(self, disc_f_name, disc_id):
        if disc_f_name in self.disciplines_id_map:
            # self.disciplines_id_map[disc_f_name].append(disc_id)
            disc1_name = self.get_discipline(disc_id).get_module()
            disc2_name = self.get_disciplines_with_name(disc_f_name)[0].get_module()
            error_msg = f'Trying to add two distinct disciplines with the same local namespace: {disc_f_name} , classes are : {disc1_name} and {disc2_name}'
            self.logger.error(error_msg)
            # Exception
            raise DataManagerException(error_msg)

            # # Off-design behavior
            # self.disciplines_id_map[disc_f_name].append(disc_id)
        else:
            self.disciplines_id_map[disc_f_name] = [disc_id]

    def update_disciplines_dict_with_id(self, disc_id, reference):
        '''
        Update the discipline_dict of a specific disc_id with the dict reference
        '''
        self.disciplines_dict[disc_id].update(reference)

    def update_data_dict_with_id(self, var_id, data_dict):
        '''
        Update the discipline_dict of a specific disc_id with the dict reference
        '''
        self.no_change = False
        self.data_dict[var_id].update(data_dict)

    def create_treeview(self, root_process, process_module, no_data=False, read_only=False, exec_display=False):
        '''
        Function that builds a composite structure
        regarding the DataManager stored data
        '''
        self.treeview = TreeView(
            name=self.name, no_data=no_data, read_only=read_only, exec_display=exec_display)
        self.treeview.create_tree_node(data_manager=self,
                                       root_process=root_process,
                                       process_module=process_module,
                                       ns_manager=self.ns_manager)

    def export_data_dict_and_zip(self, export_dir):
        '''
        method that exports the DM data dict to csv files
        by a treatment delegated to the TreeView class (using strategy object)

        :params: anonymize_function, a function that map a given key of the data
        dictionary using rule given by the execution engine for the saving process
        :type: function
        '''
        self.logger.debug('export data in exported csv files')
        # retrieve only key unit and value from data_dict
        dict_to_export = {}
        for k, v in self.convert_data_dict_with_full_name().items():
            dict_to_export[k] = {'unit': v.get('unit'),
                                 'value': v.get('value')}

        serializer = DataSerializer()
        return serializer.export_data_dict_and_zip(dict_to_export,
                                                   export_dir)

    def remove_keys(self, disc_id, data_d, io_type):
        ''' Clean namespaced keys in data_dict and remove discipline dependencies
        The io_type is only here to switch the dm io_type in case we suppress an output variable which still exist as an input
        '''
        list_data_d = [data_d] if isinstance(data_d, str) else data_d

        for var_f_name in list_data_d:
            # -- Only remove variables discid is reponsible for !
            if var_f_name in self.data_id_map.keys():
                var_id = self.get_data_id(var_f_name)

                # -- Some variables has more than one dependencies on variables.
                # -- Dependencies are ketp into DISCIPLINES_DEPENDENCIES list
                # -- When asked for a remove, this list is use to determine if the discipline was one of
                # -- the dependencies and it removed from the the list.
                # -- once the list is empty the variables is removed from the data manager.

                if disc_id in self.data_dict[var_id][ProxyDiscipline.DISCIPLINES_DEPENDENCIES]:
                    self.no_change = False
                    self.data_dict[var_id][ProxyDiscipline.DISCIPLINES_DEPENDENCIES].remove(
                        disc_id)

                    if len(self.data_dict[var_id][ProxyDiscipline.DISCIPLINES_DEPENDENCIES]) == 0:
                        # remove data in data_dict and data_id_map if no other
                        # discipline dependency
                        del self.data_dict[var_id]
                        del self.data_id_map[var_f_name]
                    else:
                        # only one discipline can declare a variable as output then if this key is removed from the discipline and the variable still exists
                        # then the variable becomes an input
                        if io_type == ProxyDiscipline.IO_TYPE_OUT:
                            self.data_dict[var_id][ProxyDiscipline.IO_TYPE] = ProxyDiscipline.IO_TYPE_IN

    def clean_from_disc(self, disc_id, clean_keys=True):
        ''' Clean disc in disciplines_dict and data_in/data_out keys in data_dict
        '''
        disc_f_name = self.get_disc_full_name(disc_id)
        if disc_f_name not in self.disciplines_id_map:
            msg = "Discipline " + str(disc_f_name) + \
                  " not found in DataManager, "
            msg += "it is not possible to delete it."
            raise KeyError(msg)

        # remove discipline in disciplines_dict and disciplines_id_map

        disc_ref = self.get_discipline(disc_id)

        if clean_keys:
            self.clean_keys(disc_id)
        else:
            keys_to_remove = [
                disc_ref._convert_to_namespace_name(key, disc_ref.IO_TYPE_IN)
                for key in disc_ref.DEFAULT_NUMERICAL_PARAM
            ]
            self.remove_keys(
                disc_id, keys_to_remove, disc_ref.IO_TYPE_IN)

        # remove discipline in disciplines_dict and disciplines_id_map
        self.disciplines_id_map[disc_f_name].remove(disc_id)
        if len(self.disciplines_id_map[disc_f_name]) == 0:
            self.disciplines_id_map.pop(disc_f_name)
        self.disciplines_dict.pop(disc_id)

    def clean_keys(self, disc_id):
        '''
        Clean data_in/out of disc_id in data_dict
        '''
        disc_ref = self.get_discipline(disc_id)
        # clean input keys from dm
        self.remove_keys(disc_id, list(disc_ref.apply_visibility_ns(
            ProxyDiscipline.IO_TYPE_IN)), ProxyDiscipline.IO_TYPE_IN)
        # clean output keys from dm
        self.remove_keys(disc_id, list(disc_ref.apply_visibility_ns(
            ProxyDiscipline.IO_TYPE_OUT)), ProxyDiscipline.IO_TYPE_OUT)

    def export_couplings(self, in_csv=False, f_name=None):
        ''' Export couplings of all disciplines registered in the DM
        '''
        d_dict = self.disciplines_dict
        # gather disciplines references
        disc_list = [d_dict[key][self.DISC_REF] for key in d_dict]
        # filter sos_couplings
        sosc_list = [disc for disc in disc_list if disc.is_sos_coupling]
        # loop on sosc and append couplings data
        if sosc_list != []:
            sosc = sosc_list.pop()
            sosc._build_coupling_structure()
            df = sosc.export_couplings()
            for sosc in sosc_list:
                df_sosc = sosc.export_couplings()

                df = concat([df, df_sosc], ignore_index=True)
            # write data or return dataframe
            if in_csv:
                # writing of the file
                if f_name is None:
                    f_name = f"{self.name}.csv"
                df.to_csv(f_name, index=False)
            else:
                return df
        else:
            return None

    def build_disc_status_dict(self, disciplines_to_keep=None):
        ''' build a containing disc/status info into dictionary) '''
        disc_status = {}
        for namespace, disciplines in self.convert_disciplines_dict_with_full_name().items():
            if namespace not in disc_status:
                disc_status[namespace] = {}

            for discipline in disciplines:
                if disciplines_to_keep is not None:
                    if discipline['reference'] in disciplines_to_keep:
                        disc_status[namespace].update(
                            {discipline['classname']: discipline[DataManager.STATUS]})
                else:
                    disc_status[namespace].update(
                        {discipline['classname']: discipline[DataManager.STATUS]})
        return disc_status

    def get_parameter_data(self, parameter_key):
        # TO DO: convert parameter_key into uid
        ''' returns BytesIO of the data value from a namespaced key based of string table format '''
        self.logger.debug(f'get BytesIO of data {parameter_key}')
        # use cache to get parameter data value
        param_data = self.convert_data_dict_with_full_name()[parameter_key]
        param_value = param_data[ProxyDiscipline.VALUE]
        if param_value is None:
            return None
        else:
            serializer = DataSerializer()
            return serializer.convert_to_dataframe_and_bytes_io(param_value,
                                                                parameter_key)

    def get_var_name_from_uid(self, var_id):
        ''' Get namespace and var_name and return namespaced variable
        '''
        return self.data_dict[var_id][ProxyDiscipline.VAR_NAME]

    def get_var_full_name(self, var_id):
        ''' Get namespace and var_name and return namespaced variable
        '''
        var_name = self.get_var_name_from_uid(var_id)
        ns_reference = self.data_dict[var_id][ProxyDiscipline.NS_REFERENCE]
        var_f_name = self.ns_manager.compose_ns([ns_reference.value, var_name])
        return var_f_name

    def get_var_display_name(self, var_id):
        ''' Get namespace and var_name and return namespaced variable
        '''
        var_name = self.get_var_name_from_uid(var_id)
        ns_reference = self.data_dict[var_id][ProxyDiscipline.NS_REFERENCE]
        var_display_name = self.ns_manager.compose_ns(
            [ns_reference.get_display_value(), var_name])
        return var_display_name

    def get_disc_full_name(self, disc_id):
        ''' Get full discipline name
        '''
        if disc_id in self.disciplines_dict:
            return self.disciplines_dict[disc_id][ProxyDiscipline.NS_REFERENCE].value
        else:
            return None

    def check_var_data_input_mismatch(self, var_name, var_f_name, var_id, data1, data2):
        '''
        Check a mismatch between two dicts (which are data dict for the same input shared by two variables
        By default the model origin fills the dm, if difference in DATA_TO_CHECK then a warning is printed
        '''

        if self.logger.level <= logging.DEBUG:

            def compare_data(data_name):

                if data_name == ProxyDiscipline.UNIT and data1[
                    ProxyDiscipline.TYPE] not in ProxyDiscipline.NO_UNIT_TYPES:
                    return str(data1[data_name]) != str(
                        data2[data_name]) or data1[data_name] is None
                elif var_f_name in self.no_check_default_variables:
                    return False
                elif isinstance(data1[data_name], pd.DataFrame) and isinstance(data2[data_name], pd.DataFrame):
                    return compare_dataframes(data1[data_name], data2[data_name], tree=None, error=None,
                                              df_equals=False)
                else:
                    return str(data1[data_name]) != str(data2[data_name])

            for data_name in ProxyDiscipline.DATA_TO_CHECK + [ProxyDiscipline.DEFAULT]:
                if compare_data(data_name):
                    self.logger.debug(
                        f"The variable {var_name} is used in input of several disciplines and does not have same {data_name} : {data1[data_name]} in {self.get_discipline(data1['model_origin']).__class__} different from {data2[data_name]} in {self.get_discipline(var_id).__class__}")
