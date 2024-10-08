'''
Copyright 2024 Capgemini

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
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.dataset_info.dataset_info_factory import DatasetInfoFactory


class DatasetsMappingException(Exception):
    """
    Generic exception for dataset mapping
    """
    pass


@dataclass()
class DatasetsMapping:
    """
    Stores namespace/dataset mapping
    """
    # Keys for parsing json
    NAMESPACE_KEY = "namespace_datasets_mapping"
    PROCESS_MODULE_PATH_KEY = "process_module_path"
    STUDY_PLACEHOLDER = "<study_ph>"
    SUB_PROCESS_MAPPING = "sub_process_datasets_mapping"

    # Mapping format
    # e.g.: {"map_version|namespace_value|parameter_name": ["connector_id|dataset_id|parameter_name",...], ...}
    MAPPING_SEP = "|"
    MAP_VERSION = "map_version"
    NAMESPACE_VALUE = "namespace_value"
    PARAMETER_NAME = "parameter_name"
    WILDCARD = AbstractDatasetInfo.WILDCARD
    MAPPING_KEY_FIELDS = [MAP_VERSION, NAMESPACE_VALUE, PARAMETER_NAME]

    KEY = 'key'
    VALUE = 'value'
    TYPE = 'type'

    # Process module name
    process_module_path: str
    # Dataset info [connector_id|dataset_id| : DatasetInfo]
    datasets_infos: dict[str:AbstractDatasetInfo]
    # Namespace mapping [namespace_name : List[connector_id|dataset_id|]]
    namespace_datasets_mapping: dict[str : list[str]]
    # Dataset namespace mapping [connector_id|dataset_id| : [namespace: dict[parameter:parameter_dataset]]]
    parameters_mapping: dict[str: dict[str:dict[str:str]]]

    @staticmethod
    def deserialize(input_dict: dict) -> DatasetsMapping:
        """
        Method to deserialize
        expected example
        {
            "process_module_path": "process.module.path"
            "namespace_datasets_mapping": {
                "v0|namespace1|*" : ["connector1|dataset1|*"],
                "v0|namespace2|*" : ["connector1|dataset1|*", "connector1|dataset2|*"]
                "v0|namespace3|parameter1" : ["connector1|dataset1|*", "connector1|dataset2|*"]
                "v0|namespace3|parameter2" : ["connector1|dataset1|parameter_name2", "connector1|dataset2|*"]
                "v0|namespace4|*" : ["connector1|dataset1|parameter_name3", "connector1|dataset2|*"]
            },
            "sub_process_datasets_mapping":{
                "v0|namespace3|*": path to other mapping json file
            }
        }
        :param input_dict: dict like input json object
        :type input_dict: dict
        """
        datasets_infos = {}
        namespace_datasets_mapping = {}
        parameters_mapping = {}

        try:
            # parse sub process datasets info
            # do it first so that info in this mapping will override the sub mappings
            if DatasetsMapping.SUB_PROCESS_MAPPING in input_dict.keys():
                for mapping_key, sub_process_mapping_path in input_dict[DatasetsMapping.SUB_PROCESS_MAPPING].items():
                    mapping_key_fields = DatasetsMapping.extract_mapping_key_fields(mapping_key)
                    namespace = mapping_key_fields[DatasetsMapping.NAMESPACE_VALUE]
                    if os.path.exists(sub_process_mapping_path):
                        # read the json mapping file
                        sub_mapping = DatasetsMapping.from_json_file(sub_process_mapping_path)
                        # retreive the datasets from this maping
                        datasets_infos.update(sub_mapping.datasets_infos)
                        # upadet all the sub namespaces with parent namespace given in mapping file
                        sub_namespace_datasets_mapping = sub_mapping.get_namespace_datasets_mapping_for_parent(namespace)
                        namespace_datasets_mapping.update(sub_namespace_datasets_mapping)
                        # update the dict of corresponding parameters/dataset/namespace
                        sub_parameters_mapping = sub_mapping.get_parameters_datasets_mapping_for_parent(namespace)
                        parameters_mapping.update(sub_parameters_mapping)
                    else:
                        raise DatasetsMappingException(f"The dataset mapping file {sub_process_mapping_path} does not exists")

            # Parse namespace datasets mapping
            if DatasetsMapping.NAMESPACE_KEY in input_dict.keys():
                input_dict_dataset_mapping = input_dict[DatasetsMapping.NAMESPACE_KEY]
                for mapping_key, datasets in input_dict_dataset_mapping.items():
                    mapping_key_fields = DatasetsMapping.extract_mapping_key_fields(mapping_key)
                    # TODO: version not handled
                    namespace = mapping_key_fields[DatasetsMapping.NAMESPACE_VALUE]
                    parameter = mapping_key_fields[DatasetsMapping.PARAMETER_NAME]
                    namespace_datasets_mapping[namespace] = namespace_datasets_mapping.get(namespace, [])

                    for dataset_mapping_key in datasets:
                        #first extract the version
                        dataset_info_version = DatasetInfoFactory.get_dataset_info_version(dataset_mapping_key)

                        # extract the fields of the dataset info key
                        dataset_fields = dataset_info_version.value.deserialize(dataset_mapping_key)
                        parameter_id = dataset_fields[DatasetsMapping.PARAMETER_NAME]

                        #create the dataset info then Check if there is wildcard in dataset info id and replace by ns if needed
                        dataset_info = dataset_info_version.value.create(dataset_fields).copy_with_new_ns(namespace)

                        # build just the id with connector and dataset
                        dataset_info_id = dataset_info.dataset_info_id

                        if dataset_info_id not in datasets_infos:
                            datasets_infos[dataset_info_id] = dataset_info

                        if dataset_info_id not in namespace_datasets_mapping[namespace]:
                            namespace_datasets_mapping[namespace].append(dataset_info_id)

                        # update dataset, namespace, parameter associations
                        parameters_mapping[dataset_info_id] = parameters_mapping.get(dataset_info_id, {})
                        parameters_mapping[dataset_info_id][namespace] = parameters_mapping[dataset_info_id].get(namespace, {})
                        parameter_to_found = parameter_id
                        parameter_from = parameter
                        # if we have '*'-> param_name or param_name -> '*' we retreive the param_name
                        if parameter_id == DatasetsMapping.WILDCARD:
                            parameter_to_found = parameter
                        if parameter == DatasetsMapping.WILDCARD:
                            parameter_from = parameter_id
                        parameters_mapping[dataset_info_id][namespace][parameter_from] = parameter_to_found

        except Exception as exception:
            raise DatasetsMappingException(f'Error reading the dataset mapping file: \n{str(exception)}')
        return DatasetsMapping(
            process_module_path=input_dict[DatasetsMapping.PROCESS_MODULE_PATH_KEY],
            datasets_infos=datasets_infos,
            namespace_datasets_mapping=namespace_datasets_mapping,
            parameters_mapping=parameters_mapping
        )

    @classmethod
    def extract_mapping_key_fields(cls, mapping_key):
        return cls.__extract_mapping_fields(mapping_key, cls.MAPPING_KEY_FIELDS, "mapping key")


    @classmethod
    def __extract_mapping_fields(cls, mapping_key_or_value: str, format_fields: list[str],
                                 error_mode: str) -> dict[str]:
        """
        Utility method to extract the fields of a mapping key or value as a string and return the fields specified
        in the reference format for namespace-dataset mappings.
        :param mapping_key_or_value: the formatted namespace parameter(s) or dataset parameter(s) formatted
        :type mapping_key_or_value: str
        :param format_fields: the fields in the format of the mapping key or value item to be extracted
        :type format_fields: list[str]
        :param error_mode: string to specify an eventual raised error "mapping key" / "mapping value item"
        :type error_mode: str
        :return: dictionary {field: field_value} for the format fields
        """
        fields = mapping_key_or_value.split(cls.MAPPING_SEP)
        if len(fields) != len(format_fields):
            raise ValueError(f"Wrong format for {mapping_key_or_value}, "
                             f"the expected {error_mode} format "
                             f"is {cls.MAPPING_SEP.join(format_fields)}")

        else:
            return dict(zip(format_fields, fields))

    @staticmethod
    def from_json_file(file_path: str) -> "DatasetsMapping":
        """
        Method to deserialize from a json file
        :param file_path: path of the file to deserialize
        :type file_path: str
        """
        with open(file_path, "rb") as file:
            json_data = json.load(file)
        return DatasetsMapping.deserialize(json_data)

    def get_datasets_info_from_namespace(self, namespace: str, study_name: str) -> dict[AbstractDatasetInfo:dict[str:str]]:
        """
        Gets the datasets info for a namespace: replace the placeholder with study names
        + if wildcard in namespaces return all dataset id associated to this wildcard

        :param namespace: Name of the namespace
        :type namespace: str
        :param study_name: Name of the study
        :type study_name: str
        """
        datasets_mapping = {}
        anonimized_ns = namespace.replace(study_name, self.STUDY_PLACEHOLDER)

        # return the datasets associated to the namespace
        if anonimized_ns in self.namespace_datasets_mapping.keys():
            dataset_ids = self.namespace_datasets_mapping[anonimized_ns]
            for dataset_id in dataset_ids:
                datasets_mapping[self.datasets_infos[dataset_id]] = self.parameters_mapping[dataset_id].get(anonimized_ns, {})

        # if there is in the mapping a wildcard at ns level, we need for all namespace to return the datasets associated
        if DatasetsMapping.WILDCARD in self.namespace_datasets_mapping.keys():
            dataset_ids = self.namespace_datasets_mapping[DatasetsMapping.WILDCARD]
            for dataset_id in dataset_ids:
                if self.WILDCARD in dataset_id:
                    # if there is still wildcard in dataset info, create a new one and replace the ns
                    dataset_info = self.datasets_infos[dataset_id].copy_with_new_ns(anonimized_ns)
                else:
                    dataset_info = self.datasets_infos[dataset_id]
                datasets_mapping[dataset_info] = datasets_mapping.get(dataset_info,{})
                datasets_mapping[dataset_info].update(self.parameters_mapping[dataset_id].get(DatasetsMapping.WILDCARD, {}))

        return datasets_mapping

    def get_datasets_namespace_mapping_for_study(self, study_name: str, namespaces_dict: dict[str:dict]) -> tuple[dict, dict]:
        """
        Get the datasets_namespace_mapping and replace the study_placeholder with the study_name
        get the mapping of parameters for this namespace
        :param study_name: name of the study
        :type study_name: str
        :param namespaces_dict: dict with all data from dm to retrieve by namespaces
        :type namespaces_dict: dict with format: {ns: {KEY:{data:data_key}, {VALUE:{data:data_value}}, {TYPE:{data:data_type}}}
        :return: datasets_parameters_mapping with all data to write in the datasets and mapping info + duplicated data values
        """
        datasets_mapping = {}
        duplicates = {}

        for dataset, namespaces_mapping_dict in self.parameters_mapping.items():
            try:
                # create the dict that will contain all data to write in the dataset for all associated namespaces
                all_data_in_dataset = {DatasetsMapping.VALUE: {}, DatasetsMapping.TYPE: {}, DatasetsMapping.KEY: {}}

                # iterate for all namespace associated to the dataset, parameters_mapping_dict contains the association param_name -> dataset_param_name
                for namespace, parameters_mapping_dict in namespaces_mapping_dict.items():

                    study_namespace = namespace.replace(self.STUDY_PLACEHOLDER, study_name)

                    # find the corresponding namespace between the namespace_dict and the current namespace
                    # (if the current namespace is a wildcard '*' we retreive all the namespaces)
                    corresponding_namespaces = []
                    if namespace == DatasetsMapping.WILDCARD:
                        corresponding_namespaces.extend(namespaces_dict.keys())
                    elif study_namespace in namespaces_dict.keys():
                        corresponding_namespaces.append(study_namespace)

                    if len(corresponding_namespaces) > 0:
                        for data, dataset_data in parameters_mapping_dict.items():

                            if dataset_data == DatasetsMapping.WILDCARD:
                                # if wildcard at parameter place: dataset_connector|dataset_name|*
                                # search for all the data in the namespaces
                                for ns in corresponding_namespaces:
                                    for key in namespaces_dict[ns][DatasetsMapping.VALUE].keys():
                                        if key in all_data_in_dataset[DatasetsMapping.KEY].keys():
                                            duplicates[key] = ns  # the last namespace is the one that will hold the value
                                    all_data_in_dataset[DatasetsMapping.VALUE].update(namespaces_dict[ns][DatasetsMapping.VALUE])
                                    all_data_in_dataset[DatasetsMapping.TYPE].update(namespaces_dict[ns][DatasetsMapping.TYPE])
                                    all_data_in_dataset[DatasetsMapping.KEY].update(namespaces_dict[ns][DatasetsMapping.KEY])
                            else:
                                # search for the data name in the corresponding namespaces
                                corresponding_data = {ns: [value for key, value in namespaces_dict[ns][DatasetsMapping.VALUE].items() if key == data] for ns in corresponding_namespaces}
                                if len(corresponding_data.keys()) > 0:
                                    # if the name of the dataset parameter already exists, it will overwrite the already set data
                                    # so we retrun the list of duplicated data
                                    if dataset_data in all_data_in_dataset[DatasetsMapping.KEY].keys():
                                        duplicates[dataset_data] = namespace  # the last namespace is the one that will hold the value
                                    # we get the last occurence or the data, the other are added in duplicates list
                                    last_ns = list(corresponding_data.keys())[-1]
                                    if len(corresponding_data[last_ns]) > 0:
                                        last_value = corresponding_data[last_ns][-1]
                                        all_data_in_dataset[DatasetsMapping.VALUE].update({dataset_data: last_value})
                                        all_data_in_dataset[DatasetsMapping.TYPE].update({dataset_data: namespaces_dict[last_ns][DatasetsMapping.TYPE][data]})
                                        all_data_in_dataset[DatasetsMapping.KEY].update({dataset_data: namespaces_dict[last_ns][DatasetsMapping.KEY][data]})
            except Exception as error:
                raise DatasetsMappingException(f'Error retrieving data from dataset {dataset}]: \n{str(error)}')

            datasets_mapping[dataset] = all_data_in_dataset

        return datasets_mapping, duplicates

    def get_namespace_datasets_mapping_for_parent(self, parent_namespace: str) -> dict[str : list[str]]:
        """
        Get the namespace_datasets_mapping and replace the study_placeholder with the parent_namespace
        :param parent_namespace: parent namespace that will replace the <study_ph> in the child namespaces
        :type parent_namespace: str
        :return: namespace_datasets_mapping with updated namespaces
        """
        datasets_mapping = {}
        for namespace, datasets in self.namespace_datasets_mapping.items():
            new_namespace = namespace.replace(self.STUDY_PLACEHOLDER, parent_namespace)
            datasets_mapping[new_namespace] = datasets

        return datasets_mapping

    def get_parameters_datasets_mapping_for_parent(self, parent_namespace: str) -> dict[str: dict[str:dict[str:str]]]:
        """
        Get the parameters_mapping and replace the study_placeholder with the parent_namespace
        :param parent_namespace: parent namespace that will replace the <study_ph> in the child namespaces
        :type parent_namespace: str
        :return: parameters_mapping with updated namespaces
        """
        parameters_mapping = {}
        for dataset_id, namespace_mapping in self.parameters_mapping.items():
            parameters_mapping[dataset_id] = {ns.replace(self.STUDY_PLACEHOLDER, parent_namespace): namespace_mapping[ns] for ns in namespace_mapping.keys()}

        return parameters_mapping
