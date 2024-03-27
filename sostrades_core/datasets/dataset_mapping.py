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
from dataclasses import dataclass
import os
import json

from sostrades_core.datasets.dataset_info import DatasetInfo


@dataclass()
class DatasetsMapping:
    """
    Stores namespace/dataset mapping
    """
    # Keys for parsing json
    DATASETS_INFO_KEY = "datasets_infos"
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
    CONNECTOR_ID_KEY = DatasetInfo.CONNECTOR_ID_KEY
    DATASET_ID_KEY = DatasetInfo.DATASET_ID_KEY
    MAPPING_KEY_FIELDS = [MAP_VERSION, NAMESPACE_VALUE, PARAMETER_NAME]
    MAPPING_ITEM_FIELDS = [CONNECTOR_ID_KEY, DATASET_ID_KEY, PARAMETER_NAME]

    # Process module name
    process_module_path:str
    # Dataset info [dataset_name : DatasetInfo]
    datasets_infos: dict[str:DatasetInfo]
    # Namespace mapping [namespace_name : List[DatasetInfo]]
    namespace_datasets_mapping: dict[str : list[DatasetInfo]]

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
                    sub_namespace_datasets_mapping = sub_mapping.get_namespace_datasets_mapping_for_parent(namespace)
                    namespace_datasets_mapping.update(sub_namespace_datasets_mapping)
                else:
                    raise Exception(f"The dataset mapping file {sub_process_mapping_path} does not exists")
                
        # Parse datasets info
        if DatasetsMapping.DATASETS_INFO_KEY in input_dict.keys():
            for dataset in input_dict[DatasetsMapping.DATASETS_INFO_KEY]:
                datasets_infos[dataset] = DatasetInfo.deserialize(
                    input_dict=input_dict[DatasetsMapping.DATASETS_INFO_KEY][dataset]
                )

        # Parse namespace datasets mapping
        if DatasetsMapping.NAMESPACE_KEY in input_dict.keys():
            input_dict_dataset_mapping = input_dict[DatasetsMapping.NAMESPACE_KEY]
            for mapping_key, datasets in input_dict_dataset_mapping.items():
                mapping_key_fields = DatasetsMapping.extract_mapping_key_fields(mapping_key)
                # TODO: version, parameter not handled
                namespace = mapping_key_fields[DatasetsMapping.NAMESPACE_VALUE]
                namespace_datasets_mapping[namespace] = []
                for dataset in datasets:
                    dataset_fields = DatasetsMapping.extract_mapping_item_fields(dataset)
                    connector_id = dataset_fields[DatasetsMapping.CONNECTOR_ID_KEY]
                    dataset_id = dataset_fields[DatasetsMapping.DATASET_ID_KEY]
                    # TODO: parameter not handled
                    if dataset not in datasets_infos:
                        datasets_infos.update({dataset: DatasetInfo(connector_id, dataset_id)})
                    namespace_datasets_mapping[namespace].append(datasets_infos[dataset])
            
        return DatasetsMapping(
            process_module_path=input_dict[DatasetsMapping.PROCESS_MODULE_PATH_KEY],
            datasets_infos=datasets_infos,
            namespace_datasets_mapping=namespace_datasets_mapping,
        )

    @classmethod
    def extract_mapping_key_fields(cls, mapping_key):
        return cls.__extract_mapping_fields(mapping_key, cls.MAPPING_KEY_FIELDS, "mapping key")

    @classmethod
    def extract_mapping_item_fields(cls, mapping_item):
        return cls.__extract_mapping_fields(mapping_item, cls.MAPPING_ITEM_FIELDS, "mapping value item")

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
        elif fields[-1] != '*':
            # TODO: remove when down to parameter level
            raise NotImplementedError("Parameter-wise referral is not yet implemented.")
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

    def get_datasets_info_from_namespace(self, namespace:str, study_name:str) -> list[DatasetInfo]:
        """
        Gets the datasets info for a namespace

        :param namespace: Name of the namespace
        :type namespace: str
        :param study_name: Name of the study
        :type study_name: str
        """
        datasets_mapping = []
        anonimized_ns = namespace.replace(study_name, self.STUDY_PLACEHOLDER)
        if anonimized_ns in self.namespace_datasets_mapping.keys():
            datasets_mapping = self.namespace_datasets_mapping[anonimized_ns]

        return datasets_mapping
    
    def get_namespace_datasets_mapping_for_parent(self, parent_namespace:str) -> dict[str : list[DatasetInfo]]:
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