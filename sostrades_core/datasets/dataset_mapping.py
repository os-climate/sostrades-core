"""
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
"""
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
            "datasets_infos": {
                "Dataset1": {
                    "connector_id": <connector_id>,
                    "dataset_id": <dataset_id>,
                },
                "Dataset2": {
                    "connector_id": <connector_id>,
                    "dataset_id": <dataset_id>,
                }
            },
            "namespace_datasets_mapping": {
                "namespace1" : ["Dataset1"],
                "namespace2" : ["Dataset1", "Dataset2"]
            },
            "sub_process_datasets_mapping":{
                "namespace3": path to other mapping json file
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
            for namespace, sub_process_mapping_path in input_dict[DatasetsMapping.SUB_PROCESS_MAPPING].items():
                if os.path.exists(sub_process_mapping_path):
                    # read the json mapping file
                    sub_mapping = DatasetsMapping.from_json_file(sub_process_mapping_path)
                    # retreive the datasets from this maping
                    datasets_infos.update(sub_mapping.datasets_infos)
                    sub_namespace_datasets_mapping = sub_mapping.get_namespace_datasets_mapping_for_parent(namespace)
                    namespace_datasets_mapping.update(sub_namespace_datasets_mapping)
                else:
                    raise Exception("the dataset mapping file does not exists")
                
        # Parse datasets info
        if DatasetsMapping.DATASETS_INFO_KEY in input_dict.keys():
            for dataset in input_dict[DatasetsMapping.DATASETS_INFO_KEY]:
                datasets_infos[dataset] = DatasetInfo.deserialize(
                    input_dict=input_dict[DatasetsMapping.DATASETS_INFO_KEY][dataset]
                )

        # Parse namespace datasets mapping
        if DatasetsMapping.NAMESPACE_KEY in input_dict.keys():
            input_dict_dataset_mapping = input_dict[DatasetsMapping.NAMESPACE_KEY]
            for namespace in input_dict_dataset_mapping:
                namespace_datasets_mapping[namespace] = []
                for dataset in input_dict_dataset_mapping[namespace]:
                    namespace_datasets_mapping[namespace].append(datasets_infos[dataset])
            
        return DatasetsMapping(
            process_module_path=input_dict[DatasetsMapping.PROCESS_MODULE_PATH_KEY],
            datasets_infos=datasets_infos,
            namespace_datasets_mapping=namespace_datasets_mapping,
        )
    
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