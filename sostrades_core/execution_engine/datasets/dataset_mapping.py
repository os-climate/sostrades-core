from dataclasses import dataclass

import json

from sostrades_core.execution_engine.datasets.dataset_info import DatasetInfo

@dataclass()
class DatasetsMapping:
    datasets_infos:dict[str:DatasetInfo]
    namespace_datasets_mapping:dict[str:list[DatasetInfo]]

    @staticmethod
    def deserialize(input_dict:dict) -> "DatasetsMapping":
        """
        Method to deserialize
        expected example
        {
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
        }
        """
        # Parse datasets info
        datasets_infos = {}
        for dataset in input_dict["datasets_infos"]:
            datasets_infos[dataset] = DatasetInfo.deserialize(input_dict=input_dict["datasets_infos"][dataset])
        
        # Parse namespace datasets mapping
        namespace_datasets_mapping = {}
        input_dict_dataset_mapping = input_dict["namespace_datasets_mapping"]
        for namespace in input_dict_dataset_mapping:
            namespace_datasets_mapping[namespace] = []
            for dataset in input_dict_dataset_mapping[namespace]:
                namespace_datasets_mapping[namespace].append(datasets_infos[dataset])
        return DatasetsMapping(
            datasets_infos=datasets_infos,
            namespace_datasets_mapping=namespace_datasets_mapping,
        )
    
    @staticmethod
    def from_json_file(file_path:str) -> "DatasetsMapping":
        with open(file_path, "rb") as file:
            json_data = json.load(file)
        return DatasetsMapping.deserialize(json_data)


if __name__ == "__main__":
    """
    Some example to wotk with dataset mapping
    """
    import os
    input_dict = {
            "datasets_infos": {
                "Dataset1": {
                    "connector_id": "<1connector_id>",
                    "dataset_id": "<1dataset_id>",
                },
                "Dataset2": {
                    "connector_id": "<2connector_id>",
                    "dataset_id": "<2dataset_id>",
                }
            },
            "namespace_datasets_mapping": {
                "namespace1" : ["Dataset1"],
                "namespace2" : ["Dataset1", "Dataset2"]
            },
        }

    dataset_mapping_from_dict = DatasetsMapping.deserialize(input_dict=input_dict)
    print("Dataset mapping from dict", dataset_mapping_from_dict)

    json_file_path = os.path.join(os.path.dirname(__file__), "example_mapping.json")
    print("Json file path", json_file_path)
    dataset_mapping_from_json = DatasetsMapping.from_json_file(file_path=json_file_path)
    
    print("Dataset mapping from json", dataset_mapping_from_json)