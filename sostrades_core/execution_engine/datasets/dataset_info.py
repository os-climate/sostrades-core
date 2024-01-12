from dataclasses import dataclass

@dataclass()
class DatasetInfo:
    connector_id:str
    dataset_id:str

    @staticmethod
    def deserialize(input_dict:dict) -> "DatasetInfo":
        """
        Method to deserialize
        expected 
        {
            "connector_id": <connector_id>,
            "dataset_id": <dataset_id>,
        }
        """
        return DatasetInfo(
            connector_id=input_dict["connector_id"],
            dataset_id=input_dict["dataset_id"],
        )

    def __hash__(self) -> int:
        return hash(str(self))
