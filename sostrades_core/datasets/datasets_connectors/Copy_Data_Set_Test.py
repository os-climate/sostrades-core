from enum import Enum
import logging
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetConnectorType, DatasetsConnectorFactory
from sostrades_core.datasets.datasets_connectors.datasets_connector_manager import DatasetsConnectorManager
# from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetsConnectorFactory
# from sostrades_core.datasets.datasets_connectors.datasets_connector_Manage import DatasetsConnectorFactory


from sostrades_core.datasets.datasets_connectors.json_datasets_connector import (
    JSONDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.arango_datasets_connector import (
    ArangoDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.datasets.datasets_connectors.sospickle_datasets_connector import SoSPickleDatasetsConnector
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta

if __name__ == "__main__":
    """
    Usage of the manager
    Copy dataset from json to arangodb

    """
    import os
    
    logging.getLogger("sostrades_core.datasets").setLevel(logging.DEBUG)

    arangoCnx = DatasetsConnectorManager.get_connector("Arango_connector")
    jsonCnx = DatasetsConnectorManager.get_connector("MVP0_datasets_connector")

    print(arangoCnx)
    print(jsonCnx)

    datasets  = jsonCnx.get_datasets_available()
    print(datasets)

    for data in datasets:
        arangoCnx.write_dataset(data,jsonCnx.get_values_all(data),True,True)  

    arangoCnx.write_dataset(",;kjsdhfkjdshf#]###@1325",jsonCnx.get_values_all("dataset_bcx_2"),True,True) 

    print(arangoCnx.get_values_all(",kjsdhfkjdshf#]@1325"))




    # # Json connector
    # json_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "data", "test_92_datasets_db.json")

    # # Explicit args
    # json_connector_with_explicit_args = DatasetsConnectorFactory.get_connector(DatasetConnectorType.JSON, file_path=json_file_path)
    # print("JSON connector from explicit args", json_connector_with_explicit_args)

    # # With instanciation dict
    # connector_instanciation_dict = {"file_path": json_file_path}
    # json_connector_from_dict = DatasetsConnectorFactory.get_connector(
    #     DatasetConnectorType.JSON, **connector_instanciation_dict
    # )
    # print("JSON connector from dict", json_connector_from_dict)

    # # Arango connector    
    # arango_instanciation_dict = {
    #     "host": 'http://127.0.0.1:8529',
    #     "db_name":'os-climate',
    #     "username":"root",
    #     "password":"ArangoDB_BfPM",
    # }
    
    # arango_connector_from_dict = DatasetsConnectorFactory.get_connector(
    #     DatasetConnectorType.Arango, **arango_instanciation_dict
    # )
    # print("Arango connector from dict", arango_connector_from_dict)

    # # Copy dataset
    # arango_connector_from_dict.copy_dataset_from(connector_from=json_connector_from_dict, dataset_identifier="default_numerical_parameters", create_if_not_exists=True, override=True)
    # print(arango_connector_from_dict.get_values_all(dataset_identifier="default_numerical_parameters"))

    # file_path = os.path.join(os.path.dirname(__file__), "uc1_test_damage_ggo.pickle")
    # pickle_connector = SoSPickleDatasetsConnector(file_path=file_path)
    
    # #arango_connector_from_dict.copy_all_datasets_from(connector_from=pickle_connector, create_if_not_exists=True, override=True)
