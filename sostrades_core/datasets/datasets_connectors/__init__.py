from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV0 import (
    JSONDatasetsConnectorV0
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV1 import (
    JSONDatasetsConnectorV1
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connectorV0 import (
    LocalFileSystemDatasetsConnectorV0
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connectorV1 import (
    LocalFileSystemDatasetsConnectorV1
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connector_multiversion import (
    LocalFileSystemDatasetsConnectorMV
)


# alias for main core connectors in submodules
JSON_V0 = JSONDatasetsConnectorV0
JSON_V1 = JSONDatasetsConnectorV1
Local_V0 = LocalFileSystemDatasetsConnectorV0
Local_V1 = LocalFileSystemDatasetsConnectorV1
Local_MV = LocalFileSystemDatasetsConnectorMV
