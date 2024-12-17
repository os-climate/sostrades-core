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

from google.cloud import bigquery
from pandas import DataFrame


def bq_to_df(dataset_name: str, table_name: str, sql_query: str = "SELECT * FROM `{}`") -> DataFrame:
    """Read a dataset from BigQuery and return it as a dataframe.

    Args:
        dataset_name: The name of the dataset to fetch.
        table_name: The name of the table to fetch inside the dataset.
        sql_query: The SQL REQUEST query, with {} for the table name.
            By default, select the whole table.

    Raises:
        ValueError: If the dataset cannot be found.
        ValueError: If the table cannot be found.
        RuntimeError: If the writing of the csv fails.

    Returns:
        The dataframe.
    """
    client = bigquery.Client(project="gcp-businessplanet")
    table_id = f"{dataset_name}.{table_name}"

    try:
        client.get_dataset(dataset_name)
    except Exception as e:
        msg = f"Dataset {dataset_name} not found"
        raise ValueError(msg) from e

    try:
        client.get_table(table_id)
    except Exception as e:
        msg = f"Table {table_name} not found"
        raise ValueError(msg) from e

    try:
        sql_cmd = sql_query.format(table_id)
        results = client.query(sql_cmd).result()
    except Exception as e:
        msg = f"Error in the SQL query: {e}"
        raise RuntimeError(msg) from e
    else:
        return DataFrame(
            data=[list(row) for row in results],
            columns=[field.name for field in results.schema],
        )
