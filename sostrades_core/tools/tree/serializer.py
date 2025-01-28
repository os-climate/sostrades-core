'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/23-2024/07/04 Copyright 2023 Capgemini

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

import contextlib
from io import BytesIO, StringIO
from pathlib import Path
from shutil import make_archive
from tempfile import gettempdir
from typing import Any

from numpy import ndarray
from pandas import DataFrame, concat, read_pickle

from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.tools.folder_operations import makedirs_safe, rmtree_safe
from sostrades_core.tools.rw.load_dump_dm_data import AbstractLoadDump, DirectLoadDump

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Data manager pickle (de)serializer
"""

CSV_SEP = ","
FILE_URL = "file:///"

# def anonymize_dict(dict_to_convert, anonymize_fct=None):
#     if anonymize_fct is not None:
#         converted_dict = {}
#         for key in dict_to_convert.keys():
#             new_key = anonymize_fct(key)
#             converted_dict[new_key] = dict_to_convert[key]
#     else:
#         converted_dict = deepcopy(dict_to_convert)
#     return converted_dict


def strip_study_from_namespace(ns_name: str) -> str:
    """Remove the study name from the namespace name.

    Args:
        ns_name: The name of the namespace.

    Returns:
        The name of the namespace, without the sutdy name.
    """
    study_name = get_study_from_namespace(ns_name)
    return ns_name.replace(study_name + NS_SEP, "")


def get_study_from_namespace(ns_name: str) -> str:
    """Get the study name from the namespace name.

    Args:
        ns_name: The name of the namespace.

    Returns:
        The study name.
    """
    return ns_name.split(NS_SEP)[0]


def generate_unique_data_csv(data_dict: dict[str, Any], csv_file_path: str | Path) -> None:
    """Writes a data dictionary to a csv file.

    Args:
        data_dict: The dictionary to write.
        csv_file_path: The target file path.
    """
    # Convert data to dataframe regarding its type
    df_data = DataSerializer.convert_data_to_dataframe(data_dict)
    # Export data as a DataFrame using buffered I/O streams
    df_data.to_csv(csv_file_path, sep=CSV_SEP, header=True, index=False)


class DataSerializer:
    """Data serializer class."""

    pkl_filename = "dm.pkl"
    val_filename = "dm_values.csv"
    disc_status_filename = "disciplines_status.pkl"
    cache_filename = "cache.pkl"

    default_persistance_strategy = DirectLoadDump

    def __init__(self, root_dir: str | Path | None = None, rw_object=None, study_filename=None):  # noqa: D107
        if root_dir is not None:
            self.dm_db_root_dir = Path(root_dir)
        else:
            self.dm_db_root_dir = Path(gettempdir()) / "DM_dB"

        self.study_filename = study_filename
        self.dm_val_file = None
        self.dm_pkl_file = None
        self.cache_file = None
        # set load/dump strategy
        self.direct_rw_strategy = DirectLoadDump()
        self.encryption_strategy = None
        self.set_strategy_method(rw_object)
        self.check_db_dir()

    def set_strategy_method(self, rw_strat: AbstractLoadDump | None) -> None:
        """Set load/dump strategy method.

        Args:
            rw_strat: The strategy to set.

        Raises:
            TypeError: If the strategy does not inherit from AbstractLoadDump.
        """
        if rw_strat is None:
            rw_strat = self.default_persistance_strategy()
        elif not isinstance(rw_strat, AbstractLoadDump):
            msg = f"object {rw_strat!s} type {type(rw_strat)} should be a strategy class"
            raise TypeError(msg)
        self.encryption_strategy = rw_strat

    def check_db_dir(self):
        """Prepare folder that will store local study DM files."""
        if not self.dm_db_root_dir.is_dir():
            if self.dm_db_root_dir.is_file():
                # Sometimes the folder is a file (!) so we remove it
                with contextlib.suppress(OSError):
                    self.dm_db_root_dir.unlink()
            # Set the option exists_ok=True so that if the folder already exists it doesn't raise an error
            makedirs_safe(self.dm_db_root_dir, exist_ok=True)

    def is_structured_data_type(self, data: Any) -> bool:
        """Check whether the data is a structured type.

        Args:
            data: The data to check.

        Returns:
            Whether the data is a structured type.
        """
        return isinstance(data, (ndarray, list, DataFrame, dict))

    @staticmethod
    def study_data_manager_file_path(study_to_load: Path | str) -> Path:
        """Get the file path to the dm.pkl file.

        Args:
            study_to_load: The study directory.

        Returns:
            The path to the dm.pkl file.
        """
        return Path(study_to_load) / DataSerializer.pkl_filename

    @staticmethod
    def study_disciplines_status_file_path(study_to_load: Path | str):
        """Get the file path to the disciplines_status.pkl file.

        Args:
            study_to_load: The study directory.

        Returns:
            The path to the discipline_status.pkl file.
        """
        return Path(study_to_load) / DataSerializer.disc_status_filename

    @staticmethod
    def study_cache_file_path(study_to_load: Path | str):
        """Get the file path to the cache.pkl file.

        Args:
            study_to_load: The study directory.

        Returns:
            _description_.
        """
        return Path(study_to_load) / DataSerializer.cache_filename

    def dump_disc_status_dict(
        self, study_to_load: Path | str, rw_strategy: AbstractLoadDump, status_dict: dict[str, Any]
    ) -> None:
        """Export disciplines status into binary file (containing disc/status info into dictionary).

        Args:
            study_to_load: The study directory.
            rw_strategy: The read/write strategy to use.
            status_dict: The status dictionary to write.
        """
        status_dict_f = Path(study_to_load) / self.disc_status_filename

        rw_strategy.dump(status_dict, status_dict_f)

    def load_cache_dict(self, study_to_load: Path | str, rw_strategy: AbstractLoadDump) -> dict[str, Any] | None:
        """Load disciplines cache from binary file (containing disc/status info into dictionary).

        Args:
            study_to_load: The study directory.
            rw_strategy: The read/write strategy to use.

        Returns:
            The discipline cache dictionary, or None if no cache pickle file is found.
        """
        cache_dict_f = self.get_dm_file(study_to_load=study_to_load, file_type=self.cache_filename)
        return rw_strategy.load(cache_dict_f) if cache_dict_f is not None else None

    def load_disc_status_dict(self, study_to_load: Path | str, rw_strategy: AbstractLoadDump):
        """Load disciplines status from binary file (containing disc/status info into dictionary).

        Args:
            study_to_load: The study directory.
            rw_strategy: The read/write strategy to use.

        Returns:
            The discipline status dictionary.
        """
        status_dict_f = self.get_dm_file(study_to_load=study_to_load, file_type=self.disc_status_filename)

        return rw_strategy.load(status_dict_f)

    def export_data_dict_to_csv(self, origin_dict: dict[str, Any], export_dir: Path | str) -> DataFrame:
        """Export values and units of the whole DM data_dict to a csv file.

        Args:
            origin_dict: The dictionary to export.
            export_dir: The directory where the csv file shall be written.

        Returns:
            The dataframe created from the dictionary.
        """
        data_df = DataFrame(columns=["unit", "value"])
        for key, val in origin_dict.items():
            val_to_display = val["value"]
            if self.is_structured_data_type(val_to_display):
                csv_f = Path(export_dir) / f"{key}.csv"
                generate_unique_data_csv(val_to_display, csv_f)
                val_to_display = FILE_URL + csv_f.name
            data_df.loc[key] = {"unit": val["unit"], "value": val_to_display}
        # force null cells to None instead of NaN (avoiding SQL issue)
        return data_df.where(data_df.notnull(), None)

    def export_data_dict_and_zip(self, origin_dict: dict[str | Any], export_dir: Path | str) -> Path:
        """Export values and units of the whole DM data_dict to csv file and zip.

        Args:
            origin_dict: The dictionary to export.
            export_dir: The directory where the files shall be written.

        Returns:
            The path to the zip file.
        """
        export_dir = Path(export_dir)
        if not export_dir.is_dir():
            makedirs_safe(export_dir)

        data_df = self.export_data_dict_to_csv(origin_dict, export_dir=export_dir)
        self.dm_val_file = export_dir / self.val_filename
        data_df.to_csv(self.dm_val_file, sep=CSV_SEP, columns=data_df.columns)
        # zip folder and return zip filepath
        export_dir_zip = make_archive(export_dir, "zip", export_dir.parent, export_dir.name)
        rmtree_safe(export_dir)
        return Path(export_dir_zip)

    def load_from_pickle(
        self,
        data_dict,
        rw_strategy,
        just_return_data_dict=False,
    ):
        """Load a pickled file to a dict
        according to serialisation strategy (pickled dataframe or pickled raw dict data)
        and update data_dict with this data
        loop on the items of this dict
            if key matches with given data_dict key
                restore data from raw data (type is kept)
                and update data_dict with this data.
        """
        loaded_dict = rw_strategy.load(self.dm_pkl_file)

        def is_variable_wo_studyname_exists(var, a_d):
            """Check if variable exists in data dict
            by ignoring the study name, i.e. the first element splitted by .
            """
            key_wo_st = strip_study_from_namespace(var)
            no_study_keys = [strip_study_from_namespace(k) for k in a_d]
            return key_wo_st in no_study_keys

        current_st_n = None
        loaded_st_n = None
        if not just_return_data_dict:
            current_st_n = get_study_from_namespace(next(iter(data_dict.keys())))
            loaded_st_n = get_study_from_namespace(next(iter(loaded_dict.keys())))
        for param_id, param_dict in loaded_dict.items():
            if just_return_data_dict or is_variable_wo_studyname_exists(param_id, data_dict):
                param_id_to_update = param_id
                if current_st_n != loaded_st_n:
                    param_id_to_update = param_id_to_update.replace(loaded_st_n, current_st_n)
                if just_return_data_dict:
                    data_dict[param_id_to_update] = {}
                data_dict[param_id_to_update].update(param_dict)
            else:
                sp_var = strip_study_from_namespace(param_id)
                msg = f"Variable {sp_var} does not exist into {data_dict.keys()}"
                raise KeyError(msg)

    def get_dm_file(self, study_to_load: Path | str, file_type: str | None = None) -> Path | None:
        """Return the path of the file containing data for a given study.

        Args:
            study_to_load: The directory of the study to load.
            file_type: The type of file to look for.

        Raises:
            FileNotFoundError: If the study directory does not exist.
            FileNotFoundError: If there is not data file in the study directory.
            FileNotFoundError: If there are more than one data file in the study directory.

        Returns:
            The path of the data file, or None if no file was found and the file type was `cache`.
        """
        if file_type is None:
            file_type = self.pkl_filename

        study_to_load = Path(study_to_load)

        if not study_to_load.is_dir():
            # If just the name of a study was given, compose the path with root_dir
            study_to_load = self.dm_db_root_dir / study_to_load
            if not study_to_load.is_dir():
                msg = f"DM directory {study_to_load} does not exist"
                raise NotADirectoryError(msg)
        dm_files = list(study_to_load.rglob(file_type))
        if len(dm_files) != 1:
            f_t = "values csv" if file_type == self.val_filename else "pickle"
            if len(dm_files) == 0:
                # Instead of of data.pkl and disciplines_status.pkl, cache.pkl file is optional
                if file_type == self.cache_filename:
                    return None
                msg = f"There is no DM {f_t} file from {study_to_load}"
                raise FileNotFoundError(msg)
            _d = dm_files[0].parent
            _f = ", ".join([f.name for f in dm_files])
            msg = f"Too many DM {f_t} files from {study_to_load}, from {_d}: {_f}"
            raise FileNotFoundError(msg)
        return dm_files[0]

    def set_dm_pkl_files(self, study_to_load: Path | str) -> None:
        """Sets the path of the study pickle file.

        Args:
            study_to_load: The study directory.
        """
        self.dm_pkl_file = self.get_dm_file(study_to_load=study_to_load)

    def set_dm_val_files(self, study_to_load: Path | str) -> None:
        """Sets the path of the study value file.

        Args:
            study_to_load: The study directory.
        """
        self.dm_val_file = self.get_dm_file(study_to_load=study_to_load, file_type=self.val_filename)

    def put_dict_from_study(
        self, study_to_load: Path | str, rw_strategy: AbstractLoadDump, data_dict: dict[str, Any]
    ) -> None:
        """Write a dictionary to a pickle file.

        Args:
            study_to_load: The directory where to write the file.
            rw_strategy: The read/write strategy to use.
            data_dict: The dictionary to write.
        """
        study_to_load = Path(study_to_load)
        if not study_to_load.is_dir():
            makedirs_safe(study_to_load, exist_ok=True)

        # Export full data_dict to unique pickle file
        self.dm_pkl_file = study_to_load / self.pkl_filename

        # Serialise raw tree_node.data dict with pickle
        rw_strategy.dump(data_dict, self.dm_pkl_file)

    def put_cache_from_study(self, study_to_load: Path | str, rw_strategy: AbstractLoadDump, cache_map: dict[str, Any]):
        """Write a cache to a pickle file.

        Args:
            study_to_load: The directory where to write the file.
            rw_strategy: The read/write strategy to use.
            cache_map: The cache data to write.
        """
        study_to_load = Path(study_to_load)
        if not study_to_load.is_dir():
            makedirs_safe(study_to_load, exist_ok=True)

        # Export full cache_map to unique pickle file
        self.cache_file = study_to_load / self.cache_filename

        # Serialise raw cache_map with pickle
        rw_strategy.dump(cache_map, self.cache_file)

    def get_dict_from_study(self, study_to_load: Path | str, rw_strategy: AbstractLoadDump) -> dict[str, Any]:
        """Load a pickle file from a location and return a dictionary updated with loaded info.

        Args:
            study_to_load: The study directory.
            rw_strategy: The read/write strategy to use.

        Raises:
            RuntimeError: If the pickle file is empty.

        Returns:
            The data dictionary.
        """
        self.set_dm_pkl_files(study_to_load)
        return_data_dict = {}
        self.load_from_pickle(data_dict=return_data_dict, just_return_data_dict=True, rw_strategy=rw_strategy)
        if not return_data_dict:
            msg = f"Nothing to load from {study_to_load}"
            raise RuntimeError(msg)
        return return_data_dict

    def get_data_dict_from_pickle(self) -> dict[str, Any]:
        """Read data from pickled file.

        No need to get the unique pickle dm file path, it should exist already.
        """
        return read_pickle(self.dm_pkl_file)

    @staticmethod
    def convert_data_to_dataframe(data: DataFrame | list | ndarray | dict[str, Any] | float | int) -> DataFrame:
        """Convert data to DataFrame.

        Args:
            data: The data to convert.

        Returns:
            A dataframe containing the data.
        """
        if isinstance(data, DataFrame):
            df_data = data
        elif isinstance(data, (ndarray, list)):
            # Add header 'value'
            if len(data) == 0:
                df_data = DataFrame(columns=["value"])
            else:
                first_el = data[0]
                if isinstance(first_el, dict):
                    # get the keys of sub dict to make columns of dataframe
                    df_sub_col = []
                    for param_dict in data:
                        df_sub_col.extend(list(param_dict.keys()))
                    df_sub_col = list(set(df_sub_col))
                    df_col = ["variable", *list(df_sub_col)]
                    df_data = DataFrame(columns=df_col)
                    # iterate on keys of this dict and concatenate value
                    for k, a_d in enumerate(data):
                        # append built dataframe to global one
                        df_sub_col = a_d.keys()
                        a_df = DataFrame([a_d.values()], columns=df_sub_col)
                        df_data = concat([df_data, a_df.assign(variable=k)], sort=False)
                    df_data = df_data[df_col]
                else:
                    df_data = DataFrame(data, columns=["value"])
        elif isinstance(data, dict):
            if len(data.keys()) == 0:
                df_data = DataFrame(columns=["variable", "value"])
            else:
                # if data is a dict, look at the content type
                first_el = data[next(iter(data.keys()))]

                if isinstance(first_el, DataFrame):
                    # use the columns of sub df as columns of dataframe
                    df_col = first_el.columns
                    df_col = df_col.insert(0, "variable")
                    df_data = DataFrame(columns=df_col)
                    # iterate on keys of this dict and concatenate value
                    for k, a_df in data.items():
                        # append built dataframe to global one
                        df_data = concat([df_data, a_df.assign(variable=k)], sort=False)
                    df_data = df_data[df_col]
                else:
                    # dict of values, so just add header 'value'
                    df_data = DataFrame(data.items(), columns=["variable", "value"])
        else:
            # # single value as scalar, compose a dataframe anyway adding header 'value'
            df_data = DataFrame([data], columns=["value"])
        return df_data

    def convert_to_dataframe_and_bytes_io(
        self, param_values: DataFrame | list | ndarray | dict[str, Any] | float | int, param_key: str
    ) -> BytesIO:
        """Convert a parameter to a dataframe, and then to a BytesIO object.

        Args:
            param_values: The parameter values.
            param_key: The parameter name.

        Returns:
            A BytesIO object containing the parameter values.
        """
        df_data = self.convert_data_to_dataframe(param_values)
        return self.convert_to_bytes_io(df_data, param_key)

    def convert_model_table_to_df_and_bytes_io(self, model_table, param_key: str):
        """Convert a model table to a dataframe, and then to a BytesIO object.

        Args:
            model_table: The model_table to convert.
            param_key: The parameter name.

        Returns:
            A BytesIO object containing the model table.
        """
        df_data = DataFrame(model_table[1:], columns=model_table[0])
        return self.convert_to_bytes_io(df_data, param_key)

    def convert_model_table_to_df(self, model_table):
        """Convert a model table to a dataframe.

        Args:
            model_table: The model_table to convert.

        Returns:
            A dataframe containing the model table.
        """
        return DataFrame(model_table[1:], columns=model_table[0])

    def convert_to_bytes_io(self, df_data: DataFrame, data_key: str):
        """Export data as a DataFrame using buffered I/O streams.

        Args:
            df_data: The dataframe to export.
            data_key: The name of the parameter to export.

        Raises:
            RuntimeError: If the conversion to BytesIO fails.

        Returns:
            A BytesIO object containing the data.
        """
        try:
            df_stream = StringIO()
            df_data.to_csv(df_stream, sep=CSV_SEP, header=True, index=False)
            df_bytes = df_stream.getvalue().encode()
            # just return the in-memory bytes buffer
            return BytesIO(df_bytes)
        except Exception as error:
            raise RuntimeError(f"ERROR converting {data_key} value to bytes: " + str(error)) from error
