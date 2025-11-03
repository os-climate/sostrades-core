'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/31-2024/06/11 Copyright 2023 Capgemini

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

import csv
import os
from typing import Any

import numpy as np

from sostrades_core.tools.folder_operations import makedirs_safe

from .diagram import Diagram


class LoggerData:
    """Pickable class used by logger to store data and draw plots when required."""

    SPECIAL_UNITS = ['(str)', '(string)']

    def __init__(self, tag: str, main_var: str = 'time', main_unit: str = '(s)', out_save_dir: str = 'Outputs') -> None:
        """
        Initialize logger data container.

        Args:
            tag: Identifier tag for this logger instance.
            main_var: Name of the main variable (default 'time').
            main_unit: Unit of the main variable (default '(s)').
            out_save_dir: Output directory for saving data (default 'Outputs').

        """
        self.base_save_dir = out_save_dir
        self.__tag = tag
        self.__main_var = main_var
        self.__main_unit = main_unit
        self.__object_dict: dict[str, Any] = dict()
        self.__database: dict[str, Any] = dict()
        self.__units: dict[str, str] = dict()
        self.__diagramm_dict: dict[str, Diagram] = dict()
        self.__index = 0

        self.__database[self.__main_var] = None
        self.__units[self.__main_var] = self.__main_unit

        self.set_complex_mode(False)

        # -- Clean Outputs directory at instantiation
        full_save_dir = self.get_full_save_dir()
#         if os.path.isdir(full_save_dir):
#             shutil.rmtree(full_save_dir)
        if not os.path.isdir(full_save_dir):
            makedirs_safe(full_save_dir)

    # -- Setters
    def set_complex_mode(self, complex_mode: bool) -> None:
        """
        Set the complex mode in the database.

        Args:
            complex_mode: Whether to use complex number types.

        """
        self.complex_mode = complex_mode
        if complex_mode:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

    # -- Accessors
    def get_units(self, var_id: str) -> str:
        """
        Get the units of a variable.

        Args:
            var_id: Variable identifier to get units for.

        Returns:
            Units string for the variable.

        Raises:
            Exception: If variable is not found in units dictionary.

        """
        if var_id in self.__units:
            return self.__units[var_id]
        else:
            raise Exception(var_id + " is not in units")

    def get_var_ids(self) -> list[str]:
        """
        Get list of all variable names in the logger database.

        Returns:
            List of variable identifiers.

        """
        return list(self.__database.keys())

    def get_data(self, var_id: str) -> Any:
        """
        Get the data for a variable.

        Args:
            var_id: Variable identifier to get data for.

        Returns:
            Data associated with the variable.

        Raises:
            Exception: If variable is not found in database.

        """
        if var_id in self.__database:
            return self.__database[var_id]
        else:
            raise Exception(var_id + " is not in database")

    def get_full_save_dir(self, save_dir: str | None = None) -> str:
        """
        Get the path of the saving directory.

        Args:
            save_dir: Optional directory path. If None, uses base_save_dir.

        Returns:
            Full path to the save directory.

        """
        if save_dir is None:
            save_dir = self.base_save_dir
        return os.path.join(save_dir, self.__tag)

    # -- Methods
    def add_unit(self, d_id: str, value: str) -> None:
        """
        Add a unit to the units dictionary to the name d_id.

        Args:
            d_id: Dictionary identifier.
            value: Unit string to add.

        """
        self.__units[d_id] = value

    def add_to_database(self, d_id: str, value: Any) -> None:
        """
        Add a value to the database with the name d_id.

        Args:
            d_id: Database identifier.
            value: Value to add to the database.

        """
        self.__database[d_id] = value

    def add_to_database_index(self, d_id: str, index: int, value: Any) -> None:
        """
        Add a value corresponding to a specific index (or time) in the database.

        Args:
            d_id: Database identifier.
            index: Array index to set.
            value: Value to add at the specified index.

        """
        self.__database[d_id][index] = value

    def add_to_main_var_index(self, index: int, value: Any) -> None:
        """
        Add the main variable (the time by default) to the database at a specific index.

        Args:
            index: Array index to set.
            value: Value to add at the specified index.

        """
        self.add_to_database_index(self.__main_var, index, value)

    def init_database(self, list_keys: list[str], dim: int) -> None:
        """
        Initialize the database with a list of keys (+ time) and a dimension.

        The type of the data is real or complex depending on the complex mode
        except special_units (which are strings).

        Args:
            list_keys: List of variable keys to initialize.
            dim: Dimension for array initialization.

        """
        full_list_keys = [self.__main_var] + list_keys
        for d_id in full_list_keys:
            d_unit = self.get_units(d_id)
            if d_unit not in self.SPECIAL_UNITS:
                dtype = self.dtype
            else:
                dtype = object
            self.__database[d_id] = np.zeros(dim, dtype=dtype)

    def init_single_var_data(self, d_id: str) -> None:
        """
        Initialize only one value d_id in the database to zero.

        Args:
            d_id: Variable identifier to initialize.

        """
        d_unit = self.get_units(d_id)
        if d_unit not in self.SPECIAL_UNITS:
            dtype = self.dtype
        else:
            dtype = object
        if self.__database[self.__main_var] is not None:
            self.__database[d_id] = np.zeros(
                len(self.__database[self.__main_var]), dtype=dtype)

    def add_diagram(self, name: str, x_axis: str, y_axis: str,
                    save_dir: str | None = None, sub_save_dir: str | None = None, y_log_scale: bool = False) -> None:
        """
        Add a diagram in the diagram dictionary.

        If the diagram already exists, a curve is gathered to the diagram.

        Args:
            name: Name of the diagram.
            x_axis: X-axis variable name.
            y_axis: Y-axis variable name.
            save_dir: Optional directory to save the diagram.
            sub_save_dir: Optional subdirectory to save the diagram.
            y_log_scale: Whether to use logarithmic scale for Y-axis.

        """
        if name not in self.__diagramm_dict:
            self.__diagramm_dict[name] = Diagram(
                self, name, x_axis, y_axis, save_dir=save_dir,
                sub_save_dir=sub_save_dir, y_log_scale=y_log_scale)
        else:
            self.__diagramm_dict[name].add_curve(y_axis)

    def resize(self, new_dim: int | None = None) -> None:
        """
        Modify the dimension of a variable.

        If new dim is not specified, last zeros of the vector are erased and the vector is resized.

        Args:
            new_dim: New dimension size. If None, automatically calculated by removing trailing zeros.

        """
        if new_dim is None:
            try:
                # remove last rows if time is 0
                ndim = len(self.__database[self.__main_var])
                while self.__database[self.__main_var][ndim - 1] == 0 and ndim > 1:
                    ndim -= 1
                new_dim = ndim
            except:
                return
        for var_id in self.__database:
            self.__database[var_id].resize(new_dim)

    def export_to_file(self) -> None:
        """Export the database to the file variable_log.csv."""
        save_dir = self.get_full_save_dir()
        filename = os.path.join(save_dir, self.__tag + '.variables_log.csv')
        fid = open(filename, 'w', newline='')

        rows_list = []
        tmp_keys = sorted(self.__database.keys())
        tmp_keys.remove(self.__main_var)
        list_keys = [self.__main_var] + tmp_keys
        rows_list.append(list_keys)

        for index in range(len(self.__database[self.__main_var])):
            list_values = []
            for var_name in list_keys:
                value = self.__database[var_name][index]
                list_values.append(value)
            rows_list.append(list_values)
        c_file = csv.writer(fid)
        c_file.writerows(rows_list)
        fid.close()

    def plot(self) -> None:
        """Plot all the diagrams added in the diagram dictionary."""
        for diagram in list(self.__diagramm_dict.values()):
            diagram.plot()
