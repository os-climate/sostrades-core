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

"""Most basic grammar implementation."""

from typing import Any, Final
from numpy import ndarray
from numpy import complex128 as np_complex128
from collections import defaultdict
from gemseo.core.data_converters.base import _NUMERIC_TYPES
from gemseo.core.data_converters.simple import SimpleGrammarDataConverter
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_array_into_new_type, \
    convert_new_type_into_array, STANDARD_TYPES
from sostrades_core.tools.base_functions.compute_len import compute_len

ValueTypes: Final[tuple[type]] = tuple(STANDARD_TYPES + [complex, ndarray, np_complex128])
ValueTypes_Numeric = ['int', 'float']


class SoSTradesDataConverter(SimpleGrammarDataConverter):
    """Data values to NumPy arrays and vice versa from a :class:`.SimpleGrammar`."""

    def __init__(self, grammar):
        super().__init__(grammar)
        self.reduced_dm = defaultdict(dict)

    def is_numeric(self, name: str) -> bool:  # noqa: D102
        element_type = self._grammar[name]
        return element_type is not None and (
                issubclass(element_type, ndarray) or element_type in _NUMERIC_TYPES
        )

    def _convert_array_to_value(self, name: str, array: ndarray) -> Any:  # noqa: D102

        if name not in self.reduced_dm or self.reduced_dm[name]['type'] == 'array':
            return array
        elif self.reduced_dm[name]['type'] in ValueTypes_Numeric:
            return array[0]
        else:
            return convert_array_into_new_type(name, array, self.reduced_dm.get(name, {}))

    def convert_value_to_array(
            self,
            name: str,
            value,
    ) -> ndarray:
        """Convert a data value to a NumPy array.

        Args:
            name: The data name.
            value: The data value.

        Returns:
            The NumPy array.
        """
        if type(value) in ValueTypes:
            return super().convert_value_to_array(name, value)
        else:
            val_converted, new_reduced_dm = convert_new_type_into_array(name, value, self.reduced_dm.get(name, {}))
            self.reduced_dm[name].update(new_reduced_dm)
            return val_converted

    @staticmethod
    def get_value_size(name: str, value) -> int:
        """Return the size of a data value.

        The size is typically what is returned by ``ndarray.size`` or ``len(list)``.
        The size of a number is 1.

        Args:
            name: The data name.
            value: The data value to get the size from.

        Returns:
            The size.
        """
        if isinstance(value, _NUMERIC_TYPES):
            return 1
        else:
            return compute_len(value)
