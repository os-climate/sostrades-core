from __future__ import annotations
from gemseo.caches.simple_cache import SimpleCache
from typing import TYPE_CHECKING
from sostrades_core.tools.compare_data_manager_tooling import dict_are_equal
# if TYPE_CHECKING:
# from gemseo.typing import StrKeyMapping
from gemseo.caches.cache_entry import CacheEntry

SOS_DATA_COMPARATOR = dict_are_equal
"""Caching module to store only one entry."""

from typing import TYPE_CHECKING

from gemseo.caches.base_cache import DATA_COMPARATOR
from gemseo.caches.base_cache import BaseCache
from gemseo.caches.cache_entry import CacheEntry
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.typing import JacobianData
    from gemseo.typing import StrKeyMapping


class SoSSimpleCache(SimpleCache):
    """Dictionary-based cache storing a unique entry."""

    __inputs: StrKeyMapping
    """The input data."""

    __outputs: StrKeyMapping
    """The output data."""

    __jacobian: JacobianData
    """The Jacobian data."""

    def __init__(  # noqa:D107
        self,
        tolerance: float = 0.0,
        name: str = "",
    ) -> None:
        super(SimpleCache, self).__init__(tolerance, name)
        self.clear()

    def clear(self) -> None:  # noqa:D102
        super().clear()
        self.__inputs = {}
        self.__outputs = {}
        self.__jacobian = {}

    def get_all_entries(self) -> Iterator[CacheEntry]:  # noqa: D102
        if self.__inputs:
            yield self.last_entry

    def __len__(self) -> int:
        return 1 if self.__inputs else 0

    def cache_outputs(  # noqa:D102
        self,
        input_data: StrKeyMapping,
        output_data: StrKeyMapping,
    ) -> None:
        if self.__is_cached(input_data):
            if not self.__outputs:
                self.__outputs = deepcopy_dict_of_arrays(output_data)
            return

        self.__inputs = deepcopy_dict_of_arrays(input_data)
        self.__outputs = deepcopy_dict_of_arrays(output_data)
        self.__jacobian = {}

        if not self._output_names:
            self._output_names = sorted(output_data.keys())

    def __getitem__(
        self,
        input_data: StrKeyMapping,
    ) -> CacheEntry:
        if not self.__is_cached(input_data):
            return CacheEntry(input_data, {}, {})
        return self.last_entry

    def cache_jacobian(  # noqa:D102
        self,
        input_data: StrKeyMapping,
        jacobian_data: JacobianData,
    ) -> None:
        if self.__is_cached(input_data):
            if not self.__jacobian:
                self.__jacobian = jacobian_data
            return

        self.__inputs = deepcopy_dict_of_arrays(input_data)
        self.__jacobian = jacobian_data
        self.__outputs = {}

    @property
    def last_entry(self) -> CacheEntry:  # noqa:D102
        return CacheEntry(self.__inputs, self.__outputs, self.__jacobian)

    def __is_cached(
        self,
        input_data,  # : StrKeyMapping,
    ) -> bool:
        """Check if an input data is cached.

        Args:
            input_data: The input data to be verified.

        Returns:
            Whether the input data is cached.
        """
        #TODO: TOLERANCE IS UNUSED
        return len(self.__inputs) != 0 and SOS_DATA_COMPARATOR(
            input_data, self.__inputs, self._tolerance
        )

