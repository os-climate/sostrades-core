'''
Copyright 2022 Airbus SAS
Modifications on 2024/07/24 Copyright 2024 Capgemini

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

from typing import Any

"""Post processing bundle model."""


class PostProcessingBundle:
    """Class that holds filter and post processing bundle data."""

    NAME = 'name'
    DISCIPLINE_NAME = 'discipline_name'
    FILTERS = 'filters'
    POST_PROCESSINGS = 'post_processings'

    def __init__(self, name: str, discipline_name: str, filters: list[Any], post_processings: list[Any]) -> None:
        """
        Initialize post processing bundle.

        Args:
            name: Name of current post processings bundle.
            discipline_name: Name of discipline hosting the post processings bundle.
            filters: Filter list used for this post-processings bundle.
            post_processings: List of post-processings in the bundle.

        """
        self.name = name
        self.discipline_name = discipline_name
        self.filters = filters
        self.post_processings = post_processings

    @property
    def has_post_processings(self) -> bool:
        """Check if bundle has any filters."""
        return len(self.filters) > 0

    def __repr__(self) -> str:
        """Return string representation of the bundle."""
        series_string = [f'\nname: {self.name}',
                         f'discipline_name: {self.discipline_name}',
                         f'filters: {self.filters}',
                         f'post-processings: {self.post_processings}'
                         ]

        return '\n'.join(series_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize bundle as dictionary.

        Returns:
            Dictionary representation of the bundle.

        """
        dict_obj = {}
        # Serialize name attribute
        dict_obj.update({PostProcessingBundle.NAME: self.name})

        # Serialize discipline name attribute
        dict_obj.update({PostProcessingBundle.DISCIPLINE_NAME: self.discipline_name})

        # Serialize filters parameter attribute
        dict_obj.update(
            {PostProcessingBundle.FILTERS: self.filters})

        # Serialize post-processings values parameter attribute
        dict_obj.update(
            {PostProcessingBundle.POST_PROCESSINGS: self.post_processings})

        return dict_obj

    @staticmethod
    def from_dict(dict_obj: dict[str, Any]) -> PostProcessingBundle:
        """
        Initialize bundle from dictionary.

        Args:
            dict_obj: Dictionary containing bundle data.

        Returns:
            PostProcessingBundle instance created from dictionary.

        """
        # Deserialize name attribute
        name = dict_obj[PostProcessingBundle.NAME]

        # Deserialize discipline name attribute
        discipline_name = ''
        if PostProcessingBundle.DISCIPLINE_NAME in dict_obj:
            discipline_name = dict_obj[PostProcessingBundle.DISCIPLINE_NAME]

        # Deserialize filters parameter attribute
        filters = dict_obj[PostProcessingBundle.FILTERS]

        # Deserialize post-processings values parameter attribute
        post_processings = dict_obj[PostProcessingBundle.POST_PROCESSINGS]

        return PostProcessingBundle(name, discipline_name, filters, post_processings)
