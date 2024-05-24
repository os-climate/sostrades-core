"""
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
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import pandas as pd


@dataclass
class ArchiNode:
    """
    A class representing an ArchiNode.

    Attributes:
        name (str): The name of the node.
        parent (str): The name of the parent node.
        type (str): The type of the node, default is "ValueBlockDiscipline".
        action (Union[Tuple[str], str]): The action associated with the node, default is "standard".
        activation (bool): Whether the node is activated or not, default is False.
        children (List["ArchiNode"]): A list of child nodes, default is an empty list.
    """

    name: str = ""
    parent: str = None
    type: str = "ValueBlockDiscipline"
    action: Union[Tuple[str], str] = "standard"
    activation: bool = False
    children: List["ArchiNode"] = field(default_factory=list)

    def __post_init__(self):
        """
        Update the parent name in all child nodes after initialization.
        """
        self.update_parent_name_in_children()

    def update_parent_name_in_children(self):
        """
        Update the parent name in all child nodes.

        This method recursively updates the parent name of all child nodes to the current node's name.
        """
        for c in self.children:
            c.parent = self.name
            c.update_parent_name_in_children()

    def get_field_as_list(self, field_name: str, skip_self=False) -> List[str]:
        """
        Get the specified field as a list of values.

        Args:
            field_name (str): The name of the field to retrieve.
            skip_self (bool, optional): Whether to skip the current node's value. Defaults to False.

        Returns:
            List[str]: A list of values for the specified field.
        """
        if skip_self:
            fields = []
        else:
            fields = [getattr(self, field_name)]

        for c in self.children:
            fields += c.get_field_as_list(field_name)

        return fields

    def to_dataframe(self, skip_self=False) -> pd.DataFrame:
        """
        Generate a pandas DataFrame where each column is an attribute of the class (except the children),
        and each row contains the information of each child (the first row is the current node).

        Returns:
            pd.DataFrame: A pandas DataFrame containing the node information.
        """

        data = {
            "Parent": self.get_field_as_list("parent", skip_self=skip_self),
            "Current": self.get_field_as_list("name", skip_self=skip_self),
            "Type": self.get_field_as_list("type", skip_self=skip_self),
            "Action": self.get_field_as_list("action", skip_self=skip_self),
            "Activation": self.get_field_as_list("activation", skip_self=skip_self),
        }
        return pd.DataFrame(data)
