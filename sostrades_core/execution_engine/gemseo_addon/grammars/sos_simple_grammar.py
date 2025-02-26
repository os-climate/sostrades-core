'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2024/05/16 Copyright 2023 Capgemini

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

import logging
from typing import ClassVar

from gemseo.core.grammars.simpler_grammar import SimplerGrammar

"""Most basic grammar implementation."""

LOGGER = logging.getLogger("gemseo.addons.grammars.sos_simple_grammar")


class SoSSimpleGrammar(SimplerGrammar):
    """
    Store the names and types of the elements as Python lists.

    Attributes:
        data_names (List[str]): The names of the elements.
        data_types (List[type]): The types of the elements,
            stored in the same order as ``data_names``.

    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "SoSTradesDataConverter"

    def update_defaults(self, defaults: dict):
        self._defaults.update({
            k: v for k, v in defaults.items() if k in self.names
        })
