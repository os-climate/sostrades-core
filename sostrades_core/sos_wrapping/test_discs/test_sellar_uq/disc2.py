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

import logging
from typing import Any, ClassVar

from numpy import array, sqrt

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

LOGGER = logging.getLogger(__name__)


class SellarDisc2(SoSWrapp):
    """The second discipline of the Sellar problem."""

    _ontology_data: ClassVar = {
        "label": "SellarDisc2",
        "type": "Research",
        "source": "Vincent Drouet",
        "version": "",
    }
    """The ontology debugrmation of the model."""

    DESC_IN: ClassVar = {
        "z": {
            "type": "array",
            "default": array([1, 1]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
        "y1": {
            "type": "array",
            "default": array([1.0]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
    }
    """The input variables."""

    DESC_OUT: ClassVar = {
        "y2": {
            "type": "array",
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
    }
    """The output variables."""

    def run(self) -> dict[str, Any]:
        """Run the model.

        Returns: The output of the model.
        """
        inputs = self.get_sosdisc_inputs()
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        y1 = inputs["y1"]

        y2 = sqrt(abs(y1)) + z1 + z2

        LOGGER.debug("Disc 2")
        LOGGER.debug("z1 = %s", z1)
        LOGGER.debug("z2 = %s", z2)
        LOGGER.debug("y1 = %s", y1)
        LOGGER.debug("y2 = %s", y2)
        LOGGER.debug("")

        self.store_sos_outputs_values({"y2": y2})

    def compute_sos_jacobian(self) -> None:
        """Compute the analytic jacobian."""
        y1 = self.get_sosdisc_inputs()["y1"]

        self.set_partial_derivative("y2", "z", array([[1, 1]]))
        self.set_partial_derivative("y2", "y1", array([[1 / (2 * sqrt(y1))]]))
