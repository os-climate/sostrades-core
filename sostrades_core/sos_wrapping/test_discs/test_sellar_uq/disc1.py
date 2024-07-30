# Copyright (c) 2023 Capgemini Engineering
# All rights reserved.
#
# Created on 18/04/2024, 11:17
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The first Sellar discipline wrapped in SoSTrades."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from numpy import array

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

LOGGER = logging.getLogger(__name__)


class SellarDisc1(SoSWrapp):
    """The first discipline of the Sellar problem."""

    _ontology_data: ClassVar = {
        "label": "SellarDisc1",
        "type": "Research",
        "source": "Vincent Drouet",
        "version": "",
    }
    """The ontology debugrmation of the model."""

    DESC_IN: ClassVar = {
        "x": {
            "type": "array",
            "default": array([2.0]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
        "z": {
            "type": "array",
            "default": array([1, 1]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
        "y2": {
            "type": "array",
            "default": array([1.0]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
    }
    """The input variables."""

    DESC_OUT: ClassVar = {
        "y1": {
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
        x = inputs["x"]
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        y2 = inputs["y2"]

        y1 = z1**2 + z2 + x - 0.2 * y2

        LOGGER.debug("Disc 1")
        LOGGER.debug("x = %s", x)
        LOGGER.debug("z1 = %s", z1)
        LOGGER.debug("z2 = %s", z2)
        LOGGER.debug("y2 = %s", y2)
        LOGGER.debug("y1 = %s", y1)
        LOGGER.debug("")

        self.store_sos_outputs_values({"y1": y1})

    def compute_sos_jacobian(self):
        """Compute the analytic jacobian."""
        z1 = self.get_sosdisc_inputs()["z"][0]

        self.set_partial_derivative("y1", "x", array([[1]]))
        self.set_partial_derivative("y1", "z", array([[2 * z1, 1]]))
        self.set_partial_derivative("y1", "y2", array([[-0.2]]))
