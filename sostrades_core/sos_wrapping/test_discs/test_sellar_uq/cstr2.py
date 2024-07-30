# Copyright (c) 2023 Capgemini Engineering
# All rights reserved.
#
# Created on 18/04/2024, 11:17
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The second Sellar constraint wrapped in SoSTrades."""

from __future__ import annotations

from typing import Any, ClassVar

from numpy import array

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class SellarCstr2(SoSWrapp):
    """The second constraint of the Sellar problem."""

    _ontology_data: ClassVar = {
        "label": "SellarCstr1",
        "type": "Research",
        "source": "Vincent Drouet",
        "version": "",
    }
    """The ontology information of the model."""

    DESC_IN: ClassVar = {
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
        "g2": {
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
        y2 = inputs["y2"]

        g2 = y2 - 24

        self.store_sos_outputs_values({"g2": g2})

    def compute_sos_jacobian(self) -> None:
        """Compute the analytic jacobian."""
        self.set_partial_derivative("g2", "y2", array([[1]]))
