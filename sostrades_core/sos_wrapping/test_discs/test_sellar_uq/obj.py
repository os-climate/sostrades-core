# Copyright (c) 2023 Capgemini Engineering
# All rights reserved.
#
# Created on 18/04/2024, 11:17
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Sellar objective discipline wrapped in SoSTrades."""

from __future__ import annotations

from typing import Any, ClassVar

from numpy import array, exp

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class SellarObj(SoSWrapp):
    """The objective discipline of the Sellar problem."""

    _ontology_data: ClassVar = {
        "label": "SellarObjDisc",
        "type": "Research",
        "source": "Vincent Drouet",
        "version": "",
    }
    """The ontology information of the model."""

    DESC_IN: ClassVar = {
        "x": {
            "type": "array",
            "default": array([1.0]),
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
        "z": {
            "type": "array",
            "default": array([0, 0]),
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
        "f": {
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
        z2 = inputs["z"][1]
        y1 = inputs["y1"]
        y2 = inputs["y2"]

        f = x**2 + z2 + y1 + exp(-y2)

        self.store_sos_outputs_values({"f": f})

    def compute_sos_jacobian(self) -> None:
        """Compute the analytic jacobian."""
        inputs = self.get_sosdisc_inputs()
        x = inputs["x"]
        y2 = inputs["y2"]

        self.set_partial_derivative("f", "x", array([[2 * x]]))
        self.set_partial_derivative("f", "z", array([[0, 1]]))
        self.set_partial_derivative("f", "y1", array([[1]]))
        self.set_partial_derivative("f", "y2", array([[-exp(-y2)]]))
