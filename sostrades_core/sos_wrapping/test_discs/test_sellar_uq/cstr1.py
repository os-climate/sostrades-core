# Copyright (c) 2023 Capgemini Engineering
# All rights reserved.
#
# Created on 18/04/2024, 11:17
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The first Sellar constraint wrapped in SoSTrades."""

from __future__ import annotations

from typing import ClassVar

from numpy import array

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class SellarCstr1(SoSWrapp):
    """The first constraint of the Sellar problem."""

    _ontology_data: ClassVar = {
        "label": "SellarCstr1",
        "type": "Fake",
        "source": "Vincent Drouet",
        "version": "",
    }
    """The ontology information of the model."""

    DESC_IN: ClassVar = {
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
        "g1": {
            "type": "array",
            "unit": "-",
            "visibility": SoSWrapp.SHARED_VISIBILITY,
            "namespace": "ns_sellar",
        },
    }
    """The output variables."""

    def run(self) -> None:
        """Run the model.

        Returns: The output of the model.
        """
        inputs = self.get_sosdisc_inputs()
        y1 = inputs["y1"]
        g1 = 3.16 - y1

        self.store_sos_outputs_values({"g1": g1})

    def compute_sos_jacobian(self) -> None:
        """Compute the analytic jacobian."""
        self.set_partial_derivative("g1", "y1", array([[-1]]))
