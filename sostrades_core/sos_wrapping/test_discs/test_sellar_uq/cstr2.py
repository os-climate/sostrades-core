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
