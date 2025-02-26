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
        """
        Run the model.

        Returns: The output of the model.
        """
        inputs = self.get_sosdisc_inputs()
        y1 = inputs["y1"]
        g1 = 3.16 - y1

        self.store_sos_outputs_values({"g1": g1})

    def compute_sos_jacobian(self) -> None:
        """Compute the analytic jacobian."""
        self.set_partial_derivative("g1", "y1", array([[-1]]))
