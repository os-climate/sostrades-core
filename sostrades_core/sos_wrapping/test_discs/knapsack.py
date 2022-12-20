"""
Copyright 2022 Airbus SAS

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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
from sys import maxsize

from numpy import ones, inf, array

from gemseo.problems.analytical.knapsack import Knapsack
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class KnapsackProblem(SoSWrapp):
    """Generic knapsack discipline."""

    # ontology information
    _ontology_data = {
        "label": "sos_trades_core.sos_wrapping.test_discs.knapsack",
        "type": "Research",
        "source": "SoSTrades Project",
        "validated": "",
        "validated_by": "SoSTrades Project",
        "last_modification_date": "",
        "category": "",
        "definition": "",
        "icon": "",
        "version": "",
    }

    _maturity = "Fake"

    DESC_IN = {
        "x": {"type": "array", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "items_value": {"type": "array", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "items_weight": {"type": "array", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "capacity_items": {
            "type": "int", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY, "default": maxsize
        },
        "capacity_weight": {"type": "float", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY, "default": inf},
        "binary": {"type": "bool", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY, "default": True},
    }

    DESC_OUT = {
        "n_items": {"type": "int", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "value": {"type": "float", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "items_and_-value": {"type": "array", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "weight": {"type": "float", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "excess_items": {"type": "int", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
        "excess_weight": {"type": "float", "unit": "-", "visibility": SoSWrapp.LOCAL_VISIBILITY},
    }

    def init_execution(self):
        values, weights = self.proxy.get_sosdisc_inputs(["items_value", "items_weight"])
        capacity_items, capacity_weight = self.proxy.get_sosdisc_inputs(["capacity_items", "capacity_weight"])
        n_items_available = len(values)
        self.model = Knapsack(
            values,
            weights,
            capacity_weight=capacity_weight,
            capacity_items=capacity_items,
            initial_guess=ones(n_items_available),
        )

    def run(self):
        # x = self.get_sosdisc_inputs(["x"])
        x = self.get_sosdisc_inputs()["x"]
        capacity_items, capacity_weight = self.get_sosdisc_inputs(["capacity_items", "capacity_weight"])

        weight = self.model.compute_knapsack_weight(x)
        value = self.model.compute_knapsack_value(x)[0]
        n_items = int(self.model.compute_knapsack_items(x))

        # Constraint on the weight.
        c_weight = weight - capacity_weight

        # Constraint on the number of items.
        c_items = n_items - capacity_items

        self.store_sos_outputs_values({
            "n_items": n_items,
            "value": value,
            "items_and_-value": array([n_items, -value]),
            "weight": weight,
            "excess_items": c_items,
            "excess_weight": c_weight,
        })
