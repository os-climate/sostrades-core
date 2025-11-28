'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16-2025/02/14 Copyright 2025 Capgemini

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

import numpy as np
import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp



class CarCostProblem(SoSWrapp):
    """Car Total Cost Computation Problem"""

    _maturity = 'Fake'
    DESC_IN = {
        'engine_power': {'type': 'dict', 'subtype_descriptor': {'dict': 'array'},
              'namespace': 'ns_Optim'},
        'manufacturing_cost': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
        'maintenance_cost': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
        'material_specs': {'type': 'array',  'namespace': 'ns_Optim'},
        'weight_factor': {'type': 'float'}}

    DESC_OUT = {'quality_constraint': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
                'budget_constraint': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
                'total_cost': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }}}

    def run(self):
        """Computes car cost analysis"""
        engine_power, manufacturing_cost, maintenance_cost, material_specs = self.get_sosdisc_inputs(['engine_power', 'manufacturing_cost', 'maintenance_cost', 'material_specs'])
        weight_factor = self.get_sosdisc_inputs('weight_factor')

        total_cost = self.compute_total_cost(engine_power, material_specs, manufacturing_cost, maintenance_cost, weight_factor)
        quality_constraint = self.compute_quality_constraint(manufacturing_cost)
        budget_constraint = self.compute_budget_constraint(maintenance_cost)
        out = {'total_cost': total_cost, 'quality_constraint': quality_constraint, 'budget_constraint': budget_constraint}
        self.store_sos_outputs_values(out)

    def compute_total_cost(self, engine_power, material_specs, manufacturing_cost, maintenance_cost, weight_factor):
        """
        Compute total car cost over years

        :param engine_power: engine power specifications
        :type engine_power: dict
        :param material_specs: material quality specifications
        :type material_specs: numpy.array
        :param manufacturing_cost: manufacturing cost per year
        :type manufacturing_cost: dataframe
        :param maintenance_cost: maintenance cost per year
        :type maintenance_cost: dataframe
        :param weight_factor: vehicle weight factor for DOE
        :type weight_factor: float
        :returns: Total cost per year
        :rtype: dataframe
        """
        out = pd.DataFrame({'years': manufacturing_cost['years'], 'value': [0]*len(manufacturing_cost)})
        # Simple formula: base cost + engine cost + material cost + weight penalty + maintenance
        out['value'] = (engine_power['value'][0] * 100 + material_specs[1] * 50 + 
                       manufacturing_cost['value'] * weight_factor + 
                       maintenance_cost['value'] * 1.2)
        return out

    def compute_quality_constraint(self, manufacturing_cost):
        """
        Quality constraint: manufacturing cost should not be too low (minimum quality threshold)

        :param manufacturing_cost: manufacturing cost from discipline 1
        :type manufacturing_cost: dataframe
        :returns: Quality constraint value (should be >= 0)
        :rtype: dataframe
        """
        return pd.DataFrame({'years': manufacturing_cost['years'], 'value': [val - 5000 for val in manufacturing_cost['value']]})

    def compute_budget_constraint(self, maintenance_cost):
        """
        Budget constraint: maintenance cost should not exceed budget limit

        :param maintenance_cost: maintenance cost from discipline 2
        :type maintenance_cost: dataframe
        :returns: Budget constraint value (should be <= 0)
        :rtype: dataframe
        """
        return pd.DataFrame({'years': maintenance_cost['years'], 'value': [val - 15000 for val in maintenance_cost['value']]})


class ManufacturingDisc(SoSWrapp):
    """Car Manufacturing Cost Discipline"""

    _maturity = 'Fake'
    DESC_IN = {'engine_power': {'type': 'dict', 'subtype_descriptor': {'dict': 'array'},  'namespace': 'ns_Optim'},
               'maintenance_cost': {'type': 'dataframe',
                       'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
               'material_specs': {'type': 'array',  'namespace': 'ns_Optim'}}

    DESC_OUT = {'manufacturing_cost': {'type': 'dataframe',
                         'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }}}

    def run(self):
        """Manufacturing discipline execution"""
        engine_power, maintenance_cost, material_specs = self.get_sosdisc_inputs(['engine_power', 'maintenance_cost', 'material_specs'])
        manufacturing_cost = self.compute_manufacturing_cost(engine_power, maintenance_cost, material_specs)
        manufacturing_out = {'manufacturing_cost': manufacturing_cost}
        self.store_sos_outputs_values(manufacturing_out)

    def compute_manufacturing_cost(self, engine_power, maintenance_cost, material_specs):
        """
        Compute manufacturing cost based on engine power, maintenance feedback, and material specifications.

        :param engine_power: engine power specifications
        :type engine_power: dict
        :param maintenance_cost: maintenance cost feedback from discipline 2
        :type maintenance_cost: dataframe
        :param material_specs: material quality and cost specifications
        :type material_specs: numpy.array
        :returns: manufacturing cost per year
        :rtype: dataframe
        """
        maintenance_cost['years'] = maintenance_cost['years'].astype('int64')
        out = pd.DataFrame({'years': maintenance_cost['years'], 'value': 0.0})

        # Manufacturing cost = base material cost + engine complexity cost + maintenance influence
        out['value'] = (int(material_specs[0]) * 200 + float(engine_power['value'][0]) * 80 + 
                       float(material_specs[1]) * 150 + 0.1 * maintenance_cost['value'])

        return out



class MaintenanceDisc(SoSWrapp):
    """Car Maintenance Cost Discipline"""

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.car_cost_computation',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'manufacturing_cost': {'type': 'dataframe',  'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }},
        'material_specs': {'type': 'array',  'namespace': 'ns_Optim'}}

    DESC_OUT = {'maintenance_cost': {'type': 'dataframe',
                         'namespace': 'ns_Optim', "dataframe_descriptor": {
                    'years': ("int", None, True),
                    'value': ("float", None, True)
                }}}

    def run(self):
        """Maintenance discipline execution"""
        manufacturing_cost, material_specs = self.get_sosdisc_inputs(['manufacturing_cost', 'material_specs'])
        maintenance_cost = self.compute_maintenance_cost(manufacturing_cost, material_specs)
        maintenance_out = {'maintenance_cost': maintenance_cost}
        self.store_sos_outputs_values(maintenance_out)

    def compute_maintenance_cost(self, manufacturing_cost, material_specs):
        """
        Compute maintenance cost based on manufacturing cost and material durability.

        :param manufacturing_cost: manufacturing cost from discipline 1
        :type manufacturing_cost: dataframe
        :param material_specs: material quality and durability specifications
        :type material_specs: numpy.array
        :returns: maintenance cost per year
        :rtype: dataframe
        """
        out = pd.DataFrame({'years': manufacturing_cost['years'], 'value': 0.0})
        manufacturing_cost['years'] = manufacturing_cost['years'].astype('int64')

        # Maintenance cost = base maintenance + quality factor + manufacturing cost influence
        # Higher manufacturing cost leads to lower maintenance (better quality)
        out['value'] = (int(material_specs[0]) * 300 + float(material_specs[1]) * 200 + 
                       manufacturing_cost['value'] * 0.05)

        return out


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
