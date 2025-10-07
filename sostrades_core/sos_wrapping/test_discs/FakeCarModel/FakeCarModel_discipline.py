'''
Copyright 2025 TCh Project

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


import math

import numpy as np

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

# SoSTrades chart framework
CHARTS_AVAILABLE = True


class FakeCarModelDiscipline(SoSWrapp):
    """
    Fake Car Model discipline for generating performance and cost metrics
    based on various car configuration parameters (discrete and continuous).

    This discipline simulates a multi-criteria evaluation of different car
    configurations and provides aggregated performance and cost scores.

    Architecture: This discipline is DoE-independent and can be used in any
    process context (standalone or with DoE). DoE-specific data collection
    and formatting is handled at the process level for maximum reusability.

    For comprehensive documentation, see:
    documentation/FakeCarModel_discipline.md
    """

    # ontology information
    _ontology_data = {
        'label': 'Fake Car Model Discipline',
        'type': 'Research',
        'source': 'SoSTrades Value Analysis Project',
        'version': '1.0',
    }

    DESC_IN = {
        # Discrete inputs
        'engine_type': {'type': 'string', 'default': 'ICE', 'possible_values': ['ICE', 'Hybrid', 'Electric'],
                       'description': 'Type of engine: Internal Combustion Engine, Hybrid, or Electric'},
        'brake_type': {'type': 'string', 'default': 'Disc', 'possible_values': ['Disc', 'Drum', 'Regenerative'],
                      'description': 'Type of brake system'},
        'tire_type': {'type': 'string', 'default': 'All-Season', 'possible_values': ['Economy', 'Performance', 'All-Season'],
                     'description': 'Type of tires'},
        'transmission_type': {'type': 'string', 'default': 'Automatic', 'possible_values': ['Manual', 'Automatic', 'CVT'],
                             'description': 'Type of transmission'},

        # Continuous inputs - flexible namespace (no forced DoE dependency)
        'battery_capacity': {'type': 'float', 'default': 50.0, 'unit': 'kWh',
                            'description': 'Battery capacity in kWh (0-100)'},
        'fuel_tank_capacity': {'type': 'float', 'default': 50.0, 'unit': 'L',
                              'description': 'Fuel tank capacity in liters (0-80)',
                              },
        'vehicle_weight': {'type': 'float', 'default': 1500.0, 'unit': 'kg',
                          'description': 'Vehicle weight in kg (1000-3000)',
                          },
        'aerodynamic_coefficient': {'type': 'float', 'default': 0.3, 'unit': '',
                                   'description': 'Aerodynamic drag coefficient (0.2-0.5)',
                                   },
        'engine_power': {'type': 'float', 'default': 150.0, 'unit': 'kW',
                        'description': 'Engine power in kW (80-300)',
                        },
        'wheel_diameter': {'type': 'float', 'default': 17.0, 'unit': 'inches',
                          'description': 'Wheel diameter in inches (15-21)',
                          },

        # Ponderation weights for aggregation
        'performance_weights': {'type': 'dict', 'default': {
            'top_speed': 0.25,
            'acceleration': 0.25,
            'range': 0.25,
            'efficiency': 0.25
        }, 'description': 'Weights for performance criteria aggregation'},
        'cost_weights': {'type': 'dict', 'default': {
            'manufacturing_cost': 0.4,
            'maintenance_cost': 0.3,
            'environmental_impact': 0.3
        }, 'description': 'Weights for cost criteria aggregation'},
    }

    DESC_OUT = {
        # Individual performance outputs
        'top_speed': {'type': 'float', 'unit': 'km/h',
                     'description': 'Maximum top speed', },
        'acceleration_0_100': {'type': 'float', 'unit': 's',
                              'description': 'Acceleration time from 0 to 100 km/h', },
        'range': {'type': 'float', 'unit': 'km',
                 'description': 'Driving range', },
        'efficiency': {'type': 'float', 'unit': 'L/100km or kWh/100km',
                      'description': 'Fuel/Energy efficiency', },

        # Individual cost outputs
        'manufacturing_cost': {'type': 'float', 'unit': '€',
                              'description': 'Manufacturing cost', },
        'maintenance_cost': {'type': 'float', 'unit': '€/year',
                            'description': 'Annual maintenance cost', },
        'environmental_impact': {'type': 'float', 'unit': 'score',
                                'description': 'Environmental impact score (lower is better)', },
        'safety_rating': {'type': 'float', 'unit': 'score',
                         'description': 'Safety rating (1-5 stars)', },

        # Aggregated outputs
        'aggregated_performance': {'type': 'float', 'unit': 'score',
                                  'description': 'Weighted performance score (0-100)', },
        'aggregated_cost': {'type': 'float', 'unit': 'score',
                           'description': 'Weighted cost score (0-100, lower is better)', },

        # Note: DoE-specific outputs (all_outputs_df, doe_results) are no longer part of the discipline
        # to maintain architectural cleanliness. DoE processes can collect and format data as needed.
    }

    def setup_sos_disciplines(self):
        """Setup the discipline"""
        if not CHARTS_AVAILABLE:
            self.logger.warning("Charts are not available.")

    def run(self):
        """Execute the car model evaluation"""
        print("TCh version marker : 007")

        # Get inputs
        engine_type = self.get_sosdisc_inputs('engine_type')
        brake_type = self.get_sosdisc_inputs('brake_type')
        tire_type = self.get_sosdisc_inputs('tire_type')
        transmission_type = self.get_sosdisc_inputs('transmission_type')

        battery_capacity = self.get_sosdisc_inputs('battery_capacity')
        fuel_tank_capacity = self.get_sosdisc_inputs('fuel_tank_capacity')
        vehicle_weight = self.get_sosdisc_inputs('vehicle_weight')
        aerodynamic_coefficient = self.get_sosdisc_inputs('aerodynamic_coefficient')
        engine_power = self.get_sosdisc_inputs('engine_power')
        wheel_diameter = self.get_sosdisc_inputs('wheel_diameter')

        performance_weights = self.get_sosdisc_inputs('performance_weights')
        cost_weights = self.get_sosdisc_inputs('cost_weights')

        # Calculate individual performance metrics
        top_speed = self._calculate_top_speed(engine_type, engine_power, vehicle_weight, aerodynamic_coefficient)
        acceleration = self._calculate_acceleration(engine_type, engine_power, vehicle_weight, transmission_type)
        range_km = self._calculate_range(engine_type, battery_capacity, fuel_tank_capacity,
                                        vehicle_weight, aerodynamic_coefficient)
        efficiency = self._calculate_efficiency(engine_type, vehicle_weight, aerodynamic_coefficient)

        # Calculate individual cost metrics
        manufacturing_cost = self._calculate_manufacturing_cost(engine_type, brake_type, engine_power,
                                                              battery_capacity, transmission_type, wheel_diameter)
        maintenance_cost = self._calculate_maintenance_cost(engine_type, brake_type, tire_type, engine_power)
        environmental_impact = self._calculate_environmental_impact(engine_type, vehicle_weight)
        safety_rating = self._calculate_safety_rating(brake_type, vehicle_weight)

        # Calculate aggregated scores
        performance_scores = {
            'top_speed': self._normalize_score(top_speed, 120, 300, maximize=True),
            'acceleration': self._normalize_score(acceleration, 3, 15, maximize=False),  # Lower is better
            'range': self._normalize_score(range_km, 200, 800, maximize=True),
            'efficiency': self._normalize_score(efficiency, 3, 15, maximize=False)  # Lower is better
        }

        cost_scores = {
            'manufacturing_cost': self._normalize_score(manufacturing_cost, 15000, 80000, maximize=False),
            'maintenance_cost': self._normalize_score(maintenance_cost, 500, 3000, maximize=False),
            'environmental_impact': self._normalize_score(environmental_impact, 20, 100, maximize=False)
        }

        # Aggregate scores using weights
        aggregated_performance = sum(performance_scores[key] * performance_weights[key]
                                   for key in performance_weights.keys())
        aggregated_cost = sum(cost_scores[key] * cost_weights[key]
                            for key in cost_weights.keys())

        # Store outputs
        self.store_sos_outputs_values({
            'top_speed': top_speed,
            'acceleration_0_100': acceleration,
            'range': range_km,
            'efficiency': efficiency,
            'manufacturing_cost': manufacturing_cost,
            'maintenance_cost': maintenance_cost,
            'environmental_impact': environmental_impact,
            'safety_rating': safety_rating,
            'aggregated_performance': aggregated_performance,
            'aggregated_cost': aggregated_cost
        })

    def _calculate_top_speed(self, engine_type: str, engine_power: float, weight: float, aero_coeff: float) -> float:
        """Calculate top speed based on engine power, weight, and aerodynamics"""
        # Power-to-weight ratio is key for top speed
        power_to_weight = engine_power / weight  # kW/kg

        # Base calculation using power-to-weight ratio
        base_speed = 120 + (power_to_weight - 0.1) * 600  # Realistic scaling

        # Engine type modifier
        type_modifiers = {'ICE': 1.0, 'Hybrid': 0.95, 'Electric': 1.05}
        base_speed *= type_modifiers[engine_type]

        # Aerodynamic factor (lower coefficient = higher top speed)
        aero_factor = (0.35 - aero_coeff) / 0.25 * 20  # Up to 20 km/h difference

        final_speed = base_speed + aero_factor + np.random.normal(0, 5)
        return max(120, min(300, final_speed))

    def _calculate_acceleration(self, engine_type: str, engine_power: float, weight: float, transmission: str) -> float:
        """Calculate 0-100 km/h acceleration time with realistic power-to-weight physics"""
        # Power-to-weight ratio drives acceleration
        power_to_weight = engine_power / weight  # kW/kg

        # Base acceleration time (lower is better)
        base_time = 12 - (power_to_weight - 0.1) * 30  # Realistic scaling

        # Engine type advantages
        type_modifiers = {'ICE': 1.0, 'Hybrid': 0.9, 'Electric': 0.8}  # Electric has instant torque
        base_time *= type_modifiers[engine_type]

        # Transmission effects
        trans_modifiers = {'Manual': 1.1, 'Automatic': 1.0, 'CVT': 1.05}
        base_time *= trans_modifiers[transmission]

        final_time = base_time + np.random.normal(0, 0.3)
        return max(3, min(15, final_time))

    def _calculate_range(self, engine_type: str, battery_cap: float, fuel_cap: float,
                        weight: float, aero_coeff: float) -> float:
        """Calculate driving range"""
        if engine_type == 'Electric':
            base_efficiency = 0.15  # kWh/km
            efficiency = base_efficiency * (1 + (weight - 1500) / 1500 * 0.3) * (aero_coeff / 0.3)
            return battery_cap / efficiency
        elif engine_type == 'Hybrid':
            base_efficiency = 0.04  # L/km
            efficiency = base_efficiency * (1 + (weight - 1500) / 1500 * 0.2) * (aero_coeff / 0.3)
            electric_range = battery_cap / 0.15
            fuel_range = fuel_cap / efficiency
            return electric_range + fuel_range
        else:  # ICE
            base_efficiency = 0.07  # L/km
            efficiency = base_efficiency * (1 + (weight - 1500) / 1500 * 0.3) * (aero_coeff / 0.3)
            return fuel_cap / efficiency + np.random.normal(0, 20)

    def _calculate_efficiency(self, engine_type: str, weight: float, aero_coeff: float) -> float:
        """Calculate fuel/energy efficiency"""
        if engine_type == 'Electric':
            base_eff = 15.0  # kWh/100km
        elif engine_type == 'Hybrid':
            base_eff = 4.0   # L/100km
        else:  # ICE
            base_eff = 7.0   # L/100km

        weight_penalty = (weight - 1500) / 1500 * 2.0
        aero_penalty = (aero_coeff - 0.3) / 0.2 * 1.5

        return base_eff + weight_penalty + aero_penalty + np.random.normal(0, 0.3)

    def _calculate_manufacturing_cost(self, engine_type: str, brake_type: str, engine_power: float,
                                    battery_cap: float, transmission: str, wheel_diameter: float) -> float:
        """Calculate manufacturing cost with enhanced component pricing"""
        base_costs = {'ICE': 25000, 'Hybrid': 35000, 'Electric': 30000}
        brake_costs = {'Disc': 2000, 'Drum': 1000, 'Regenerative': 3000}
        trans_costs = {'Manual': 2000, 'Automatic': 4000, 'CVT': 3000}

        # Base cost
        cost = base_costs[engine_type] + brake_costs[brake_type] + trans_costs[transmission]

        # Engine power cost (€50 per kW above 100kW baseline)
        if engine_power > 100:
            cost += (engine_power - 100) * 50

        # Battery cost for electric/hybrid
        if engine_type in ['Electric', 'Hybrid']:
            cost += battery_cap * 200  # €200 per kWh

        # Wheel cost (larger wheels cost more)
        wheel_cost = (wheel_diameter - 16) * 300  # €300 per inch above 16"
        cost += max(0, wheel_cost)

        return cost + np.random.normal(0, 2000)

    def _calculate_maintenance_cost(self, engine_type: str, brake_type: str, tire_type: str, engine_power: float) -> float:
        """Calculate annual maintenance cost with power dependency"""
        base_costs = {'ICE': 1200, 'Hybrid': 900, 'Electric': 600}
        brake_costs = {'Disc': 300, 'Drum': 200, 'Regenerative': 150}
        tire_costs = {'Economy': 200, 'Performance': 400, 'All-Season': 300}

        # Higher power engines cost more to maintain
        power_cost = (engine_power - 150) * 2  # €2 per kW above 150kW baseline

        total_cost = (base_costs[engine_type] + brake_costs[brake_type] +
                     tire_costs[tire_type] + max(0, power_cost))

        return total_cost + np.random.normal(0, 100)

    def _calculate_environmental_impact(self, engine_type: str, weight: float) -> float:
        """Calculate environmental impact score (lower is better)"""
        base_scores = {'ICE': 80, 'Hybrid': 45, 'Electric': 25}
        weight_penalty = (weight - 1500) / 1500 * 15

        return base_scores[engine_type] + weight_penalty + np.random.normal(0, 3)

    def _calculate_safety_rating(self, brake_type: str, weight: float) -> float:
        """Calculate safety rating (1-5 stars)"""
        brake_ratings = {'Disc': 4.5, 'Drum': 3.5, 'Regenerative': 4.8}

        # Heavier cars tend to be safer in crashes (up to a point)
        weight_bonus = min((weight - 1200) / 800 * 0.5, 0.5)

        rating = brake_ratings[brake_type] + weight_bonus + np.random.normal(0, 0.1)
        return min(max(rating, 1.0), 5.0)

    def _normalize_score(self, value: float, min_val: float, max_val: float, maximize: bool = True) -> float:
        """Normalize a value to 0-100 scale"""
        if maximize:
            return max(0, min(100, (value - min_val) / (max_val - min_val) * 100))
        else:
            return max(0, min(100, (max_val - value) / (max_val - min_val) * 100))

    def get_chart_filter_list(self):
        """Return list of chart filters"""
        print("DEBUG: get_chart_filter_list called")
        chart_filters = []

        if CHARTS_AVAILABLE:
            chart_list = ['Performance Profile', 'Star Diagram', 'Cost vs Performance']
            chart_filters.append(ChartFilter('charts', chart_list, chart_list, 'charts'))

        print(f"DEBUG: Returning {len(chart_filters)} chart filters")
        for i, cf in enumerate(chart_filters):
            print(f"DEBUG: Filter {i}: filter_key='{cf.filter_key}', selected_values={cf.selected_values}")

        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        """Generate post-processing charts"""
        print(f"DEBUG: get_post_processing_list called with chart_filters: {chart_filters}")
        print(f"DEBUG: CHARTS_AVAILABLE = {CHARTS_AVAILABLE}")

        charts = []

        # Default charts if no filter (following LogisticsNetworkOptimizer pattern)
        if chart_filters is None:
            chart_list = ['Performance Profile', 'Star Diagram', 'Cost vs Performance']
        else:
            chart_list = []
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        print(f"DEBUG: Chart list to process: {chart_list}")

        # Get current outputs for single point visualization
        try:
            top_speed = self.get_sosdisc_outputs('top_speed')
            acceleration = self.get_sosdisc_outputs('acceleration_0_100')
            range_km = self.get_sosdisc_outputs('range')
            efficiency = self.get_sosdisc_outputs('efficiency')
            manufacturing_cost = self.get_sosdisc_outputs('manufacturing_cost')
            maintenance_cost = self.get_sosdisc_outputs('maintenance_cost')
            environmental_impact = self.get_sosdisc_outputs('environmental_impact')
            safety_rating = self.get_sosdisc_outputs('safety_rating')
            agg_performance = self.get_sosdisc_outputs('aggregated_performance')
            agg_cost = self.get_sosdisc_outputs('aggregated_cost')
            print(f"DEBUG: Successfully retrieved outputs - agg_cost: {agg_cost}, agg_performance: {agg_performance}")
        except Exception as e:
            print(f"DEBUG: Error retrieving outputs: {e}")
            # If outputs are not available (discipline hasn't run), return empty list
            return charts

        # Performance Profile Chart
        if 'Performance Profile' in chart_list:
            print("DEBUG: Creating Performance Profile chart...")
            chart = self._create_performance_profile_chart(
                top_speed, acceleration, range_km, efficiency,
                manufacturing_cost, maintenance_cost, environmental_impact, safety_rating
            )
            if chart is not None:
                print(f"DEBUG: Performance Profile chart created successfully: {type(chart)}")
                charts.append(chart)
            else:
                print("DEBUG: Performance Profile chart creation returned None")

        # Star Diagram Chart
        if 'Star Diagram' in chart_list:
            print("DEBUG: Creating Star Diagram chart...")
            chart = self._create_star_diagram(
                top_speed, acceleration, range_km, efficiency,
                manufacturing_cost, maintenance_cost, environmental_impact, safety_rating
            )
            if chart is not None:
                print(f"DEBUG: Star Diagram chart created successfully: {type(chart)}")
                charts.append(chart)
            else:
                print("DEBUG: Star Diagram chart creation returned None")

        # Cost vs Performance Chart
        if 'Cost vs Performance' in chart_list:
            print("DEBUG: Creating Cost vs Performance chart...")
            chart = self._create_cost_performance_chart(agg_cost, agg_performance)
            if chart is not None:
                print(f"DEBUG: Cost vs Performance chart created successfully: {type(chart)}")
                charts.append(chart)
            else:
                print("DEBUG: Cost vs Performance chart creation returned None")

        print(f"DEBUG: Returning {len(charts)} charts")
        for i, chart in enumerate(charts):
            print(f"DEBUG: Chart {i}: '{chart.chart_name}' ({type(chart).__name__})")
            if hasattr(chart, 'series') and chart.series:
                for j, series in enumerate(chart.series):
                    print(f"DEBUG:   Series {j}: '{series.series_name}' type={series.display_type}")
                    print(f"DEBUG:     Data length: x={len(series.abscissa) if series.abscissa else 0}, y={len(series.ordinate) if series.ordinate else 0}")
        return charts

    def _create_performance_profile_chart(self, top_speed, acceleration, range_km, efficiency,
                                         manufacturing_cost, maintenance_cost, environmental_impact, safety_rating):
        """Create performance profile bar chart with all metrics"""
        print("DEBUG: Creating Performance Profile bar chart for FakeCarModelDiscipline")
        print(f"DEBUG: Input values - top_speed: {top_speed}, acceleration: {acceleration}, range: {range_km}")

        # Normalize all values to 0-100 scale for visualization
        metrics = {
            'Top Speed': self._normalize_score(top_speed, 120, 300, True),
            'Acceleration': self._normalize_score(acceleration, 3, 15, False),
            'Range': self._normalize_score(range_km, 200, 800, True),
            'Efficiency': self._normalize_score(efficiency, 3, 15, False),
            'Manufacturing Cost': self._normalize_score(manufacturing_cost, 15000, 80000, False),
            'Maintenance Cost': self._normalize_score(maintenance_cost, 500, 3000, False),
            'Environmental Impact': self._normalize_score(environmental_impact, 20, 100, False),
            'Safety Rating': self._normalize_score(safety_rating, 1, 5, True)
        }

        print(f"DEBUG: Normalized metrics: {metrics}")

        # Create standard SoSTrades bar chart
        chart = TwoAxesInstanciatedChart(
            'Metrics', 'Normalized Score (0-100)',
            chart_name='Performance Profile'
        )

        # Add metrics as a bar series using standard SoSTrades approach
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())

        print(f"DEBUG: Chart data - names: {metrics_names}")
        print(f"DEBUG: Chart data - values: {metrics_values}")

        # For bar charts, names go on x-axis (abscissa) and values on y-axis (ordinate)
        series = InstanciatedSeries(
            metrics_names, metrics_values,
            'Current Configuration', 'bar'
        )
        chart.series.append(series)

        print(f"DEBUG: Chart created with {len(chart.series)} series")
        print(f"DEBUG: Chart type: {type(chart)}")
        print(f"DEBUG: Chart name: {chart.chart_name}")
        print(f"DEBUG: Chart series types: {[type(s) for s in chart.series]}")
        return chart

    def _create_star_diagram(self, top_speed, acceleration, range_km, efficiency,
                           manufacturing_cost, maintenance_cost, environmental_impact, safety_rating):
        """Create star/radar diagram with normalized metrics using Cartesian coordinates"""
        print("DEBUG: Creating Star Diagram for FakeCarModelDiscipline")
        print(f"DEBUG: Input values - top_speed: {top_speed}, acceleration: {acceleration}, range: {range_km}")

        # Normalize all values to 0-100 scale for visualization
        metrics = {
            'Top Speed': self._normalize_score(top_speed, 120, 300, True),
            'Acceleration': self._normalize_score(acceleration, 3, 15, False),
            'Range': self._normalize_score(range_km, 200, 800, True),
            'Efficiency': self._normalize_score(efficiency, 3, 15, False),
            'Manufacturing Cost': self._normalize_score(manufacturing_cost, 15000, 80000, False),
            'Maintenance Cost': self._normalize_score(maintenance_cost, 500, 3000, False),
            'Environmental Impact': self._normalize_score(environmental_impact, 20, 100, False),
            'Safety Rating': self._normalize_score(safety_rating, 1, 5, True)
        }

        print(f"DEBUG: Normalized metrics: {metrics}")

        # Create Cartesian chart for star display
        chart = TwoAxesInstanciatedChart(
            'X Coordinate', 'Y Coordinate',
            chart_name='Star Diagram'
        )

        # Create star diagram using polar to Cartesian conversion
        n_metrics = len(metrics)
        angle_step = 2 * math.pi / n_metrics  # Convert to radians

        # Convert polar coordinates to Cartesian for proper star shape
        x_coords = []
        y_coords = []

        metrics_items = list(metrics.items())

        # Add each metric point around the circle
        for i, (name, value) in enumerate(metrics_items):
            angle_rad = i * angle_step
            # Convert polar (angle, radius) to Cartesian (x, y)
            # Use value as radius, scale to reasonable display size
            radius = value  # Already normalized to 0-100
            x = radius * math.cos(angle_rad)
            y = radius * math.sin(angle_rad)
            x_coords.append(x)
            y_coords.append(y)
            print(f"DEBUG: Metric '{name}': value={value:.1f}, angle={math.degrees(angle_rad):.1f}°, x={x:.1f}, y={y:.1f}")

        # Close the polygon by adding the first point at the end
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

        print(f"DEBUG: Star diagram x_coords: {x_coords}")
        print(f"DEBUG: Star diagram y_coords: {y_coords}")

        # Create line series to form the star shape
        series = InstanciatedSeries(
            x_coords, y_coords,
            'Configuration Star', 'lines'
        )
        chart.series.append(series)

        # Add individual points as scatter (without the closing point)
        point_series = InstanciatedSeries(
            x_coords[:-1], y_coords[:-1],
            'Metric Points', 'scatter'
        )
        chart.series.append(point_series)

        # Add origin point for reference
        origin_series = InstanciatedSeries(
            [0], [0],
            'Origin', 'scatter'
        )
        chart.series.append(origin_series)

        print(f"DEBUG: Star diagram created with {len(chart.series)} series")
        print(f"DEBUG: Chart type: {type(chart)}")
        print(f"DEBUG: Chart name: {chart.chart_name}")
        print(f"DEBUG: Chart series types: {[type(s) for s in chart.series]}")
        return chart

    def _create_cost_performance_chart(self, cost, performance):
        """Create 2D scatter plot with cost vs performance"""
        print("DEBUG: Creating Cost vs Performance chart for FakeCarModelDiscipline")
        print(f"DEBUG: Input values - cost: {cost}, performance: {performance}")

        # Create standard SoSTrades scatter chart
        chart = TwoAxesInstanciatedChart(
            'Aggregated Cost Score (lower is better)',
            'Aggregated Performance Score (higher is better)',
            chart_name='Cost vs Performance'
        )

        # Add point as a scatter series using standard SoSTrades approach
        # For scatter plots: x-axis (abscissa) = cost, y-axis (ordinate) = performance
        series = InstanciatedSeries(
            [cost], [performance],
            'Current Configuration', 'scatter'
        )
        chart.series.append(series)

        print(f"DEBUG: Scatter chart created with {len(chart.series)} series")
        print(f"DEBUG: Series data - x: {series.abscissa}, y: {series.ordinate}")
        print(f"DEBUG: Chart type: {type(chart)}")
        print(f"DEBUG: Chart name: {chart.chart_name}")
        print(f"DEBUG: Chart series types: {[type(s) for s in chart.series]}")
        return chart


# Export the main discipline class for SoSTrades factory
__all__ = ['FakeCarModelDiscipline']
