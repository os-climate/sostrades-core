# FakeCarModel Discipline Documentation

## 1. Overview

The **FakeCarModel** discipline is a comprehensive automotive performance and cost evaluation tool designed for multi-criteria decision analysis within the SoSTrades platform. It simulates the performance characteristics and cost implications of different car configurations based on both discrete and continuous design parameters.

### Purpose

The discipline provides a realistic simulation of automotive engineering trade-offs, enabling users to:
- Evaluate the impact of different engine technologies (ICE, Hybrid, Electric)
- Analyze performance metrics (top speed, acceleration, range, efficiency)
- Assess cost implications (manufacturing, maintenance, environmental impact)
- Perform multi-criteria optimization studies using aggregated performance and cost scores

### Architecture

This discipline follows a clean architecture pattern:
- **DoE Independence**: Can be used standalone or within Design of Experiments workflows
- **Modular Design**: Clear separation between input processing, calculation methods, and output generation
- **Extensible**: Easy to add new car components or evaluation criteria
- **Visualization Ready**: Includes post-processing charts for result analysis

## 2. Technical Specifications

### 2.1 Input Parameters

#### Discrete Configuration Parameters
| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `engine_type` | string | "ICE" | ["ICE", "Hybrid", "Electric"] | Type of engine/powertrain |
| `brake_type` | string | "Disc" | ["Disc", "Drum", "Regenerative"] | Brake system type |
| `tire_type` | string | "All-Season" | ["Economy", "Performance", "All-Season"] | Tire specification |
| `transmission_type` | string | "Automatic" | ["Manual", "Automatic", "CVT"] | Transmission system |

#### Continuous Design Variables
| Parameter | Type | Unit | Default | Range | Description |
|-----------|------|------|---------|-------|-------------|
| `battery_capacity` | float | kWh | 50.0 | 0-100 | Battery capacity for electric/hybrid |
| `fuel_tank_capacity` | float | L | 50.0 | 0-80 | Fuel tank capacity |
| `vehicle_weight` | float | kg | 1500.0 | 1000-3000 | Total vehicle weight |
| `aerodynamic_coefficient` | float | - | 0.3 | 0.2-0.5 | Drag coefficient (Cd) |
| `engine_power` | float | kW | 150.0 | 80-300 | Maximum engine power |
| `wheel_diameter` | float | inches | 17.0 | 15-21 | Wheel diameter |

#### Weighting Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `performance_weights` | dict | Weights for aggregating performance criteria |
| `cost_weights` | dict | Weights for aggregating cost criteria |

### 2.2 Output Parameters

#### Individual Performance Metrics
| Output | Type | Unit | Description |
|--------|------|------|-------------|
| `top_speed` | float | km/h | Maximum achievable speed (120-300 km/h) |
| `acceleration_0_100` | float | s | 0-100 km/h acceleration time (3-15 s) |
| `range` | float | km | Driving range on full tank/battery |
| `efficiency` | float | L/100km or kWh/100km | Fuel/energy consumption |

#### Individual Cost Metrics
| Output | Type | Unit | Description |
|--------|------|------|-------------|
| `manufacturing_cost` | float | € | Total manufacturing cost (15,000-80,000€) |
| `maintenance_cost` | float | €/year | Annual maintenance cost (500-3,000€) |
| `environmental_impact` | float | score | Environmental impact score (20-100, lower is better) |
| `safety_rating` | float | score | Safety rating (1-5 stars) |

#### Aggregated Scores
| Output | Type | Unit | Description |
|--------|------|------|-------------|
| `aggregated_performance` | float | score | Weighted performance score (0-100, higher is better) |
| `aggregated_cost` | float | score | Weighted cost score (0-100, lower is better) |

## 3. Calculation Methods

### 3.1 Performance Calculations

#### Top Speed Calculation
```
power_to_weight = engine_power / vehicle_weight  # kW/kg
base_speed = 120 + (power_to_weight - 0.1) * 600
engine_modifier = {"ICE": 1.0, "Hybrid": 0.95, "Electric": 1.05}
aero_factor = (0.35 - aerodynamic_coefficient) / 0.25 * 20
top_speed = base_speed * engine_modifier + aero_factor + noise
```

#### Acceleration Calculation
```
power_to_weight = engine_power / vehicle_weight
base_time = 12 - (power_to_weight - 0.1) * 30
engine_modifier = {"ICE": 1.0, "Hybrid": 0.9, "Electric": 0.8}
transmission_modifier = {"Manual": 1.1, "Automatic": 1.0, "CVT": 1.05}
acceleration = base_time * engine_modifier * transmission_modifier + noise
```

#### Range Calculation
- **Electric**: `range = battery_capacity / (base_efficiency * weight_penalty * aero_penalty)`
- **Hybrid**: `range = electric_range + fuel_range`
- **ICE**: `range = fuel_tank_capacity / (base_efficiency * penalties)`

### 3.2 Cost Calculations

#### Manufacturing Cost
```
base_cost = engine_base_cost + brake_cost + transmission_cost
power_cost = max(0, (engine_power - 100) * 50)  # €50/kW above 100kW
battery_cost = battery_capacity * 200  # €200/kWh for electric/hybrid
wheel_cost = max(0, (wheel_diameter - 16) * 300)  # €300/inch above 16"
manufacturing_cost = base_cost + power_cost + battery_cost + wheel_cost + noise
```

#### Maintenance Cost
```
base_maintenance = engine_base_maintenance + brake_maintenance + tire_maintenance
power_penalty = max(0, (engine_power - 150) * 2)  # €2/kW above 150kW
maintenance_cost = base_maintenance + power_penalty + noise
```

### 3.3 Aggregation Method

Both performance and cost scores use weighted linear aggregation:
```
aggregated_score = Σ(normalized_criterion_i * weight_i)
```
Where each criterion is normalized to a 0-100 scale using min-max normalization.

## 4. Usage Examples

### 4.1 Standalone Usage

```python
from supply_logistics.models.FakeCarModel.FakeCarModel_discipline import FakeCarModelDiscipline

# Create discipline instance
discipline = FakeCarModelDiscipline()

# Configure inputs
inputs = {
    'engine_type': 'Electric',
    'vehicle_weight': 1800.0,
    'battery_capacity': 75.0,
    'engine_power': 200.0,
    'aerodynamic_coefficient': 0.25
}

# Execute evaluation
discipline.load_data(inputs)
discipline.run()

# Get results
performance_score = discipline.get_sosdisc_outputs('aggregated_performance')
cost_score = discipline.get_sosdisc_outputs('aggregated_cost')
```

### 4.2 Process Integration

The discipline can be integrated into SoSTrades processes:

```python
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder

class CarOptimizationProcess(BaseProcessBuilder):
    def get_builders(self):
        ns_dict = {'ns_public': 'CarOptim'}
        mods_dict = {
            'FakeCarModel': 'supply_logistics.models.FakeCarModel.FakeCarModel_discipline.FakeCarModelDiscipline'
        }
        return self.create_builder_list(mods_dict, ns_dict=ns_dict)
```

### 4.3 Design of Experiments (DoE)

For DoE studies, the continuous parameters can be used as design variables:

```python
# DoE variables (typically 5-6 continuous parameters)
doe_variables = [
    'battery_capacity',     # 0-100 kWh
    'vehicle_weight',       # 1000-3000 kg  
    'aerodynamic_coefficient',  # 0.2-0.5
    'engine_power',         # 80-300 kW
    'wheel_diameter'        # 15-21 inches
]
```

## 5. Visualization and Post-Processing

### 5.1 Available Charts

The discipline provides two main visualization types:

#### Performance Profile Bar Chart
- Displays all performance and cost metrics on a normalized 0-100 scale
- Bar chart format for easy comparison of different criteria
- Shows strengths and weaknesses at a glance

#### Cost vs Performance Scatter Plot
- 2D scatter plot with aggregated cost (x-axis) vs aggregated performance (y-axis)
- Helps identify trade-offs between cost and performance
- Useful for decision analysis

### 5.2 Chart Access

Charts are available through the SoSTrades post-processing interface:

```python
# Get chart filters
chart_filters = discipline.get_chart_filter_list()

# Generate charts
charts = discipline.get_post_processing_list(chart_filters)
```

## 6. Validation and Testing

### 6.1 Test Process
A dedicated test process is available at:
```
platform/sostrades-core/sostrades_core/sos_processes/test/testFakeCarModel/
```

### 6.2 Expected Ranges

The discipline has been calibrated to produce realistic automotive values:
- Top speeds: 120-300 km/h
- Acceleration: 3-15 seconds (0-100 km/h)
- Manufacturing costs: 15,000-80,000 €
- Maintenance costs: 500-3,000 €/year
- Safety ratings: 1-5 stars

### 6.3 Validation Checks

The discipline includes built-in validation:
- Parameter bounds enforcement
- Physical constraint checking
- Noise addition for realistic variability
- Cross-validation between related parameters

## 7. Extension Guidelines

### 7.1 Adding New Components

To add new discrete components (e.g., suspension type):

1. Add to `DESC_IN` with possible values
2. Update relevant calculation methods
3. Add component costs/impacts
4. Update documentation

### 7.2 Adding New Metrics

To add new performance/cost metrics:

1. Add to `DESC_OUT` specification
2. Implement calculation method
3. Add to aggregation weights
4. Update normalization ranges
5. Include in visualizations

### 7.3 Modifying Calculation Models

The discipline uses modular calculation methods that can be easily modified:
- Each metric has its own `_calculate_*()` method
- Physics-based formulas are documented
- Noise parameters can be adjusted
- Component interactions are clearly defined

## 8. Dependencies

### Required Packages
- `numpy`: Numerical calculations and random noise
- `sostrades-core`: SoSTrades framework integration

### Optional Packages
- Charts are implemented using standard SoSTrades TwoAxesInstanciatedChart (no additional dependencies required)

### SoSTrades Version Compatibility
- Compatible with SoSTrades core framework
- Uses standard SoSWrapp base class
- Follows SoSTrades naming conventions
- Supports SHARED_VISIBILITY for DoE integration

## 9. Troubleshooting

### Common Issues

**Issue**: Parameters not visible in GUI
**Solution**: Ensure parameters have `'visibility': SoSWrapp.SHARED_VISIBILITY`

**Issue**: Charts not displaying
**Solution**: Ensure the discipline has been executed (run) before accessing post-processing charts

**Issue**: Unrealistic results
**Solution**: Check parameter ranges and calculation method calibration

**Issue**: DoE integration problems
**Solution**: Verify the discipline is namespace-independent (no forced DoE coupling)

### Performance Considerations

- The discipline is computationally lightweight
- Suitable for large DoE studies (1000+ evaluations)
- Random noise ensures statistical robustness
- Vectorization possible for batch evaluations

## 10. Version History

- **v1.0** (September 2025): Initial implementation
  - Complete automotive simulation model
  - DoE-independent architecture
  - Plotly visualization support
  - Comprehensive test coverage

---

**Authors**: TCh Project Team  
**License**: Apache License 2.0  
**Contact**: For questions or contributions, please refer to the project documentation.
