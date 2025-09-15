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

from sostrades_core.tools.discipline_decorator.function_discipline_decorator import auto_sos_discipline


@auto_sos_discipline(outputs={'thermal_energy': float, 'total_energy': float})
def EnergyCalc(fahrenheit: float = 77.0, mass: float = 1.0) -> dict:
    """Calculate thermal energy using Fahrenheit temperature and mass."""
    # Convert Fahrenheit to Celsius for calculation
    celsius = (fahrenheit - 32) * 5/9
    # Simple thermal energy calculation (arbitrary formula for demo)
    thermal_energy = mass * celsius * 4.18  # Specific heat of water
    total_energy = thermal_energy + (mass * 9.81 * 10)  # Add potential energy at 10m height
    return {'thermal_energy': thermal_energy, 'total_energy': total_energy}
