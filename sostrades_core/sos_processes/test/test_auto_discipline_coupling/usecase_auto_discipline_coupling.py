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

from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    """Study class for testing auto discipline coupling."""

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        """Set up the usecase with initial values."""
        values_dict = {}

        # Initial temperature in Celsius and mass
        values_dict[f'{self.study_name}.celsius'] = 85.0  # Temperature in Celsius
        values_dict[f'{self.study_name}.mass'] = 2.0      # Mass in kg

        return [values_dict]

if __name__ == '__main__':
    study = Study()
    study.load_data()
    study.run()

    # Show results
    print("🎯 Auto Discipline Coupling Demo")
    print("=" * 35)
    celsius = study.execution_engine.dm.get_value(f'{study.study_name}.celsius')
    fahrenheit = study.execution_engine.dm.get_value(f'{study.study_name}.fahrenheit')
    kelvin = study.execution_engine.dm.get_value(f'{study.study_name}.kelvin')
    mass = study.execution_engine.dm.get_value(f'{study.study_name}.mass')
    thermal_energy = study.execution_engine.dm.get_value(f'{study.study_name}.thermal_energy')
    total_energy = study.execution_engine.dm.get_value(f'{study.study_name}.total_energy')

    print(f"Temperature: {celsius}°C = {fahrenheit}°F = {kelvin}K")
    print(f"Mass: {mass} kg")
    print(f"Thermal Energy: {thermal_energy:.2f} J")
    print(f"Total Energy: {total_energy:.2f} J")
    print("\n✅ Auto SoS Discipline decorator demonstration complete!")
