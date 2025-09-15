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
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)


class Study(StudyManager):
    """Study class for testing auto discipline with charts."""

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self, study_folder_path=None):
        """Set up the usecase with initial values."""
        values_dict = {}

        # Initial temperature in Celsius and simulation duration
        values_dict[f'{self.study_name}.celsius_base'] = 85.0  # Base temperature in Celsius
        values_dict[f'{self.study_name}.days_duration'] = 30.0  # Duration in days
        values_dict[f'{self.study_name}.mass'] = 2.0      # Mass in kg

        return [values_dict]


if __name__ == '__main__':
    study = Study()
    study.load_data()
    study.run()

    # Show results
    print("🎯 Auto Discipline With Charts Demo")
    print("=" * 40)
    celsius_base = study.execution_engine.dm.get_value(f'{study.study_name}.celsius_base')
    days_duration = study.execution_engine.dm.get_value(f'{study.study_name}.days_duration')
    mass = study.execution_engine.dm.get_value(f'{study.study_name}.mass')

    fahrenheit_series = study.execution_engine.dm.get_value(f'{study.study_name}.fahrenheit_series')
    thermal_energy_series = study.execution_engine.dm.get_value(f'{study.study_name}.thermal_energy_series')
    time_days = study.execution_engine.dm.get_value(f'{study.study_name}.time_days')

    print(f"Base Temperature: {celsius_base}°C")
    print(f"Simulation Duration: {days_duration} days")
    print(f"Mass: {mass} kg")
    print(f"Generated {len(time_days)} time points with temperature and energy series")

    if fahrenheit_series and thermal_energy_series:
        print(f"Temperature range: {min(fahrenheit_series):.1f}°F to {max(fahrenheit_series):.1f}°F")
        print(f"Thermal energy range: {min(thermal_energy_series):.1f}J to {max(thermal_energy_series):.1f}J")

    # Generate and display post-processing charts
    ppf = PostProcessingFactory()
    all_post_processings = ppf.get_all_post_processings(study.execution_engine, False, as_json=False, for_test=False)

    chart_count = sum(len(post_proc_list) for post_proc_list in all_post_processings.values())
    print(f"\n📊 Generated {chart_count} charts with post-processing!")
    print("✅ Auto SoS Discipline with charts demonstration complete!")

    # Optionally show charts (uncomment to display)
    for post_proc_list in all_post_processings.values():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly().show()
