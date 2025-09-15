"""Energy calculation discipline with post-processing charts using callback approach"""

import numpy as np

from sostrades_core.tools.discipline_decorator.function_discipline_decorator import auto_sos_discipline
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


def create_energy_charts(data):
    """Create charts for thermal energy time series calculation"""
    # Extract energy time series data
    fahrenheit_series = data['fahrenheit_series']
    thermal_energy_series = data['thermal_energy_series']
    total_energy_series = data['total_energy_series']
    mass = data['mass']

    # Generate time_days from the length of the series
    time_days = list(range(len(thermal_energy_series)))

    # Create energy evolution chart
    energy_chart = TwoAxesInstanciatedChart(abscissa_axis_name='Time (days)',
                                            primary_ordinate_axis_name='Energy (J)',
                                            chart_name='Energy Evolution Over Time')

    # Create series for energy evolution
    thermal_series = InstanciatedSeries(time_days, thermal_energy_series, 'Thermal Energy (J)', 'lines')
    total_series = InstanciatedSeries(time_days, total_energy_series, 'Total Energy (J)', 'lines')

    energy_chart.add_series(thermal_series)
    energy_chart.add_series(total_series)

    # Create energy statistics chart
    stats_chart = TwoAxesInstanciatedChart(abscissa_axis_name='Metric',
                                           primary_ordinate_axis_name='Energy (J)',
                                           chart_name='Energy Statistics')

    if thermal_energy_series and total_energy_series:
        avg_thermal = np.mean(thermal_energy_series)
        max_thermal = np.max(thermal_energy_series)
        min_thermal = np.min(thermal_energy_series)

        avg_total = np.mean(total_energy_series)
        max_total = np.max(total_energy_series)
        min_total = np.min(total_energy_series)

        # Statistics series
        thermal_stats_series = InstanciatedSeries(['Min', 'Avg', 'Max'], [min_thermal, avg_thermal, max_thermal], 'Thermal Energy Stats', 'bar')
        total_stats_series = InstanciatedSeries(['Min', 'Avg', 'Max'], [min_total, avg_total, max_total], 'Total Energy Stats', 'bar')

        stats_chart.add_series(thermal_stats_series)
        stats_chart.add_series(total_stats_series)

    # Create energy density chart (energy per kg over time)
    density_chart = TwoAxesInstanciatedChart(abscissa_axis_name='Time (days)',
                                             primary_ordinate_axis_name='Energy Density (J/kg)',
                                             chart_name='Energy Density Over Time')

    if mass > 0:
        thermal_density_series = [e / mass for e in thermal_energy_series]
        total_density_series = [e / mass for e in total_energy_series]

        thermal_density = InstanciatedSeries(time_days, thermal_density_series, 'Thermal Energy Density (J/kg)', 'lines')
        total_density = InstanciatedSeries(time_days, total_density_series, 'Total Energy Density (J/kg)', 'lines')

        density_chart.add_series(thermal_density)
        density_chart.add_series(total_density)

    return [energy_chart, stats_chart, density_chart]


@auto_sos_discipline(
    outputs={'thermal_energy_series': list, 'total_energy_series': list},
    post_processing_callback=create_energy_charts
)
def EnergyCalc(fahrenheit_series: list[float], mass: float) -> dict[str, list[float]]:
    """Calculate thermal energy time series using Fahrenheit temperature series and mass."""
    time_days = list(range(len(fahrenheit_series)))
    thermal_energy_series = []
    total_energy_series = []

    for fahrenheit in fahrenheit_series:
        # Convert Fahrenheit to Celsius for calculation
        celsius = (fahrenheit - 32) * 5/9
        # Simple thermal energy calculation (arbitrary formula for demo)
        thermal_energy = mass * celsius * 4.18  # Specific heat of water
        total_energy = thermal_energy + (mass * 9.81 * 10)  # Add potential energy at 10m height

        thermal_energy_series.append(thermal_energy)
        total_energy_series.append(total_energy)

    return {
        'thermal_energy_series': thermal_energy_series,
        'total_energy_series': total_energy_series
    }
