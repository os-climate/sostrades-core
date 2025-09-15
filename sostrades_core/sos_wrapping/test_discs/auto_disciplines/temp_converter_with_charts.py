"""Temperature converter discipline with post-processing charts using callback approach."""

import numpy as np

from sostrades_core.tools.discipline_decorator.function_discipline_decorator import auto_sos_discipline
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


def create_temperature_charts(data):
    """Create charts for temperature time series"""
    # Extract temperature time series data
    time_days = data['time_days']
    fahrenheit_series_data = data['fahrenheit_series']
    kelvin_series_data = data['kelvin_series']
    celsius_base = data['celsius_base']

    # Create temperature evolution chart
    temp_chart = TwoAxesInstanciatedChart(abscissa_axis_name='Time (days)',
                                          primary_ordinate_axis_name='Temperature',
                                          chart_name='Temperature Evolution Over Time')

    # Create series for temperature evolution
    fahrenheit_series = InstanciatedSeries(time_days, fahrenheit_series_data, 'Temperature (°F)', 'lines')
    kelvin_series = InstanciatedSeries(time_days, kelvin_series_data, 'Temperature (K)', 'lines')

    temp_chart.add_series(fahrenheit_series)
    temp_chart.add_series(kelvin_series)

    # Create temperature comparison chart (current vs average)
    comparison_chart = TwoAxesInstanciatedChart(abscissa_axis_name='Metric',
                                                primary_ordinate_axis_name='Temperature',
                                                chart_name='Temperature Statistics')

    if fahrenheit_series_data and kelvin_series_data:
        avg_fahrenheit = np.mean(fahrenheit_series_data)
        max_fahrenheit = np.max(fahrenheit_series_data)
        min_fahrenheit = np.min(fahrenheit_series_data)

        avg_kelvin = np.mean(kelvin_series_data)
        max_kelvin = np.max(kelvin_series_data)
        min_kelvin = np.min(kelvin_series_data)

        # Statistics series
        f_stats_series = InstanciatedSeries(['Min', 'Avg', 'Max'], [min_fahrenheit, avg_fahrenheit, max_fahrenheit], 'Fahrenheit Stats', 'bar')
        k_stats_series = InstanciatedSeries(['Min', 'Avg', 'Max'], [min_kelvin, avg_kelvin, max_kelvin], 'Kelvin Stats', 'bar')

        comparison_chart.add_series(f_stats_series)
        comparison_chart.add_series(k_stats_series)

    return [temp_chart, comparison_chart]


@auto_sos_discipline(
    outputs={'fahrenheit_series': list, 'kelvin_series': list, 'time_days': list},
    post_processing_callback=create_temperature_charts
)
def TempConverter(celsius_base: float, days_duration: float) -> dict[str, list[float]]:
    """Generate temperature time series from base Celsius temperature over specified days."""
    # Generate time series (daily measurements)
    time_days = list(range(int(days_duration)))

    # Generate realistic temperature variation (seasonal + daily variation)
    celsius_series = []
    fahrenheit_series = []
    kelvin_series = []

    for day in time_days:
        # Add seasonal variation (sine wave over the period) and some randomness
        seasonal_variation = 10 * np.sin(2 * np.pi * day / max(days_duration, 1))
        daily_noise = np.random.normal(0, 2)  # Random daily variation

        celsius_day = celsius_base + seasonal_variation + daily_noise
        fahrenheit_day = celsius_day * 9/5 + 32
        kelvin_day = celsius_day + 273.15

        celsius_series.append(celsius_day)
        fahrenheit_series.append(fahrenheit_day)
        kelvin_series.append(kelvin_day)

    return {
        'fahrenheit_series': fahrenheit_series,
        'kelvin_series': kelvin_series,
        'time_days': time_days
    }
