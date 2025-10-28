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
import traceback

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)
from sostrades_core.tools.post_processing.map_charts.instanciated_map_chart import InstanciatedMapChart, MapStyle


class DiscMapCharts(SoSWrapp):
    """Test discipline for map chart capabilities in SOSTrades core"""

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_map_charts',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': 'Test',
        'definition': 'Test discipline for map-based post-processing capabilities',
        'icon': 'fas fa-map fa-fw',
        'version': '1.0',
    }

    _maturity = 'Fake'

    DESC_IN = {
        'locations_data': {
            'type': 'dataframe',
            'unit': '-',
            'description': 'Geographic locations for mapping',
            'dataframe_descriptor': {
                'id': ('string', None, True),
                'name': ('string', None, True),
                'lat': ('float', None, True),
                'lon': ('float', None, True),
                'type': ('string', None, True),
                'value': ('float', None, True),
            },
            'user_level': 1
        },
        'connections_data': {
            'type': 'dataframe',
            'unit': '-',
            'description': 'Connections between locations',
            'dataframe_descriptor': {
                'origin_id': ('string', None, True),
                'destination_id': ('string', None, True),
                'distance': ('float', None, True),
                'weight': ('float', None, True),
            },
            'user_level': 1
        }
    }

    DESC_OUT = {
        'processed_locations': {
            'type': 'dataframe',
            'unit': '-',
            'description': 'Processed location data with computed values'
        },
        'network_stats': {
            'type': 'dict',
            'unit': '-',
            'description': 'Network statistics and analysis'
        },
        'total_distance': {
            'type': 'float',
            'unit': 'km',
            'description': 'Total network distance'
        }
    }

    def run(self):
        """Execute the discipline computation"""
        # Get inputs
        locations_df = self.get_sosdisc_inputs('locations_data')
        connections_df = self.get_sosdisc_inputs('connections_data')

        # Process locations (add computed values)
        processed_locations = locations_df.copy()
        processed_locations['computed_value'] = processed_locations['value'] * 1.5
        processed_locations['category'] = processed_locations['type'].apply(
            lambda x: 'Primary' if x in ['hub', 'factory'] else 'Secondary'
        )

        # Calculate network statistics
        total_distance = connections_df['distance'].sum() if not connections_df.empty else 0.0
        network_stats = {
            'total_locations': len(locations_df),
            'total_connections': len(connections_df),
            'avg_location_value': locations_df['value'].mean() if not locations_df.empty else 0.0,
            'max_distance': connections_df['distance'].max() if not connections_df.empty else 0.0,
            'min_distance': connections_df['distance'].min() if not connections_df.empty else 0.0
        }

        # Store outputs
        outputs = {
            'processed_locations': processed_locations,
            'network_stats': network_stats,
            'total_distance': total_distance
        }

        self.store_sos_outputs_values(outputs)

    def _create_test_map(self):
        """Create a simple test map showing locations and connections"""
        try:
            self.logger.info("Creating test map...")

            # Get data
            locations_df = self.get_sosdisc_inputs('locations_data')
            connections_df = self.get_sosdisc_inputs('connections_data')
            processed_locations = self.get_sosdisc_outputs('processed_locations')
            network_stats = self.get_sosdisc_outputs('network_stats')

            # Create the Plotly figure
            map_chart = InstanciatedMapChart('map chart')

            # Add connection lines first (so they appear behind markers)
            map_chart.add_trace(locations_df, connections_df)

            return map_chart

        except Exception as e:
            self.logger.error(f"Error creating test map: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def get_chart_filter_list(self):
        """Return list of chart filters"""
        chart_filters = []
        chart_list = ['Network Statistics', 'Location Values', 'Geographic Map']
        chart_filters.append(ChartFilter('Charts', chart_list, chart_list, 'charts'))
        return chart_filters

    def get_post_processing_list(self, filters=None):
        """Generate post-processing charts"""
        instanciated_charts = []
        charts_list = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts_list = chart_filter.selected_values
        else:
            charts_list = ['Network Statistics', 'Location Values', 'Geographic Map']

        # Get data
        processed_locations = self.get_sosdisc_outputs('processed_locations')
        network_stats = self.get_sosdisc_outputs('network_stats')

        # Network Statistics Chart
        if 'Network Statistics' in charts_list and network_stats:
            chart_name = 'Network Statistics'

            stats_keys = list(network_stats.keys())
            stats_values = list(network_stats.values())

            new_chart = TwoAxesInstanciatedChart(
                'Statistics', 'Values',
                chart_name=chart_name
            )

            serie = InstanciatedSeries(
                stats_keys, stats_values, 'Network Stats', 'bar'
            )
            new_chart.series.append(serie)
            new_chart.post_processing_section_name = 'Statistics'
            instanciated_charts.append(new_chart)

        # Location Values Chart
        if 'Location Values' in charts_list and processed_locations is not None and not processed_locations.empty:
            chart_name = 'Location Values Comparison'

            new_chart = TwoAxesInstanciatedChart(
                'Locations', 'Values',
                chart_name=chart_name
            )

            # Original values series
            serie1 = InstanciatedSeries(
                list(processed_locations['name']),
                list(processed_locations['value']),
                'Original Values', 'bar'
            )
            new_chart.series.append(serie1)

            # Computed values series
            serie2 = InstanciatedSeries(
                list(processed_locations['name']),
                list(processed_locations['computed_value']),
                'Computed Values', 'bar'
            )
            new_chart.series.append(serie2)

            new_chart.post_processing_section_name = 'Values Analysis'
            instanciated_charts.append(new_chart)

        # Geographic Map Chart
        if 'Geographic Map' in charts_list :
            map_chart = self._create_test_map()
            if map_chart:
                instanciated_charts.append(map_chart)

            other_map_chart = self._create_test_map()
            other_map_chart.chart_name = 'Alternate Map View: USGS Topo'
            other_map_chart.set_map_style(MapStyle.USGS_TOPO)
            instanciated_charts.append(other_map_chart)

            other_map_chart = self._create_test_map()
            other_map_chart.chart_name = 'Alternate Map View: USGS Hydrography'
            other_map_chart.set_map_style(MapStyle.USGS_HYDROGRAPHY)
            instanciated_charts.append(other_map_chart)

            other_map_chart = self._create_test_map()
            other_map_chart.chart_name = 'Alternate Map View: USGS Imagery'
            other_map_chart.set_map_style(MapStyle.USGS_IMAGERY)
            instanciated_charts.append(other_map_chart)

            other_map_chart = self._create_test_map()
            other_map_chart.chart_name = 'Alternate Map View: USGS Imagery Topo'
            other_map_chart.set_map_style(MapStyle.USGS_IMAGERY_TOPO)
            instanciated_charts.append(other_map_chart)

            other_map_chart = self._create_test_map()
            other_map_chart.chart_name = 'Alternate Map View: USGS Shadded Relief'
            other_map_chart.set_map_style(MapStyle.USGS_SHADED_RELIEF)
            instanciated_charts.append(other_map_chart)

        return instanciated_charts
