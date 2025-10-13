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

import plotly.graph_objects as go

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class MapChart(TwoAxesInstanciatedChart):
    """Custom chart class for map visualization"""

    def __init__(self, fig, chart_name="Test Map Chart"):
        # Initialize with dummy data since we'll override to_plotly
        super().__init__('', '', chart_name=chart_name)
        self.geographic_fig = fig

    def to_plotly(self, logger=None):
        """Override to return our custom geographic figure"""
        return self.geographic_fig


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
            fig = go.Figure()

            # Add connection lines first (so they appear behind markers)
            if not connections_df.empty:
                for _, conn_row in connections_df.iterrows():
                    origin_id = conn_row['origin_id']
                    dest_id = conn_row['destination_id']
                    distance = conn_row['distance']
                    weight = conn_row['weight']

                    # Find origin and destination coordinates
                    origin_loc = locations_df[locations_df['id'] == origin_id]
                    dest_loc = locations_df[locations_df['id'] == dest_id]

                    if not origin_loc.empty and not dest_loc.empty:
                        origin_lat = origin_loc.iloc[0]['lat']
                        origin_lon = origin_loc.iloc[0]['lon']
                        dest_lat = dest_loc.iloc[0]['lat']
                        dest_lon = dest_loc.iloc[0]['lon']

                        # Line width based on weight
                        line_width = max(1, min(10, weight / 10))

                        # Add connection line
                        fig.add_trace(go.Scattermap(
                            mode="lines",
                            lon=[origin_lon, dest_lon],
                            lat=[origin_lat, dest_lat],
                            line=dict(width=line_width, color='rgba(52, 152, 219, 0.6)'),
                            name=f"Connection {origin_id}->{dest_id}",
                            hoverinfo="skip",
                            showlegend=False
                        ))

                        # Add midpoint marker for connection info
                        mid_lon = (origin_lon + dest_lon) / 2
                        mid_lat = (origin_lat + dest_lat) / 2
                        hover_text = (
                            f"<b>Connection</b><br>"
                            f"From: {origin_loc.iloc[0]['name']}<br>"
                            f"To: {dest_loc.iloc[0]['name']}<br>"
                            f"Distance: {distance:.1f} km<br>"
                            f"Weight: {weight:.1f}"
                        )

                        fig.add_trace(go.Scattermap(
                            mode="markers",
                            lon=[mid_lon],
                            lat=[mid_lat],
                            marker=dict(size=4, color='rgba(52, 152, 219, 0.8)', symbol='triangle'),
                            hovertemplate='%{customdata}<extra></extra>',
                            customdata=[hover_text],
                            showlegend=False
                        ))

            # Add location markers
            if not locations_df.empty:
                for _, loc_row in locations_df.iterrows():
                    location_id = loc_row['id']
                    name = loc_row['name']
                    lat = loc_row['lat']
                    lon = loc_row['lon']
                    loc_type = loc_row['type']
                    value = loc_row['value']

                    # Get processed value if available
                    processed_value = value
                    if processed_locations is not None and not processed_locations.empty:
                        proc_loc = processed_locations[processed_locations['id'] == location_id]
                        if not proc_loc.empty:
                            processed_value = proc_loc.iloc[0]['computed_value']

                    # Marker properties based on type
                    if loc_type == 'hub':
                        marker_color = 'red'
                        marker_size = 15
                        marker_symbol = 'star'
                    elif loc_type == 'factory':
                        marker_color = 'orange'
                        marker_size = 12
                        marker_symbol = 'square'
                    else:
                        marker_color = 'blue'
                        marker_size = 10
                        marker_symbol = 'circle'

                    hover_text = (
                        f"<b>{name}</b><br>"
                        f"Type: {loc_type}<br>"
                        f"Original Value: {value:.2f}<br>"
                        f"Processed Value: {processed_value:.2f}<br>"
                        f"Coordinates: ({lat:.3f}, {lon:.3f})"
                    )

                    fig.add_trace(go.Scattermap(
                        mode="markers+text",
                        lon=[lon],
                        lat=[lat],
                        text=[name],
                        textposition='top center',
                        textfont=dict(size=10, color=marker_color, family='Arial Black'),
                        marker=dict(size=marker_size, color=marker_color, symbol=marker_symbol),
                        hovertemplate='%{customdata}<extra></extra>',
                        customdata=[hover_text],
                        name=f"{loc_type.title()} Locations",
                        showlegend=True
                    ))

            # Update layout
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(
                        lat=locations_df['lat'].mean() if not locations_df.empty else 0,
                        lon=locations_df['lon'].mean() if not locations_df.empty else 0
                    ),
                    zoom=5
                ),
                title=dict(
                    text='Test Geographic Network Map',
                    x=0.5,
                    font=dict(size=18, color='#2C3E50'),
                    pad=dict(t=20)
                ),
                height=600,
                width=1000,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='#FFFFFF'
            )

            # Add summary statistics annotation
            if network_stats:
                summary_text = (
                    f"<b>Network Summary</b><br>"
                    f"Locations: {network_stats.get('total_locations', 0)}<br>"
                    f"Connections: {network_stats.get('total_connections', 0)}<br>"
                    f"Avg Value: {network_stats.get('avg_location_value', 0):.2f}<br>"
                    f"Total Distance: {self.get_sosdisc_outputs('total_distance'):.1f} km"
                )

                fig.add_annotation(
                    text=summary_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.98, y=0.98,
                    xanchor='right', yanchor='top',
                    font=dict(size=11, color='#2C3E50'),
                    bgcolor='rgba(255, 255, 255, 0.95)',
                    bordercolor='#3498DB',
                    borderwidth=2,
                    borderpad=10
                )

            # Create chart object
            map_chart = MapChart(fig, "Test Geographic Network Map")
            self.logger.info("Test map created successfully")
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

        return instanciated_charts
