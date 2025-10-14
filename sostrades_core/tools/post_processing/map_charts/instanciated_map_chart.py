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

import plotly.graph_objects as go
from pandas import DataFrame

from sostrades_core.tools.post_processing.post_processing_plotly_tooling import AbstractPostProcessingPlotlyTooling


class InstanciatedMapChart(AbstractPostProcessingPlotlyTooling):
    """
    Specialized chart class for geographic map visualization in SOSTrades.
    This chart uses a placeholder URL for tiles that will be replaced by the
    post-processing server with the actual server base URL + proxy endpoints.
    """

    def __init__(self,
                 chart_name="Geographic Network Map", 
                 locations_df = None,
                 connections_df = None,
                 marker_config=None,
                 initial_zoom = 2
                 ):
        """
        Initialize map chart
        Args:
            chart_name: Name of the chart
            locations_df: DataFrame with columns ['id', 'name', 'lat', 'lon', 'type', 'value']
            connections_df: DataFrame with columns ['origin_id', 'destination_id', 'distance', 'weight']
            marker_config: Configuration for the markers on the map, dict with location type as keys, values as dict with 'color', 'size', 'symbol'
            initial_zoom: Initial zoom level for the map

        """
        super().__init__()
        self.locations_df = None
        self.connections_df = None
        self.add_trace(locations_df, connections_df)
        self.chart_name = chart_name
        self.initial_zoom = initial_zoom

        # Store map-specific data
        if marker_config is None:
            self.marker_config = {
                'hub': {'color': 'red', 'size': 15, 'symbol': 'circle'},
                'factory': {'color': 'orange', 'size': 12, 'symbol': 'circle'},
                'warehouse': {'color': 'blue', 'size': 10, 'symbol': 'circle'},
                'distribution': {'color': 'green', 'size': 10, 'symbol': 'circle'},
                'port': {'color': 'purple', 'size': 12, 'symbol': 'circle'}
            }
        else:
            self.marker_config = marker_config

        # Placeholder tile URL - will be replaced by post-processing server
        self.tile_url_placeholder = "https://tile.openstreetmap.org"

        # Chart type identifier for post-processing
        self.chart_type = "geographic_map"


    def add_trace(self, locations_df, connections_df):
        """
        Method to add trace to current spider chart

        @param location_df: DataFrame with location data
        @param connection_df: DataFrame with connection data
        """
        if locations_df is not None:
            if not isinstance(locations_df, DataFrame) or locations_df.columns.tolist() != ['id', 'name', 'lat', 'lon', 'type', 'value']:
                message = f'"locations_df" argument is intended to be a DataFrame with columns ["id", "name", "lat", "lon", "type", "value"] not {type(locations_df)}'
                raise TypeError(message)
            # merge data, get new from location_df while keeping old ones
            if self.locations_df is None:
                self.locations_df = locations_df
            else:
                self.locations_df = self.locations_df.merge(locations_df, on='id', how='outer', ignore_index=True)

        if connections_df is not None:
            if not isinstance(connections_df, DataFrame) or connections_df.columns.tolist() != ['origin_id', 'destination_id', 'distance', 'weight']:
                message = f'"connections_df" argument is intended to be a DataFrame with columns ["origin_id", "destination_id", "distance", "weight"] not {type(connections_df)}'
                raise TypeError(message)
            if self.connections_df is None:
                self.connections_df = connections_df
            else:
                self.connections_df = self.connections_df.merge(connections_df, on=['origin_id', 'destination_id', 'distance'], how='outer', ignore_index=True)


    def to_plotly(self, app_logger=None):
        fig = go.Figure()
        """Convert the map chart to a Plotly figure"""
        self._configure_map_layout(fig, self.locations_df)
        if self.locations_df is not None:
            self._add_locations_to_figure(fig, self.locations_df)
        if self.connections_df is not None:
            self._add_connections_to_figure(fig, self.locations_df, self.connections_df)

        
        return fig


    def _add_connections_to_figure(self, fig, locations_df, connections_df):
        """Add connection lines to the map"""
        for _, conn_row in connections_df.iterrows():
            origin_id = conn_row['origin_id']
            dest_id = conn_row['destination_id']
            distance = conn_row.get('distance', 0)
            weight = conn_row.get('weight', 1)

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

    def _add_locations_to_figure(self, fig, locations_df):
        """Add location markers to the map"""
        for _, loc_row in locations_df.iterrows():
            name = loc_row['name']
            lat = loc_row['lat']
            lon = loc_row['lon']
            loc_type = loc_row.get('type', 'default')
            value = loc_row.get('value', 0)

            # Marker properties based on type
            marker_config = self._get_marker_config(loc_type)

            hover_text = (
                f"<b>{name}</b><br>"
                f"Type: {loc_type}<br>"
                f"Value: {value:.2f}<br>"
                f"Coordinates: ({lat:.3f}, {lon:.3f})"
            )

            fig.add_trace(go.Scattermap(
                mode="markers+text",
                lon=[lon],
                lat=[lat],
                text=[name],
                textposition='top center',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=[hover_text],
                name=f"{loc_type.title()} Locations",
                showlegend=True
            ))

    def _get_marker_config(self, loc_type):
        """Get marker configuration based on location type"""
        return self.marker_config.get(loc_type, {'color': 'gray', 'size': 8, 'symbol': 'circle'})

    def _configure_map_layout(self, fig, location_df):
        """Configure the map layout with placeholder tile URL"""
        center_lat = location_df['lat'].mean() if location_df is not None and not location_df.empty else 0
        center_lon = location_df['lon'].mean() if location_df is not None and not location_df.empty else 0

        # Custom mapbox style with placeholder URL
        tiles_url = self.tile_url_placeholder + "/{z}/{x}/{y}.png"

        custom_style = {
            'version': 8,
            'glyphs': 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',  # Ajout de la propriété glyphs
            'sources': {
                'osm-tiles': {
                    'type': 'raster',
                    "below": "traces",
                    'tiles': [tiles_url],  # Placeholder URL
                    'tileSize': 256,
                    'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }
            },
            'layers': [
                {
                    'id': 'osm-layer',
                    'type': 'raster',
                    'source': 'osm-tiles',
                    'minzoom': 0,
                    'maxzoom': 18,
                    "below": "traces"
                }
            ]
        }

        fig.update_layout(
            map=dict(
                style=custom_style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=self.initial_zoom,

            ),
            title=dict(
                text=self.chart_name
            ),
            legend=dict(
            orientation="h",           # horizontal
            x=0.5,                    # centré horizontalement
            y=-0.05,                  # en dessous de la carte
            xanchor="center",         # ancrage centré
            yanchor="top",            # ancrage en haut de la légende
            bgcolor="rgba(255, 255, 255, 0.9)",  # fond blanc semi-transparent
            bordercolor="rgba(0, 0, 0, 0.2)",    # bordure grise
            borderwidth=1,
            font=dict(size=10),       # taille de police réduite
            itemsizing="constant",    # taille constante des items
            itemwidth=30,             # largeur des items réduite
            tracegroupgap=10          # espacement entre les groupes
        )
        )

        
    def __to_csv(self):
        global_list = []
        header = []
        max_len = 0

        for serie in self.__traces:
            if serie.series_name is not None and len(serie.series_name) > 0:
                header.append(f'{serie.series_name} {self.abscissa_axis_name}')
                header.append(
                    f'{serie.series_name} {self.primary_ordinate_axis_name}')
            else:
                header.append(f'{self.abscissa_axis_name}')
                header.append(f'{self.primary_ordinate_axis_name}')

            global_list.append(serie.abscissa)
            if len(serie.abscissa) > max_len:
                max_len = len(serie.abscissa)

            global_list.append(serie.ordinate)
            if len(serie.ordinate) > max_len:
                max_len = len(serie.ordinate)

        csv_list = [','.join(header)]

        for i in range(max_len):
            csv_line = []
            for gl in global_list:
                if i < len(gl):
                    csv_line.append(f'{gl[i]}')
                else:
                    csv_line.append('')
            csv_list.append(','.join(csv_line))

        self.set_csv_data(csv_list)

    def to_plotly_dict(self, logger=None):
        """
        Method that convert current instance to plotly object and then to a dictionary

        @param logger: logger instance
        @type Logging.loger
        """
        json = self.to_plotly(logger).to_dict()
        #json[self.CSV_DATA] = self._plot_csv_data

        #add chart metadata as watermarks or sections
        json.update(self.get_metadata_dict())
        json.update({'tile_url_placeholder': self.tile_url_placeholder})

        return json
