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
from pandas import DataFrame

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import (
    PostProcessingFactory,
)


class Study(StudyManager):

    def __init__(self, execution_engine=None) -> None:
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        """Setup test case with sample geographic data"""
        # Sample locations data (representing different types of facilities)
        locations_data = DataFrame([
            {
                'id': 'LOC001',
                'name': 'Central Hub',
                'lat': 48.8566,  # Paris
                'lon': 2.3522,
                'type': 'hub',
                'value': 100.5
            },
            {
                'id': 'LOC002',
                'name': 'Factory North',
                'lat': 50.8503,  # Brussels
                'lon': 4.3517,
                'type': 'factory',
                'value': 75.2
            },
            {
                'id': 'LOC003',
                'name': 'Factory South',
                'lat': 43.2965,  # Marseille
                'lon': 5.3698,
                'type': 'factory',
                'value': 85.7
            },
            {
                'id': 'LOC004',
                'name': 'Warehouse East',
                'lat': 48.5734,  # Orleans
                'lon': 1.9050,
                'type': 'warehouse',
                'value': 45.3
            },
            {
                'id': 'LOC005',
                'name': 'Distribution Center',
                'lat': 47.2184,  # Tours
                'lon': 0.2047,
                'type': 'distribution',
                'value': 60.8
            },
            {
                'id': 'LOC006',
                'name': 'Port Facility',
                'lat': 43.3047,  # Toulon
                'lon': 5.9306,
                'type': 'port',
                'value': 120.4
            }
        ])

        # Sample connections data (representing routes between facilities)
        connections_data = DataFrame([
            {
                'origin_id': 'LOC001',
                'destination_id': 'LOC002',
                'distance': 264.0,  # Paris to Brussels
                'weight': 50.0
            },
            {
                'origin_id': 'LOC001',
                'destination_id': 'LOC003',
                'distance': 661.0,  # Paris to Marseille
                'weight': 75.0
            },
            {
                'origin_id': 'LOC001',
                'destination_id': 'LOC004',
                'distance': 130.0,  # Paris to Orleans
                'weight': 30.0
            },
            {
                'origin_id': 'LOC004',
                'destination_id': 'LOC005',
                'distance': 95.0,   # Orleans to Tours
                'weight': 25.0
            },
            {
                'origin_id': 'LOC003',
                'destination_id': 'LOC006',
                'distance': 68.0,   # Marseille to Toulon
                'weight': 40.0
            },
            {
                'origin_id': 'LOC002',
                'destination_id': 'LOC005',
                'distance': 450.0,  # Brussels to Tours
                'weight': 35.0
            }
        ])

        disc_name = 'DiscMapCharts'
        values_dict = {
            f'{self.study_name}.{disc_name}.locations_data': locations_data,
            f'{self.study_name}.{disc_name}.connections_data': connections_data
        }

        return values_dict


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()

    # Generate post-processing
    ppf = PostProcessingFactory()
    all_post_processings = ppf.get_all_post_processings(
        uc_cls.execution_engine, False, as_json=False, for_test=False
    )

    for post_proc_list in all_post_processings.values():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly().show()
