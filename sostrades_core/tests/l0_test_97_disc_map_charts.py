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
import unittest

import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestDiscMapCharts(unittest.TestCase):
    """Test class for the map charts discipline"""

    def setUp(self):
        """Set up test environment"""
        self.name = 'test_disc_charts'
        self.ee = ExecutionEngine(self.name)
        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_disc_charts'

    def test_disc_map_charts_execution(self):
        """Test the basic execution of the map charts discipline"""
        # Build the process
        builder = self.ee.factory.get_builder_from_process(
            repo=self.repo,
            mod_id=self.proc_name
        )
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()

        # Setup test data
        locations_data = pd.DataFrame([
            {
                'id': 'LOC001',
                'name': 'Test Hub',
                'lat': 48.8566,
                'lon': 2.3522,
                'type': 'hub',
                'value': 100.0
            },
            {
                'id': 'LOC002',
                'name': 'Test Factory',
                'lat': 50.8503,
                'lon': 4.3517,
                'type': 'factory',
                'value': 75.0
            }
        ])

        connections_data = pd.DataFrame([
            {
                'origin_id': 'LOC001',
                'destination_id': 'LOC002',
                'distance': 264.0,
                'weight': 50.0
            }
        ])

        # Set input values
        values_dict = {
            f'{self.name}.DiscMapCharts.locations_data': locations_data,
            f'{self.name}.DiscMapCharts.connections_data': connections_data
        }

        self.ee.load_study_from_input_dict(values_dict)

        # Execute
        self.ee.execute()

        # Check outputs
        processed_locations = self.ee.dm.get_value(f'{self.name}.DiscMapCharts.processed_locations')
        network_stats = self.ee.dm.get_value(f'{self.name}.DiscMapCharts.network_stats')
        total_distance = self.ee.dm.get_value(f'{self.name}.DiscMapCharts.total_distance')

        # Assertions
        self.assertIsNotNone(processed_locations)
        self.assertIsNotNone(network_stats)
        self.assertIsNotNone(total_distance)
        self.assertEqual(len(processed_locations), 2)
        self.assertEqual(network_stats['total_locations'], 2)
        self.assertEqual(network_stats['total_connections'], 1)
        self.assertEqual(total_distance, 264.0)

        # Test post-processing generation
        disc = self.ee.dm.get_disciplines_with_name(f'{self.name}.DiscMapCharts')[0]

        # Test chart filters
        chart_filters = disc.get_chart_filter_list()
        self.assertIsNotNone(chart_filters)
        self.assertEqual(len(chart_filters), 1)

        # Test post-processing list generation
        post_processings = disc.get_post_processing_list()
        self.assertIsNotNone(post_processings)
        self.assertGreater(len(post_processings), 0)

        print("✓ All tests passed successfully!")
        print(f"✓ Generated {len(post_processings)} post-processing charts")
        print(f"✓ Network stats: {network_stats}")


if __name__ == '__main__':
    unittest.main()
