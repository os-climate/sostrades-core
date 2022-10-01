'''
Copyright 2022 Airbus SAS

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
from os.path import join, dirname

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.tools.dashboard.dashboard_factory import generate_dashboard
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
import pandas as pd
from sos_trades_core.tools.vectorization.vectorization_methods import *


class TestDashboard(unittest.TestCase):
    """
    Base Function test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'EETests'
        self.repo = 'sos_trades_core.sos_processes.test'
        self.file_to_del = join(dirname(__file__), 'MyCase.csv')

    def test_01_test_dashboard_creation(self):
        '''
        check_var_data_mismatch method in sos_coupling (not recursive)
        '''
        namespace = 'MyCase'
        ee = ExecutionEngine(namespace)
        ee.select_root_process(self.repo, 'test_disc1_disc2_coupling')
        ee.configure()
        # check treeview structure
        exp_tv_list = ['Nodes representation for Treeview MyCase',
                       '|_ MyCase',
                       '\t|_ Disc1',
                       '\t|_ Disc2']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == ee.display_treeview_nodes()
        #-- setup inputs
        dm = ee.dm
        values_dict = {}
        values_dict[f'{namespace}.Disc2.constant'] = -10.
        values_dict[f'{namespace}.Disc2.power'] = -10
        values_dict[f'{namespace}.Disc1.a'] = 10.
        values_dict[f'{namespace}.Disc1.b'] = 20.
        values_dict[f'{namespace}.Disc1.indicator'] = 10.
        values_dict[f'{namespace}.x'] = 3.

        dm.set_values_from_dict(values_dict)

        ee.configure()
        ee.execute()
        post_processings = PostProcessingFactory().get_all_post_processings(ee, False)
        dashboard = generate_dashboard(ee, post_processings)

        validated_dashboard = {
            "title": "Disc1-Disc2 coupling",
            "rows": [
                [
                    {
                        "content_type": "TEXT",
                        "content": "This is a dashboard test ",
                        "style": "text-align:left"
                    }
                ],
                [
                    {
                        "content_type": "SCALAR",
                        "content_namespace": "{study_name}.Disc1.a",
                        "content": {"var_name": "a", "value": 10.0}
                    }
                ],
                [
                    {
                        "content_type": "POST_PROCESSING",
                        "content_namespace": "{study_name}.Disc1",
                        "graph_index": 0,
                        "content": {'data': [{'mode': 'lines', 'name': '', 'visible': True, 'x': [3.0], 'y': [50.0], 'yaxis': 'y', 'type': 'scatter'}],
                                    'layout': {'title': {'text': '<b>y vs x</b>', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, 'xaxis': {'title': {'text': 'x (-)'}, 'automargin': True}, 'yaxis': {'title': {'text': 'y (-)'}, 'automargin': True},
                                               'yaxis2': {'title': {'text': ''}, 'automargin': True, 'anchor': 'x', 'overlaying': 'y', 'side': 'right'}, 'legend': {'orientation': 'h', 'xanchor': 'left', 'yanchor': 'top', 'bgcolor': 'rgba(255, 255, 255, 0)', 'bordercolor': 'rgba(255, 255, 255, 0)', 'y': -0.2, 'x': 0},
                                               'font': {'family': 'Arial', 'size': 10, 'color': '#7f7f7f'}, 'barmode': 'group', 'width': 1200, 'height': 450, 'autosize': False},
                                    'csv_data': ['x (-),y (-)', '3.0,50.0'], 'logo_notofficial': False, 'logo_official': False, 'logo_work_in_progress': False, 'is_pareto_trade_chart': False, 'config': {'responsive': True}
                                    }}
                ],
                [
                    {
                        "content_type": "INVALID_CONTENT_TYPE",
                        "content_namespace": "{study_name}.Disc1",
                        "graph_index": 0
                    }
                ]
            ]
        }
        assert validated_dashboard == dashboard
