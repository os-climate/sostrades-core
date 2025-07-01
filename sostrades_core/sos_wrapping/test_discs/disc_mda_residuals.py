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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class Disc1Residuals(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc1Residuals',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'w1': {'type': 'float', 'unit': '-', 'default': 0.0},
        'y2': {'type': 'float', 'unit': '-', 'default': 2.0,  'namespace': 'ns_coupling'},
        'x': {'type': 'float', 'unit': '-', 'default': 3.0,  'namespace': 'ns_coupling'}
    }
    DESC_OUT = {
        'w1': {'type': 'float', 'unit': '-'},
        'y1': {'type': 'float', 'unit': '-',  'namespace': 'ns_coupling'},
        'r1': {'type': 'float', 'unit': '-'}
    }

    def run(self):
        # From https://gitlab.com/gemseo/dev/gemseo/-/blob/develop/tests/mda/test_mda_residuals.py
        w1, y2, x = self.get_sosdisc_inputs(['w1', 'y2', 'x'])
        w1 = (3 * x - y2) / 7.0
        y1 = 5 * w1 + x + 3 * y2
        r1 = y2 - 3 * x + 7 * w1

        self.store_sos_outputs_values({'y1': y1, 'w1': w1, 'r1': r1})

    def get_chart_filter_list(self):

        chart_filters = []

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []
        charts_list = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        return instanciated_charts


class Disc2Residuals(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc2Residuals',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'w2': {'type': 'float', 'unit': '-', 'default': 0.0},
        'y1': {'type': 'float', 'unit': '-', 'default': 2.0,  'namespace': 'ns_coupling'},
        'x': {'type': 'float', 'unit': '-', 'default': 3.0,  'namespace': 'ns_coupling'}
    }
    DESC_OUT = {
        'y2': {'type': 'float', 'unit': '-',  'namespace': 'ns_coupling'},
        'w2': {'type': 'float', 'unit': '-'},
        'r2': {'type': 'float', 'unit': '-'}
    }

    def run(self):
        # From https://gitlab.com/gemseo/dev/gemseo/-/blob/develop/tests/mda/test_mda_residuals.py
        w2, y1, x = self.get_sosdisc_inputs(['w2', 'y1', 'x'])
        w2 = (2 * x - y1) / 5.0
        y2 = 13 * w2 + x + 2 * y1

        r2 = y1 - 2 * x + 5 * w2

        self.store_sos_outputs_values({'y2': y2, 'w2': w2, 'r2': r2})

    def get_chart_filter_list(self):

        chart_filters = []

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []
        charts_list = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        return instanciated_charts
