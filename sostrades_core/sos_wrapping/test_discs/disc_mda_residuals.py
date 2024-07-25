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
from sostrades_core.sos_wrapping.test_discs.test_mda_residuals_gemseo import (
    disc_1_expr,
    disc_2_expr,
)


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
        'w1': {'type': 'float', 'unit': '-','default':0.0},
        'y2': {'type': 'float', 'unit': '-','default':2.0,'visibility':'Shared','namespace':'ns_coupling'},
        'x': {'type': 'float', 'unit': '-','default':3.0,'visibility':'Shared','namespace':'ns_coupling'}
    }
    DESC_OUT = {
        'w1': {'type': 'float', 'unit': '-'},
        'y1': {'type': 'float', 'unit': '-','visibility':'Shared','namespace':'ns_coupling'},
        'r1': {'type': 'float', 'unit': '-'}
    }

    def run(self):
        w1, y2, x = self.get_sosdisc_inputs(['w1','y2','x'])
        y1, w1, r1=disc_1_expr(w1, y2, x)
        self.store_sos_outputs_values({'y1':y1,'w1':w1,'r1':r1})
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
        'w2': {'type': 'float', 'unit': '-','default':0.0},
        'y1': {'type': 'float', 'unit': '-','default':2.0,'visibility':'Shared','namespace':'ns_coupling'},
        'x': {'type': 'float', 'unit': '-','default':3.0,'visibility':'Shared','namespace':'ns_coupling'}
    }
    DESC_OUT = {
        'y2': {'type': 'float', 'unit': '-','visibility':'Shared','namespace':'ns_coupling'},
        'w2': {'type': 'float', 'unit': '-'},
        'r2': {'type': 'float', 'unit': '-'}
    }

    def run(self):
        w1, y2, x = self.get_sosdisc_inputs(['w2','y1','x'])
        y2, w2, r2=disc_2_expr(w1, y2, x)
        self.store_sos_outputs_values({'y2':y2,'w2':w2,'r2':r2})
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
