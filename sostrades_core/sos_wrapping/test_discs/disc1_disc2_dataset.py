'''
Copyright 2023 Capgemini

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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter


class Disc1(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'Disc1 for simple dataset testing purpose',
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
        'x': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_xy_disc1'},
        'a': {'type': 'int', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_a'},
        'b': {'type': 'string', 'unit': '-'}
    }
    DESC_OUT = {
        'flag': {'type': 'string', 'unit': '-'},
        'y': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_xy_disc1'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        dict_values = {'flag': b, 'y': a * x}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['y vs x']

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'graphs'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        if 'y vs x' in charts_list:

            chart_name = 'y vs x'

            y = self.get_sosdisc_outputs('y')
            x = self.get_sosdisc_inputs('x')
            print(y, x)
            new_chart = TwoAxesInstanciatedChart('x (-)', 'y (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                [x], [y], '', 'scatter')

            new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

        return instanciated_charts


class Disc2(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'Disc2 for simple dataset testing purpose',
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
        'x': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_xy_disc2'},
        'a': {'type': 'int', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_a'},
        'b': {'type': 'string', 'unit': '-'}
    }
    DESC_OUT = {
        'flag': {'type': 'string', 'unit': '-'},
        'y': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_xy_disc2'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        dict_values = {'flag': b, 'y': a * x}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['y vs x']

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'graphs'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        if 'y vs x' in charts_list:

            chart_name = 'y vs x'

            y = self.get_sosdisc_outputs('y')
            x = self.get_sosdisc_inputs('x')
            print(y, x)
            new_chart = TwoAxesInstanciatedChart('x (-)', 'y (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                [x], [y], '', 'scatter')

            new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

        return instanciated_charts