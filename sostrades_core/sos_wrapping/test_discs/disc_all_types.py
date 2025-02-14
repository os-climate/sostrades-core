'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
# coding: utf-8
from numpy import array
from pandas import DataFrame

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class DiscAllTypes(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_all_types',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'z': {'type': 'float', 'default': 90., 'unit': 'kg', 'user_level': 1,
              'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test', 'range': [-1e4, 1e4]},
        'z_list': {'type': 'list', ProxyDiscipline.SUBTYPE: {'list': 'float'}, 'unit': 'kg', 'user_level': 1,
                   'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test', 'range': [-1e4, 1e4]},
        'h': {'type': 'array', 'unit': 'kg', 'user_level': 1},
        'dict_in': {'type': 'dict', ProxyDiscipline.SUBTYPE: {'dict': 'float'}, 'unit': 'kg', 'user_level': 1},
        'dict_string_in': {'type': 'dict', ProxyDiscipline.SUBTYPE: {'dict': 'string'}, 'unit': 'kg', 'user_level': 1},
        'list_dict_string_in': {'type': 'list', ProxyDiscipline.SUBTYPE: {'list': {'dict': 'string'}}, 'unit': 'kg', 'user_level': 1},
        'df_in': {'type': 'dataframe', 'unit': 'kg', 'user_level': 1, 'dataframe_descriptor': {'variable': ('float', [-1e4, 1e4], True),  # input function
                                                                                               'c2': ('float', None, True), 'c3': ('float', None, True), },
                  'dataframe_edition_locked': False, },
        'weather': {'type': 'string', 'default': 'cloudy, it is Toulouse ...', 'user_level': 1,
                    'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test', 'possible_values': ['cloudy, it is Toulouse ...', 'sunny', 'rainy']},
        'weather_list': {'type': 'list', ProxyDiscipline.SUBTYPE: {'list': 'string'}, 'default': ['cloudy, it is Toulouse ...'], 'user_level': 1,
                         'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test', 'possible_values': ['cloudy, it is Toulouse ...', 'sunny', 'rainy']},
        'dict_of_dict_in': {'type': 'dict', ProxyDiscipline.SUBTYPE: {'dict': {'dict': 'float'}}, 'user_level': 1},
        'dict_of_df_in': {'type': 'dict', ProxyDiscipline.SUBTYPE: {'dict': 'dataframe'}, 'user_level': 1}
    }
    DESC_OUT = {
        'df_out': {'type': 'dataframe', 'unit': 'kg', 'user_level': 1,
                   'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'o': {'type': 'array', 'unit': 'kg', 'user_level': 1,
              'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'unit': 'kg', 'user_level': 1,
                     'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):
        h = self.get_sosdisc_inputs('h')
        z = self.get_sosdisc_inputs('z')
        dict_in = self.get_sosdisc_inputs('dict_in')
        df_in = self.get_sosdisc_inputs('df_in')
        dict_of_dict_in = self.get_sosdisc_inputs('dict_of_dict_in')
        z = z / 3.1416
        key1 = dict_in['key1'] + dict_of_dict_in['key_A']['subKey2']
        key2 = df_in['c2'][0] * dict_of_dict_in['key_B']['subKey1']
        h = array([0.5 * (h[0] + 1. / (2 * key1)),
                   0.5 * (h[-1] + 1. / (2 * key2))])
        dict_out = {'key1': ((h[0] + h[1]) / z * 100),
                    'key2': ((h[0] + h[1]) / z * 100)}
        dict_values = {'o': z,
                       'dict_out': dict_out}
        df_in = DataFrame(array([[(h[0] + h[1]) / 2, (h[0] + h[1]) / 2]]),
                          columns=['c1', 'c2'])
        df_in['z'] = [2 * z] * len(df_in)
        dict_values.update({'df_out': df_in})
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['c1 vs h', 'c2 vs h', 'z vs h']

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'graphs'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []
        charts_list = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values
        else:
            charts_list = ['c1 vs h', 'c2 vs h', 'z vs h']

        if 'c1 vs h' in charts_list:
            chart_name = 'c1 vs h'

            df_out = self.get_sosdisc_outputs('df_out')
            h = self.get_sosdisc_inputs('h')
            new_chart = TwoAxesInstanciatedChart('h (-)', 'c1 (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                list(h), list(df_out['c1'].values), '', 'scatter')

            new_chart.series.append(serie)
            new_chart.post_processing_section_name = 'section 1'
            new_chart.post_processing_section_is_opened = True
            instanciated_charts.append(new_chart)



        if 'c2 vs h' in charts_list:
            chart_name = 'c2 vs h'

            df_out = self.get_sosdisc_outputs('df_out')
            h = self.get_sosdisc_inputs('h')
            new_chart = TwoAxesInstanciatedChart('h (-)', 'c2 (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                list(h), list(df_out['c2'].values), '', 'scatter')

            new_chart.series.append(serie)
            new_chart.post_processing_section_name = 'section 2'
            new_chart.post_processing_is_key_chart = True
            instanciated_charts.append(new_chart)

        if 'z vs h' in charts_list:
            chart_name = 'z vs h'

            df_out = self.get_sosdisc_outputs('df_out')
            h = self.get_sosdisc_inputs('h')
            new_chart = TwoAxesInstanciatedChart('h (-)', 'z (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                list(h), list(df_out['z'].values), '', 'scatter')

            new_chart.series.append(serie)
            instanciated_charts.append(new_chart)

            chart_name = 'z in section 3'

            df_out = self.get_sosdisc_outputs('df_out')
            h = self.get_sosdisc_inputs('h')
            new_chart = TwoAxesInstanciatedChart('h (-)', 'z (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                list(h), list(df_out['z'].values), '', 'scatter')

            new_chart.series.append(serie)
            new_chart.post_processing_section_name = 'KPIs'
            new_chart.post_processing_section_is_opened = False
            instanciated_charts.append(new_chart)
        return instanciated_charts
