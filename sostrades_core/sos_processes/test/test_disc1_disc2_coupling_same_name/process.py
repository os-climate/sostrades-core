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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# -- Generate test 2 process

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import TwoAxesInstanciatedChart, \
    InstanciatedSeries
from sostrades_core.execution_engine.data_manager import DataManager

class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Disc2 Coupling Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        same_name = 'SameName'
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        disc1_builder = self.ee.factory.get_builder_from_module(
            same_name, disc_dir+'disc1.Disc1')
        disc2_builder = self.ee.factory.get_builder_from_module(
            same_name, disc_dir+'disc2.Disc2')
        return [disc1_builder]+[disc2_builder]
