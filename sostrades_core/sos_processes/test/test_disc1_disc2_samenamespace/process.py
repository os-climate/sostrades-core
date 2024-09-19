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

from sostrades_core.execution_engine.data_manager import DataManager
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Disc2 Coupling Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.'
        mods_dict = {'Disc1': disc_dir + 'disc1.Disc1',
                     'Disc2': disc_dir + 'disc2.Disc2', }
        builder_list = self.create_builder_list(mods_dict, ns_dict={'ns_ac': f'{self.ee.study_name}.DiscSameNamespace',
                                                                    'ns_2':f'{self.ee.study_name}.DiscSameNamespace'})

        for builder in builder_list :
            self.ee.ns_manager.add_display_ns_to_builder(
            builder, f'{self.ee.study_name}.DiscSameNamespace')
        self.ee.post_processing_manager.add_post_processing_functions_to_namespace(
            'ns_ac', post_processing_filters, post_processings)
        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_2', 'sostrades_core.sos_processes.test.test_disc1_disc2_coupling.post_processing_libraries')

        return builder_list

def post_processing_filters(execution_engine, namespace):
    """ post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :returns: ChartFilter[]
    """

    filters = []
    (x, y) = get_x_and_y(execution_engine, namespace)

    if x is not None and y is not None:
        filters.append(ChartFilter('Selected charts', filter_values=['X versus Y'], selected_values=[
            'X versus Y'], filter_key='x_vs_y', multiple_selection=True))

    return filters


def post_processings(execution_engine, namespace, filters):
    """ post processing function designed to build a rc vs saleprice 2D chart

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :params: filters, list of filters to applies to the post processing
    :type: ChartFilter[]

    :returns: list of post processing
    """
    chart_results = []

    return chart_results


def get_x_and_y(execution_engine, namespace):
    """ post processing function designed to build check if data can be retrieved
    (x and y)

    :params: execution_engine, execution engine instance that hold data
    :type: ExecutionEngine

    :params: namespace, namespace value that request post processing
    :type: string

    :returns: tuple (rc dataframe, sale price dataframe)
    """

    x = None
    y = None

    try:
        x_namespace = f'{namespace}.x'
        y_namespace = f'{namespace}.y'

        x_key = execution_engine.dm.get_data_id(x_namespace)
        y_key = execution_engine.dm.get_data_id(y_namespace)

        x = execution_engine.dm.data_dict[x_key][DataManager.VALUE]
        y = execution_engine.dm.data_dict[y_key][DataManager.VALUE]
    except:
        x = None
        y = None

    return (x, y)
