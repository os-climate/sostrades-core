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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from numpy import std

class SimpleDisc1(SoSWrapp):
    """ Discipline used in Driver coupling of simple discipline output with driver subprocess input.
    """
    _maturity = 'Fake'
    DESC_IN = {'z_in': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'z': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'sampling_algo': {'type': 'string', 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                  'namespace': 'ns_doe_eval'}}


    def run(self):
        """ Discipline 1 execution
        """
        z_in = self.get_sosdisc_inputs(['z_in'])
        z = self.compute_z(z_in)
        z_out = {'z': z}
        self.store_sos_outputs_values(z_out)

        sampling_algo = self.decide_sampling_algo(z_in)
        self.store_sos_outputs_values({'sampling_algo': sampling_algo})


    @staticmethod
    def compute_z(z):
        """ Computes the output of the simple discipline in array form
        """
        out = z * pow(2, -1)
        return out

    @staticmethod
    def decide_sampling_algo(z):
        """ Computes the output of the simple discipline in array form
        """

        if z[0] > 0.5:
            sampling_algo = "lhs"
        else:
            sampling_algo = "fullfact"

        return sampling_algo

class SimpleDisc2(SoSWrapp):
    """ Discipline used in Driver coupling of simple discipline output with driver subprocess input.
    """
    _maturity = 'Fake'
    DESC_IN = {'c_1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_1_dict': {'type': 'dict', 'unit': None, 'visibility': SoSWrapp.SHARED_VISIBILITY,
                            'namespace': 'ns_OptimSellar'}
               }

    DESC_OUT = {'out_simple2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ Discipline 2 execution
        """
        c_1 = self.get_sosdisc_inputs(['c_1'])
        y_1_dict = self.get_sosdisc_inputs('y_1_dict')

        out_simple2 = self.compute_out_simple2(c_1, y_1_dict)
        out = {'out_simple2': out_simple2}
        self.store_sos_outputs_values(out)

    @staticmethod
    def compute_out_simple2(c_1, y_1_dict):
        """ Computes the output of the simple discipline in array form
        """
        values_dict = list(y_1_dict.values())
        out = c_1 * std(values_dict[:-1])
        return out
