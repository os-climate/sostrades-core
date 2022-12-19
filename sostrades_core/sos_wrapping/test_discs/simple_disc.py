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

class SimpleDisc(SoSWrapp):
    """ Discipline used in Driver coupling of simple discipline output with driver subprocess input.
    """
    _maturity = 'Fake'
    DESC_IN = {'z_in': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_z'}}

    DESC_OUT = {'z': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_z'}}

    def run(self):
        """ Discipline 1 execution
        """
        z_in = self.get_sosdisc_inputs(['z_in'])
        z = self.compute_z(z_in)
        z_out = {'z': z}
        self.store_sos_outputs_values(z_out)

    @staticmethod
    def compute_z(z):
        """ Computes the output of the simple discipline in array form
        """
        out = z * pow(2, -1)
        return out

