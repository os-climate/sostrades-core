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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''

import logging
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline
from sostrades_core.execution_engine.sample_generators.sensitivity_analysis_sample_generator import SensitivityAnalysisSampleGenerator
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.tools.gather.gather_tool import gather_selected_outputs

class TornadoChartAnalysis(SoSWrapp):
    """
    Generic Uncertainty Quantification class
    """

    # ontology information
    _ontology_data = {
        'label': 'Tornado chart analysis Model',
        SoSWrapp.TYPE: 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fa-solid fa-chart-area',
        'version': '',
    }
    GATHER_OUTPUTS = GatherDiscipline.GATHER_OUTPUTS
    GATHER_OUTPUTS_DESC = GatherDiscipline.EVAL_OUTPUTS_DESC.copy()
    GATHER_OUTPUTS_DESC[SoSWrapp.NAMESPACE] = ProxySampleGenerator.NS_SAMPLING
    GATHER_OUTPUTS_DESC[SoSWrapp.VISIBILITY] = SoSWrapp.SHARED_VISIBILITY

    DESC_IN = {
        GATHER_OUTPUTS: GATHER_OUTPUTS_DESC,
        SensitivityAnalysisSampleGenerator.SCENARIO_VARIABLE_VARIATIONS:{
            SoSWrapp.TYPE: 'dataframe'
        }

    }
    DESC_OUT = {
        'outputs_variations': {
            SoSWrapp.TYPE: 'dataframe',
            SoSWrapp.UNIT: '%'
        }
    }
    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.selected_outputs_dict = {}

    def setup_sos_disciplines(self):
        """setup sos disciplines"""
        data_in = self.get_data_in()
        if data_in != {}:
            # Add the outputs of the driver eval selected in gather_outputs in input of the disc
            dynamic_outputs = {}
            dynamic_inputs = {}
            if self.GATHER_OUTPUTS in data_in:
                gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
                # get only variables that are selected
                self.selected_outputs_dict = gather_selected_outputs(gather_outputs, GatherDiscipline.GATHER_SUFFIX)
                # add dynamic input for each output name
                for output_name in self.selected_outputs_dict.values():
                    dynamic_inputs[output_name] = {
                        SoSWrapp.TYPE: 'dict',
                        SoSWrapp.NAMESPACE: ProxySampleGenerator.NS_SAMPLING,
                        SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY
                    }

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def run(self):
        output_data = []
        if len(self.selected_outputs_dict) > 0:
            output_data = self.get_sosdisc_inputs(self.selected_outputs_dict.values())
            print(output_data)