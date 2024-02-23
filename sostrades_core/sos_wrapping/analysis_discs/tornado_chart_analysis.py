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

import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.gather_discipline import GatherDiscipline
from sostrades_core.execution_engine.sample_generators.tornado_chart_analysis_sample_generator import TornadoChartAnalysisSampleGenerator
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

    OUTPUT_VARIATIONS_SUFFIX = '_variations'
    INPUT_COL = 'input'
    VARIATION_INPUT_COL = 'input_variation'
    VARIATION_OUTPUT_COL = 'output_variation'

    REFERENCE_SCENARIO_NAME = TornadoChartAnalysisSampleGenerator.REFERENCE_SCENARIO_NAME
    SCENARIO_NAME_COL = TornadoChartAnalysisSampleGenerator.SCENARIO_NAMES
    SCENARIO_VARIABLE_VARIATIONS = TornadoChartAnalysisSampleGenerator.SCENARIO_VARIABLE_VARIATIONS
    DESC_IN = {        GATHER_OUTPUTS: GATHER_OUTPUTS_DESC,
        SCENARIO_VARIABLE_VARIATIONS:{
            SoSWrapp.TYPE: 'dataframe'
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
                    dynamic_outputs[f'{output_name}{self.OUTPUT_VARIATIONS_SUFFIX}'] = {
                        SoSWrapp.TYPE: 'dataframe',
                        SoSWrapp.DATAFRAME_DESCRIPTOR:{self.INPUT_COL:('string',None, False),
                                                       self.VARIATION_INPUT_COL:('float',None, False),
                                                       self.VARIATION_OUTPUT_COL:('float',None, False)
                                                       }
                    }

            self.add_inputs(dynamic_inputs)
            self.add_outputs(dynamic_outputs)

    def run(self):
        dict_values = {}
        if len(self.selected_outputs_dict) > 0:
            
            
            variation_data_df = self.get_sosdisc_inputs(self.SCENARIO_VARIABLE_VARIATIONS)
            # get the list of inputs by removing the column of scenario_names
            variables_list = [col for col in variation_data_df.columns if col != self.SCENARIO_NAME_COL]

            for output_name in self.selected_outputs_dict.values():
                output_data = self.get_sosdisc_inputs(output_name)

                # create one output for each output_data
                output_variations_dict = {self.INPUT_COL:[], 
                                          self.VARIATION_INPUT_COL:[], 
                                          self.VARIATION_OUTPUT_COL:[]}
                
                # create a dataframe that contains scenario_name, inputs variation, output value per scenario
                variation_with_output_df = variation_data_df.merge(pd.DataFrame({self.SCENARIO_NAME_COL:output_data.keys(),
                                                                                 output_name:list(output_data.values())}), on=self.SCENARIO_NAME_COL )
                # get reference value
                reference_value = output_data[self.REFERENCE_SCENARIO_NAME]

                # compute the output variation for each input variation
                for input_name in variables_list:
                    # get the rows where the input variations is not 0%
                    input_variations_df = variation_with_output_df[variation_with_output_df[input_name]!= 0.0]

                    # build the variations results: the input, the variation on input, the variation on output
                    output_variations_dict[self.INPUT_COL].extend([input_name]*len(input_variations_df))
                    output_variations_dict[self.VARIATION_INPUT_COL].extend(list(input_variations_df[input_name].values))
                    # compute the variations
                    if reference_value != 0.0:
                        output_variations_dict[self.VARIATION_OUTPUT_COL].extend([100.0 * (output_value - reference_value)/reference_value 
                                                                             for output_value 
                                                                             in list(input_variations_df[output_name])])
                    else:
                        output_variations_dict[self.VARIATION_OUTPUT_COL].extend([100.0 * output_value 
                                                                             for output_value 
                                                                             in list(input_variations_df[output_name])])

                
                dict_values[f'{output_name}{self.OUTPUT_VARIATIONS_SUFFIX}'] = pd.DataFrame(output_variations_dict)
        
        self.store_sos_outputs_values(dict_values)
                