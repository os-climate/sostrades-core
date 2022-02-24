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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import copy
from pandas.core.frame import DataFrame

from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.api import get_sos_logger


class SoSSensitivity(SoSEval):
    '''**SoSSensitivity**
    COmpute the Df for a given Dx with specified F and x
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Sensitivity Model',
        'type': 'Official',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    def __init__(self, sos_name, ee, cls_builder):
        super(SoSSensitivity, self).__init__(
            sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Sensitivity')
        self.variation_list = []
        self.n_samples = 0

    def generate_samples(self, variation_list):
        '''
        Generate the samples needed to compute the sensitivity 1 sample without variation and 2 samples by input values
        '''
        x_samples = []
        input_in_samples = []
        variation_samples = []
        self.n_samples = 0

        for variation in variation_list:
            for input_in_sample in self.eval_in_list:
                x_pos_sample = []
                x_neg_sample = []
# Create the sample with the  Dx variatiob
                for input_in in self.eval_in_list:
                    if input_in == input_in_sample:
                        variation_input = self.dm.get_value(
                            input_in) * (1.0 + variation / 100.0)
                        x_pos_sample.append(variation_input)
                    else:
                        input_value = self.dm.get_value(
                            input_in)
                        x_pos_sample.append(input_value)
                x_samples.append(x_pos_sample)
                input_in_samples.append(input_in_sample)
                variation_samples.append(variation)
                self.n_samples += 1
# Create the sample with the - Dx variation
                for input_in in self.eval_in_list:
                    if input_in == input_in_sample:
                        variation_input = self.dm.get_value(
                            input_in) * (1.0 - variation / 100.0)
                        x_neg_sample.append(variation_input)
                    else:
                        input_value = self.dm.get_value(
                            input_in)
                        x_neg_sample.append(input_value)
                self.n_samples += 1
                x_samples.append(x_neg_sample)
                input_in_samples.append(input_in_sample)
                variation_samples.append(-variation)
        x_sample_novar = [self.dm.get_value(
            input_in) for input_in in self.eval_in_list]
        x_samples.append(x_sample_novar)
        self.n_samples += 1
        input_in_samples.append('NOVAR')
        variation_samples.append(0.0)

        return x_samples, input_in_samples, variation_samples

    def launch_sensitivity_analysis(self, variation_list):
        '''
        Launch sensitivity analysis by computing the function to evaluate on each sample generated
        '''
        self.variation_list = variation_list
        output_dict = {}

        x_samples, input_in_samples, variation_samples = self.generate_samples(
            variation_list)
        for i, x_sample in enumerate(x_samples):

            output_eval = copy.deepcopy(
                self.sample_evaluation(x_sample, convert_to_array=False))

            for j, output_sens in enumerate(self.eval_out_list):
                if variation_samples[i] == 0.0:
                    output_name = f'novariation_{output_sens}'
                else:
                    output_name = f'{variation_samples[i]}percent_{output_sens} vs {input_in_samples[i]}'

                output_dict[output_name] = output_eval[j]

        return output_dict

    def compute_df(self, sensitivity_outputs):
        '''
        Compute the variation (absolute and relative) between two results of the sensitivity in order to compute sensitivity output
        '''
        sens_dict = {}
        for variation in self.variation_list:
            sens_dict_minus = {}
            sens_dict_plus = {}
            sens_dict_minus_relative = {}
            sens_dict_plus_relative = {}
            for input_sens in self.eval_in_list:
                for output_sens in self.eval_out_list:
                    output_name = f'{output_sens} vs {input_sens}'
                    output_value = sensitivity_outputs[
                        f'novariation_{output_sens}']
                    sens_minus = copy.deepcopy(output_value)
                    sens_minus_relative = copy.deepcopy(output_value)
                    sens_plus = copy.deepcopy(output_value)
                    sens_plus_relative = copy.deepcopy(output_value)
                    if isinstance(output_value, dict):
                        for key in output_value:

                            if isinstance(output_value[key], dict):
                                for sub_key in output_value[key]:
                                    if type(output_value[key][sub_key]) is str or type(sensitivity_outputs[f'{variation}percent_' + output_name][key][sub_key]) is str or type(sensitivity_outputs[f'-{variation}percent_' + output_name][key][sub_key]) is str:
                                        continue
                                    sens_minus[key][sub_key] = sensitivity_outputs[
                                        f'-{variation}percent_' + output_name][key][sub_key] - output_value[key][sub_key]
                                    sens_plus[key][sub_key] = sensitivity_outputs[
                                        f'{variation}percent_' + output_name][key][sub_key] - output_value[key][sub_key]
                                    if output_value[key][sub_key] != 0:
                                        sens_minus_relative[key][sub_key] = 100.0 * sens_minus[key][sub_key] / \
                                            output_value[key][sub_key]
                                        sens_plus_relative[key][sub_key] = 100.0 * sens_plus[key][sub_key] / \
                                            output_value[key][sub_key]

                            elif isinstance(output_value[key], DataFrame):
                                sens_minus[key] = sensitivity_outputs[
                                    f'-{variation}percent_' + output_name][key] - output_value[key]
                                sens_minus_relative[key] = 100.0 * sens_minus[key].divide(
                                    output_value[key], fill_value=0.0)
                                sens_plus[key] = sensitivity_outputs[
                                    f'{variation}percent_' + output_name][key] - output_value[key]
                                sens_plus_relative[key] = 100.0 * sens_plus[key].divide(
                                    output_value[key], fill_value=0.0)
                            # Should handle other element type (array
                            # and float) :
                            else:
                                if type(output_value[key]) is str or type(sensitivity_outputs[f'{variation}percent_' + output_name][key]) is str or type(sensitivity_outputs[f'-{variation}percent_' + output_name][key]) is str:
                                    continue
                                sens_minus[key] = sensitivity_outputs[
                                    f'-{variation}percent_' + output_name][key] - output_value[key]
                                sens_plus[key] = sensitivity_outputs[
                                    f'{variation}percent_' + output_name][key] - output_value[key]

                                if output_value[key] != 0:
                                    sens_minus_relative[key] = 100.0 * sens_minus[key] / \
                                        output_value[key]
                                    sens_plus_relative[key] = 100.0 * sens_plus[key] / \
                                        output_value[key]
                    elif isinstance(output_value, DataFrame):
                        sens_minus = sensitivity_outputs[
                            f'-{variation}percent_' + output_name] - output_value
                        sens_minus_relative = 100.0 * sens_minus.divide(
                            output_value, fill_value=0.0)
                        sens_plus = sensitivity_outputs[
                            f'{variation}percent_' + output_name] - output_value
                        sens_plus_relative = 100.0 * sens_plus.divide(
                            output_value, fill_value=0.0)
                    else:
                        if type(output_value) is str or output_value is None:
                            continue
                        sens_minus = sensitivity_outputs[
                            f'-{variation}percent_' + output_name] - output_value
                        sens_plus = sensitivity_outputs[
                            f'{variation}percent_' + output_name] - output_value
                        if output_value != 0:
                            sens_minus_relative = 100.0 * sens_minus / output_value
                            sens_plus_relative = 100.0 * sens_plus / output_value
                    sens_dict_plus[output_name] = sens_plus
                    sens_dict_plus_relative[output_name] = sens_plus_relative
                    sens_dict_minus[output_name] = sens_minus
                    sens_dict_minus_relative[output_name] = sens_minus_relative

            sens_dict[f'+{variation}%'] = sens_dict_plus
            sens_dict[f'-{variation}%'] = sens_dict_minus
            sens_dict[f'+{variation}%_relative'] = sens_dict_plus_relative
            sens_dict[f'-{variation}%_relative'] = sens_dict_minus_relative
        return sens_dict
