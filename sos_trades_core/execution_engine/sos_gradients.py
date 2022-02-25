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
import numpy as np
from pandas.core.frame import DataFrame

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_eval import SoSEval
from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient


class SoSGradients(SoSEval):
    '''SoSGradients class
    '''


    # ontology information
    _ontology_data = {
        'label': 'Core Gradients Model',
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
        '''
        Constructor
        '''
        super(SoSGradients, self).__init__(sos_name, ee, cls_builder)
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Gradients')

    def set_x0(self, x0):
        '''
        Set initial values for input values decided in the evaluation
        '''
        x_dict = {}
        for i, x_id in enumerate(self.eval_in_list):
            x_dict[x_id] = x0[i]

        self.dm.set_values_from_dict(x_dict)

    def launch_gradient_analysis(self, grad_method):
        eps = 1.e-4
        if grad_method == 'Complex Step':
            grad_method_number = 1j
        elif grad_method == '1st order FD':
            grad_method_number = 1
        elif grad_method == '2nd order FD':
            grad_method_number = 2
        else:
            raise Exception(
                'Wrong gradient method, methods available are "Complex Step", "1st order FD" and "2nd order FD"')

        grad_eval = FDGradient(
            grad_method_number, self.sample_evaluation, fd_step=eps)
        grad_eval.set_multi_proc(False)

        x0 = self.get_x0()

        outputs_grad = grad_eval.grad_f(x0)

        self.set_x0(x0)

        outgrad_final_dict = self.reconstruct_output_results(outputs_grad)
        return outgrad_final_dict

    def compute_form_outputs(self, gradient_outputs, variation_list):
        '''
        Compute the FORM outputs with gradient results
        '''
        form_dict = {}
        gradient_output_relative = None
        for variation in variation_list:
            form_dict_var = {}
            form_dict_relative = {}
            for input_sens in self.eval_in_list:
                # We multiply by Dx= N % * x
                dx = variation / 100.0 * self.dm.get_value(input_sens)

                for output_sens in self.eval_out_list:
                    real_value = self.dm.get_value(output_sens)
                    output_name = f'{output_sens} vs {input_sens}'
                    gradient_output_relative = copy.deepcopy(
                        gradient_outputs[output_name])
                    if type(gradient_outputs[output_name]) is dict:
                        for key in gradient_outputs[output_name]:
                            if type(gradient_outputs[output_name][key]) is dict:
                                for sub_key in gradient_outputs[output_name][key]:
                                    gradient_outputs[output_name][key][sub_key] *= dx
                                    if real_value[key][sub_key] != 0:
                                        gradient_output_relative[key][sub_key] = 100.0 * gradient_outputs[
                                            output_name][key][sub_key] / real_value[key][sub_key]
                            elif type(gradient_outputs[output_name][key]) is DataFrame:
                                gradient_outputs[output_name][key] *= dx
                                gradient_output_relative[key] = 100.0 * gradient_outputs[output_name][key].divide(
                                    real_value[key], fill_value=0.0)
                                gradient_output_relative[key] = gradient_output_relative[key].replace(
                                    [np.inf, -np.inf], np.nan)
                                gradient_output_relative[key] = gradient_output_relative[key].fillna(
                                    0.0)
                            else:
                                gradient_outputs[output_name][key] *= dx
                                if real_value[key] != 0:
                                    gradient_output_relative[key] = 100.0 * gradient_outputs[output_name][key] / \
                                        real_value[key]
                    elif type(gradient_outputs[output_name]) is DataFrame:
                        gradient_outputs[output_name] *= dx
                        gradient_output_relative = 100.0 * gradient_outputs[output_name].divide(
                            real_value, fill_value=0.0)
                        gradient_output_relative = gradient_output_relative.replace(
                            [np.inf, -np.inf], np.nan)
                        gradient_output_relative = gradient_output_relative.fillna(
                            0.0)
                    else:
                        gradient_outputs[output_name] *= dx
                        if real_value != 0:
                            gradient_output_relative = 100.0 * \
                                gradient_outputs[output_name] / real_value
                    form_dict_var[output_name] = gradient_outputs[output_name]
                    form_dict_relative[output_name] = gradient_output_relative

            form_dict[f'{variation}%_relative'] = form_dict_relative
            form_dict[f'{variation}%'] = form_dict_var

        return form_dict
