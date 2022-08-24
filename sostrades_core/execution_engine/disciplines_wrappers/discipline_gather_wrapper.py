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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class DisciplineGatherWrapper(SoSWrapp):
    """Wrapper for the gather discipline
    """
    _maturity = 'Fake'

    # def __init__(self):
    #     '''
    #     Constructor
    #     '''
    #     self.attributes = {}

    def run(self):
        '''
        Run function of the SoSGather : Collect variables to gather in a dict
        Assemble the output dictionary and store it in the DM
        '''
        # get gather builder
        cls_gather = self.attributes['cls_gather']

        new_values_dict = {}

        input_name = self.attributes['input_name']  # ac_name_list
        gather_inputs = self.get_sosdisc_inputs(in_dict=True, full_name_keys=True)
        builder_cls = self.attributes['builder_cls']
        gather_ns = self.attributes['gather_ns']
        for var_gather in self.attributes['var_gather']:
            gather_dict = {}
            if builder_cls == cls_gather:
                for name in gather_inputs[input_name]:
                    for sub_name, value in gather_inputs[f'{gather_ns}.{name}.{var_gather}'].items():
                        gather_dict[f'{name}.{sub_name}'] = value
                new_values_dict[f'{var_gather}'] = gather_dict
            else:
                for name in gather_inputs[input_name]:
                    if f'{gather_ns}.{name}.{var_gather}' in gather_inputs:
                        gather_dict[name] = gather_inputs[f'{gather_ns}.{name}.{var_gather}']
                new_values_dict[f'{var_gather}_dict'] = gather_dict

        self.store_sos_outputs_values(new_values_dict)
    # -- Configure handling
