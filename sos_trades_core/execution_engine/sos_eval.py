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
import numpy as np
from pandas.core.frame import DataFrame

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.sos_discipline_builder import SoSDisciplineBuilder
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.ns_manager import NS_SEP
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class SoSEval(SoSDisciplineBuilder):
    '''
        SOSEval class which creates a sub process to evaluate
        with different methods (Gradient,FORM,Sensitivity ANalysis, DOE, ...)
    '''
    DESC_IN = {
        'eval_inputs': {'type': 'string_list', 'unit': None, 'structuring': True},
        'eval_outputs': {'type': 'string_list', 'unit': None, 'structuring': True},
    }

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        super(SoSEval, self).__init__(sos_name, ee)
        self.eval_in_base_list = None
        self.eval_in_list = None
        self.eval_out_base_list = None
        self.eval_out_list = None
        # Needed to reconstruct objects from flatten list
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Eval')
        self.cls_builder = cls_builder
        self.eval_coupling = None
        # Create the eval_coupling associated to the eval
        self.__reset_coupling()

    def set_eval_in_out_lists(self, in_list, out_list):
        '''
        Set the evaluation variable list (in and out) present in the DM
        which fits with the eval_in_base_list filled in the usecase or by the user
        '''
        self.eval_in_base_list = in_list
        self.eval_out_base_list = out_list
        self.eval_in_list = []
        for v_id in in_list:
            full_id_list = self.dm.get_all_namespaces_from_var_name(v_id)
            for full_id in full_id_list:
                if self.dm.data_dict[self.dm.data_id_map[full_id]]['io_type'] == 'in':
                    self.eval_in_list.append(full_id)

        self.eval_out_list = []
        for v_id in out_list:
            full_id_list = self.dm.get_all_namespaces_from_var_name(v_id)
            for full_id in full_id_list:
                if self.dm.data_dict[self.dm.data_id_map[full_id]]['io_type'] == 'out':
                    self.eval_out_list.append(full_id)

    def __reset_coupling(self):
        '''
        Create the builder sub_coupling and build it with the process builder_list in argument 
        '''
        self.eval_coupling_builder = self.ee.factory.create_builder_coupling(
            self.sos_name)
        if not isinstance(self.cls_builder, list):
            self.cls_builder = [self.cls_builder]
        self.eval_coupling_builder.set_builder_info(
            'cls_builder', self.cls_builder)

        self.eval_coupling = self.eval_coupling_builder.build()
        # eval_coupling must not be run by execution_engine
        # (only during eval_run method)
        self.eval_coupling.no_run = True

        self.ee.factory.add_discipline(self.eval_coupling)

    def get_eval_coupling(self):
        '''
        Return the eval_coupling of the SoSEval
        '''
        return self.eval_coupling

    def fill_possible_values(self, disc):
        '''
            Fill possible values lists for eval inputs and outputs
            an input variable must be a float coming from a data_in of a discipline in all the process
            and not a default variable
            an output variable must be any data from a data_out discipline
        '''
        poss_in_values = []
        poss_out_values = []
        for data_in_key in disc._data_in.keys():
            is_float = disc._data_in[data_in_key][self.TYPE] == 'float'
            in_coupling_numerical = data_in_key in list(SoSCoupling.DEFAULT_NUMERICAL_PARAM.keys()) + \
                list(SoSCoupling.DEFAULT_NUMERICAL_PARAM_OUT_OF_INIT.keys())
            if is_float and not in_coupling_numerical:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                poss_in_values.append(data_in_key)
        for data_out_key in disc._data_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            poss_out_values.append(data_out_key.split(NS_SEP)[-1])

        return poss_in_values, poss_out_values

    def add_discipline(self, disc):
        '''
        Add a discipline directly to the eval_coupling
        '''
        self.eval_coupling.add_discipline(disc)

    def build(self):
        '''
        The build of the SoSEval calls directly the build of the rootcoupling
        '''
        self.eval_coupling.build()

    def configure(self):
        '''
        Configure the SoSEval and the eval_coupling of the SoSEval + set eval possible values for the GUI 
        '''
        if self._data_in == {} or self.eval_coupling.is_configured():
            # Call standard configure methods to set the process discipline
            # tree
            SoSDiscipline.configure(self)

            # Extract variables for eval analysis
            if self.eval_coupling.sos_disciplines is not None and len(self.eval_coupling.sos_disciplines) > 0:
                self.set_eval_possible_values()

    def is_configured(self):
        '''
        Return False if discipline is not configured or structuring variables have changed or eval coupling is not configured
        '''
        return SoSDiscipline.is_configured(self) and self.eval_coupling.is_configured()

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        analyzed_disc = self.eval_coupling
        possible_in_values, possible_out_values = self.fill_possible_values(
            analyzed_disc)

        possible_in_values, possible_out_values = self.find_possible_values(
            analyzed_disc, possible_in_values, possible_out_values)

        # Take only unique values in the list
        possible_in_values = list(set(possible_in_values))
        possible_out_values = list(set(possible_out_values))

        # Fill the possible_values of eval_inputs
        self.dm.set_data(f'{self.get_disc_full_name()}.eval_inputs',
                         self.POSSIBLE_VALUES, possible_in_values)
        self.dm.set_data(f'{self.get_disc_full_name()}.eval_outputs',
                         self.POSSIBLE_VALUES, possible_out_values)

    def find_possible_values(
            self, disc, possible_in_values, possible_out_values):
        '''
            Run through all disciplines and sublevels
            to find possible values for eval_inputs and eval_outputs
        '''
        if len(disc.sos_disciplines) != 0:
            for disc in disc.sos_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    disc, possible_in_values, possible_out_values)

        return possible_in_values, possible_out_values

    def get_x0(self):
        '''
        Get initial values for input values decided in the evaluation
        '''
        x0 = []
        for x_id in self.eval_in_list:
            x_val = self.dm.get_value(x_id)
            x0.append(x_val)
        return np.array(x0)

    def FDeval_func(self, x, convert_to_array=True):
        '''
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays. 
        '''
        # -- need to clear cash to avoir GEMS preventing execution when using disciplinary variables
        eval_process = self.get_eval_coupling()
        eval_process.clear_cache()
        values_dict = {}
        for i, x_id in enumerate(self.eval_in_list):
            values_dict[x_id] = x[i]
        # configure eval_process with values_dict inputs
        eval_process.ee.load_study_from_input_dict(
            values_dict, update_status_configure=False)
        eval_process.execute()

        if convert_to_array:
            out_values = self.convert_output_results_toarray()
        else:
            out_values = []
            for y_id in self.eval_out_list:
                y_val = self.dm.get_value(y_id)
                out_values.append(y_val)

        return out_values

    def convert_output_results_toarray(self):
        '''
        COnvert toutput results into array in order to apply FDGradient on it for example
        '''
        out_values = []
        self.eval_out_type = []
        self.eval_out_list_size = []
        for y_id in self.eval_out_list:

            y_val = self.dm.get_value(y_id)
            self.eval_out_type.append(type(y_val))
            # Need a flatten list for the eval computation if val is dict
            if type(y_val) in [dict, DataFrame]:
                val_dict = {y_id: y_val}
                dict_flatten = self._convert_new_type_into_array(
                    val_dict)
                y_val = dict_flatten[y_id].tolist()

            else:
                y_val = [y_val]
            self.eval_out_list_size.append(len(y_val))
            out_values.extend(y_val)

        return np.array(out_values)

    def reconstruct_output_results(self, outputs_eval):
        '''
        Reconstruct the metadata saved earlier to get same object in output
        instead of a flatten list
        '''
        outeval_final_dict = {}
        for j, key_in in enumerate(self.eval_in_list):
            outeval_dict = {}
            old_size = 0
            for i, key in enumerate(self.eval_out_list):

                output_eval_key = outputs_eval[old_size:old_size +
                                               self.eval_out_list_size[i]]
                old_size = self.eval_out_list_size[i]

                if self.eval_out_type[i] in [dict, DataFrame]:
                    outeval_dict[key] = np.array([
                        sublist[j] for sublist in output_eval_key])
                else:
                    outeval_dict[key] = output_eval_key[0][j]

            outeval_dict = self._convert_array_into_new_type(outeval_dict)
            outeval_base_dict = {f'{key_out} vs {key_in}': value for key_out, value in zip(
                self.eval_out_list, outeval_dict.values())}
            outeval_final_dict.update(outeval_base_dict)

        return outeval_final_dict

    def run(self):
        '''
        Overloaded SoSDiscpline method
        '''
        # set eval_coupling no_run flag to False to run eval_copling into
        # eval_run method
        self.eval_coupling.no_run = False
        self.eval_run()
        # reset eval_coupling no_run flag to True
        self.eval_coupling.no_run = True

    def eval_run(self):
        '''
            SoSEval run method, to be overladed by inherited classes
        '''
        pass
