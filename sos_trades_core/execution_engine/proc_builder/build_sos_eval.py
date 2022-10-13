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
import platform
import pandas as pd
import re

from tqdm import tqdm
import time

from gemseo.core.parallel_execution import ParallelExecution
from sos_trades_core.tools.base_functions.compute_len import compute_len

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np
from pandas.core.frame import DataFrame

from sos_trades_core.api import get_sos_logger
from sos_trades_core.execution_engine.ns_manager import NS_SEP
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.proc_builder.sos_add_subproc_to_driver import AddSubProcToDriver


class BuildBuildSoSEvalException(Exception):
    pass


class BuildSoSEval(AddSubProcToDriver):
    '''
        BuildSoSEval class which creates a sub process to evaluate
        with different methods (Gradient,FORM,Sensitivity ANalysis, DOE, ...)
    '''

    # ontology information
    _ontology_data = {
        'label': 'Core Eval Model',
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
    MULTIPLIER_PARTICULE = '__MULTIPLIER__'
    DESC_IN = {
        'eval_inputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'eval_outputs': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'unit': None, 'structuring': True},
        'n_processes': {'type': 'int', 'numerical': True, 'default': 1},
        'wait_time_between_fork': {'type': 'float', 'numerical': True, 'default': 0.0},

    }

    def __init__(self, sos_name, ee, cls_builder, associated_namespaces=[]):
        '''
        Constructor
        '''
        super(BuildSoSEval, self).__init__(sos_name, ee,
                                           associated_namespaces=associated_namespaces)
        self.eval_in_base_list = None
        self.eval_in_list = None
        self.eval_out_base_list = None
        self.eval_out_list = None
        # Needed to reconstruct objects from flatten list
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Eval')
        self.cls_builder = cls_builder
        # Create the eval process builder associated to BuildSoSEval
        self.driver_process_builder = self._set_driver_process_builder()
        self.eval_process_disc = None

    def _set_driver_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) > 1 or not self.cls_builder[0]._is_executable:
            # if eval process is a list of builders or a non executable builder,
            # then we build a coupling containing the eval porcess
            disc_builder = self.ee.factory.create_builder_coupling(
                self.sos_name)
            disc_builder.set_builder_info('cls_builder', self.cls_builder)
        else:
            # else return the single builder
            disc_builder = self.cls_builder[0]

        return disc_builder

    def set_eval_in_out_lists(self, in_list, out_list, inside_evaluator=False):
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
                if not inside_evaluator:
                    self.eval_in_list.append(full_id)
                else:
                    if full_id.startswith(self.get_disc_full_name()):
                        self.eval_in_list.append(full_id)
        self.eval_out_list = []
        for v_id in out_list:
            full_id_list = self.dm.get_all_namespaces_from_var_name(v_id)
            for full_id in full_id_list:
                self.eval_out_list.append(full_id)

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
            # structuring variables are excluded from possible values!!!
            is_structuring = disc._data_in[data_in_key].get(
                self.STRUCTURING, False)
            in_coupling_numerical = data_in_key in list(
                SoSCoupling.DESC_IN.keys())
            full_id = self.dm.get_all_namespaces_from_var_name(data_in_key)[0]
            is_in_type = self.dm.data_dict[self.dm.data_id_map[full_id]
                                           ]['io_type'] == 'in'
            if is_float and is_in_type and not in_coupling_numerical and not is_structuring:
                # Caution ! This won't work for variables with points in name
                # as for ac_model
                poss_in_values.append(data_in_key)
        for data_out_key in disc._data_out.keys():
            # Caution ! This won't work for variables with points in name
            # as for ac_model
            poss_out_values.append(data_out_key.split(NS_SEP)[-1])

        return poss_in_values, poss_out_values

    def build(self):
        '''
        Method copied from SoSCoupling: build and store disciplines in sos_disciplines
        '''
        # set current_discipline to self to build and store eval process in the
        # children of BuildSoSEval
        old_current_discipline = self.ee.factory.current_discipline
        self.ee.factory.current_discipline = self

        # if we want to build an eval coupling containing eval process,
        # we have to remove BuildSoSEval name in current_ns to build eval coupling
        # at the same node as BuildSoSEval
        if len(self.cls_builder) == 0:  # added condition for proc build
            pass
        elif self.cls_builder[0] != self.driver_process_builder:
            current_ns = self.ee.ns_manager.current_disc_ns
            self.ee.ns_manager.set_current_disc_ns(
                current_ns.split(f'.{self.sos_name}')[0])
            # build coupling containing eval process
            self.eval_process_disc = self.driver_process_builder.build()
            # store coupling in the children of BuildSoSEval
            if self.eval_process_disc not in self.sos_disciplines:
                self.ee.factory.add_discipline(self.eval_process_disc)
            # reset current_ns after build
            self.ee.ns_manager.set_current_disc_ns(current_ns)
        else:
            # build and store eval process in the children of BuildSoSEval
            self.eval_process_disc = self.driver_process_builder.build()
            if self.eval_process_disc not in self.sos_disciplines:
                self.ee.factory.add_discipline(self.eval_process_disc)

        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
        # level coupling and we do not have to restore the current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    def get_disciplines_to_configure(self):
        '''
        Method copied from SoSCoupling: get sub disciplines list to configure
        '''
        disc_to_configure = []
        for disc in self.sos_disciplines:
            if not disc.is_configured():
                disc_to_configure.append(disc)
        return disc_to_configure

    def configure(self):
        '''
        Configure the BuildSoSEval and its children sos_disciplines + set eval possible values for the GUI 
        '''
        # configure eval process stored in children
        for disc in self.get_disciplines_to_configure():
            disc.configure()
            # Now that we use local_data as output of an execute a single
            # discipline needs to have grammar configured (or the filter after
            # execute will delete output results from local_data
            # If it is a coupling the grammar is already configured
            if not disc.is_sos_coupling:
                disc.update_gems_grammar_with_data_io()

        if self._data_in == {} or (self.get_disciplines_to_configure() == [] and len(self.sos_disciplines) != 0) or len(self.cls_builder) == 0:
            # Explanation:
            # 1. self._data_in == {} : if the discipline as no input key it should have and so need to be configured
            # 2. Added condition compared to SoSDiscipline(as sub_discipline or associated sub_process builder)
            # 2.1 (self.get_disciplines_to_configure() == [] and len(self.sos_disciplines) != 0) : sub_discipline(s) exist(s) but all configured
            # 2.2 len(self.cls_builder) == 0 No yet provided builder but we however need to configure (as in 2.1 when we have sub_disciplines which no need to be configured)
            # Remark: condition "(   and len(self.sos_disciplines) != 0) or len(self.cls_builder) == 0" added for proc build
            #
            # Call standard configure methods to set the process discipline
            # tree
            SoSDiscipline.configure(self)

            # Extract variables for eval analysis
            if len(self.sos_disciplines) > 0:
                self.set_eval_possible_values()

        if len(self.get_disciplines_to_configure()) == 0:
            self.set_children_cache_inputs()

    def is_configured(self):
        '''
        Return False if discipline is not configured or structuring variables have changed or children are not all configured
        '''
        return SoSDiscipline.is_configured(self) and (
            (self.get_disciplines_to_configure() == [] and len(self.sos_disciplines) != 0) or len(
                self.cls_builder) == 0)
        # Explanation:
        # 1. SoSDiscipline.is_configured(self) : as in discipline (i.e. conf status of Eval = True and no change in structuring variables of Eval
        # 2. Added condition compared to SoSDiscipline :
        # 2.1 (self.get_disciplines_to_configure() == [] and len(self.sos_disciplines) != 0) : sub_discipline exist but all configured
        # 2.2 len(self.cls_builder) == 0 No yet provided builder and so Is configured is True
        # Remark: condition "and len(self.sos_disciplines) != 0) or
        # len(self.cls_builder) == 0)" added for proc build

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # the eval process to analyse is stored as the only child of BuildSoSEval
        # (coupling chain of the eval process or single discipline)
        analyzed_disc = self.sos_disciplines[0]

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
            for sub_disc in disc.sos_disciplines:
                sub_in_values, sub_out_values = self.fill_possible_values(
                    sub_disc)
                possible_in_values.extend(sub_in_values)
                possible_out_values.extend(sub_out_values)
                self.find_possible_values(
                    sub_disc, possible_in_values, possible_out_values)

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

    def evaluation(self, x, scenario_name=None, convert_to_array=True, completed_eval_in_list=None):
        '''
        Call to the function to evaluate with x : values which are modified by the evaluator (only input values with a delta)
        Only these values are modified in the dm. Then the eval_process is executed and output values are convert into arrays. 
        '''
        # -- need to clear cash to avoir GEMS preventing execution when using disciplinary variables
        self.clear_cache()

        values_dict = {}
        eval_in = self.eval_in_list
        if completed_eval_in_list is not None:
            eval_in = completed_eval_in_list
        for i, x_id in enumerate(eval_in):
            values_dict[x_id] = x[i]

        # Because we use set_data instead of load_data_from_inputs_dict it isn't possible
        # to run  BuildSoSEval on a structuring variable. Therefore structuring variables are
        # excluded from eval possible values
        # set values_dict in the data manager to execute the sub process
        self.ee.dm.set_values_from_dict(values_dict)

        # execute eval process stored in children
        if len(self.sos_disciplines) > 1:
            # the only child must be a coupling or a single discipline
            raise BuildSoSEvalException(
                f'BuildSoSEval discipline has more than one child discipline')
        else:
            local_data = self.sos_disciplines[0].execute()

        out_local_data = {key: value for key,
                          value in local_data.items() if key in self.eval_out_list}

        # needed for gradient computation
        self.update_dm_with_local_data(out_local_data)

        if convert_to_array:
            out_local_data_converted = self._convert_new_type_into_array(
                out_local_data)
            out_values = np.concatenate(
                list(out_local_data_converted.values())).ravel()
        else:
            out_values = []
            # get back out_local_data is not enough because some variables
            # could be filtered for unsupported type for gemseo
            for y_id in self.eval_out_list:
                y_val = self.dm.get_value(y_id)
                out_values.append(y_val)

        return out_values

    def samples_evaluation(self, samples, convert_to_array=True, completed_eval_in_list=None):
        '''This function executes a parallel execution of the function sample_evaluation
        over a list a samples. Depending on the numerical parameter n_processes it loops
        on a sequential or parallel way over the list of samples to evaluate
        '''
        evaluation_output = {}
        n_processes = self.get_sosdisc_inputs('n_processes')
        wait_time_between_samples = self.get_sosdisc_inputs(
            'wait_time_between_fork')
        if platform.system() == 'Windows' or n_processes == 1:
            if n_processes != 1:
                self.logger.warning(
                    "multiprocessing is not possible on Windows")
                n_processes = 1
            self.logger.info("running sos eval in sequential")

            for i in tqdm(range(len(samples)), ncols=100, position=0):
                time.sleep(0.1)
                self.logger.info(f'   Scenario_{str(i + 1)} is running.')
                x = samples[i]
                scenario_name = "scenario_" + str(i + 1)
                evaluation_output[scenario_name] = x, self.evaluation(
                    x, scenario_name, convert_to_array, completed_eval_in_list)
            return evaluation_output

        if n_processes > 1:
            self.logger.info(
                "Running SOS EVAL in parallel on n_processes = %s", str(n_processes))

            # Create the parallel execution object. The function we want to
            # parallelize is the sample_evaluation
            def sample_evaluator(sample_to_evaluate):
                """Evaluate a sample
                """
                return self.evaluation(sample_to_evaluate, convert_to_array=False, completed_eval_in_list=completed_eval_in_list)

            parallel = ParallelExecution(sample_evaluator, n_processes=n_processes,
                                         wait_time_between_fork=wait_time_between_samples)

            # Define a callback function to store the samples on the fly
            # during the parallel execution
            def store_callback(
                    index,  # type: int
                    outputs,  # type: DOELibraryOutputType
            ):  # type: (...) -> None
                """Store the outputs in dedicated dictionnary:
                - Here the dictionnary key is the sample evaluated and the value is the evaluation output
                Args:
                    index: The sample index.
                    outputs: The outputs of the parallel execution.
                """
                scenario_name = "scenario_" + str(index + 1)
                evaluation_output[scenario_name] = (samples[index], outputs)
                self.logger.info(
                    f'{scenario_name} has been run. computation progress: {int(((len(evaluation_output)) / len(samples)) * 100)}% done.')
                time.sleep(0.05)

            try:
                # execute all the scenarios (except the reference scenario)  in
                # parallel
                parallel.execute(samples[0:-1], exec_callback=store_callback)
                # execute the reference scenario in a sequential way so that
                # sostrades objects are updated
                scenario_name = "scenario_" + str(len(samples))
                evaluation_output[scenario_name] = samples[-1], self.evaluation(
                    samples[-1], scenario_name, convert_to_array, completed_eval_in_list)
                self.sos_disciplines[0]._update_status_recursive(
                    self.STATUS_DONE)
                dict_to_return = {}
                for (scenario_name, sample_value) in sorted(evaluation_output.items(),
                                                            key=lambda scenario: int(
                                                                scenario[0].split("scenario_")[1])):
                    dict_to_return[scenario_name] = sample_value
                return dict_to_return

            except:
                self.sos_disciplines[0]._update_status_recursive(
                    self.STATUS_FAILED)

    def apply_muliplier(self, multiplier_name, multiplier_value, var_to_update):
        col_index = multiplier_name.split(self.MULTIPLIER_PARTICULE)[
            0].split('@')[1]
        if any(char.isdigit() for char in col_index):
            col_index = re.findall(r'\d+', col_index)[0]
            cols_list = var_to_update.columns.to_list()
            key = cols_list[int(col_index)]
            var_to_update[key] = multiplier_value * var_to_update[key]
        else:
            if isinstance(var_to_update, dict):
                float_cols_ids_list = [dict_keys for dict_keys in var_to_update if isinstance(
                    var_to_update[dict_keys], float)]
            elif isinstance(var_to_update, pd.DataFrame):
                float_cols_ids_list = [
                    df_keys for df_keys in var_to_update if var_to_update[df_keys].dtype == 'float']
            for key in float_cols_ids_list:
                var_to_update[key] = multiplier_value * var_to_update[key]
        return var_to_update

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
                eval_out_size = compute_len(
                    self.eval_process_disc.local_data[key])
                output_eval_key = outputs_eval[old_size:old_size +
                                               eval_out_size]
                old_size = eval_out_size
                type_sos = self.dm.get_data(key, 'type')
                if type_sos in ['dict', 'dataframe']:
                    outeval_dict[key] = np.array([
                        sublist[j] for sublist in output_eval_key])
                else:
                    outeval_dict[key] = output_eval_key[0][j]

            outeval_dict = self._convert_array_into_new_type(outeval_dict)
            outeval_base_dict = {f'{key_out} vs {key_in}': value for key_out, value in zip(
                self.eval_out_list, outeval_dict.values())}
            outeval_final_dict.update(outeval_base_dict)

        return outeval_final_dict
