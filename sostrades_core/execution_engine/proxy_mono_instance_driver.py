'''
Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
s
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.tools.gather.gather_tool import gather_selected_outputs


class ProxyMonoInstanceDriverException(Exception):
    pass


class ProxyMonoInstanceDriver(ProxyDriverEvaluator):
    SUBCOUPLING_NAME = 'subprocess'
    # TODO: manage desc_in in correct classes
    DESC_IN = {
        ProxyDriverEvaluator.GATHER_OUTPUTS: {ProxyDriverEvaluator.TYPE: 'dataframe',
                                            ProxyDriverEvaluator.DATAFRAME_DESCRIPTOR: {
                                                'selected_output': ('bool', None, True),
                                                'full_name': ('string', None, False),
                                                'output_name': ('multiple', None, True)},
                                            ProxyDriverEvaluator.DATAFRAME_EDITION_LOCKED: False,
                                            ProxyDriverEvaluator.STRUCTURING: True},
        'n_processes': {ProxyDriverEvaluator.TYPE: 'int', ProxyDriverEvaluator.NUMERICAL: True,
                        ProxyDriverEvaluator.DEFAULT: 1},
        'wait_time_between_fork': {ProxyDriverEvaluator.TYPE: 'float', ProxyDriverEvaluator.NUMERICAL: True,
                                   ProxyDriverEvaluator.DEFAULT: 0.0}
    }

    DESC_IN.update(ProxyDriverEvaluator.DESC_IN)

    DESC_OUT = {'samples_inputs_df': {ProxyDriverEvaluator.TYPE: 'dataframe', 'unit': None}}

    def setup_sos_disciplines(self):
        disc_in = self.get_data_in()
        dynamic_inputs = {}
        dynamic_outputs = {}
        if disc_in:
            if self.GATHER_OUTPUTS in disc_in:
                gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
                selected_outputs_dict = gather_selected_outputs(gather_outputs, self.GATHER_DEFAULT_SUFFIX)
                self.selected_outputs = selected_outputs_dict.keys()
                if len(selected_outputs_dict) > 0:
                    self.eval_out_list = [f'{self.get_disc_full_name()}.{element}' for element in selected_outputs_dict.keys()]
                    self.eval_out_names = selected_outputs_dict.values()
                    # setting dynamic outputs. One output of type dict per selected output
                    dynamic_outputs.update(
                        {out_name: {self.TYPE: 'dict'} 
                        for out_name in selected_outputs_dict.values()})
                    dynamic_outputs.update({'samples_outputs_df': {self.TYPE: 'dataframe'}})

                    self.add_outputs(dynamic_outputs)

            if self.SAMPLES_DF in disc_in:
                samples_df = self.get_sosdisc_inputs(self.SAMPLES_DF)
                if samples_df is not None:
                    selected_inputs = set(samples_df.columns)
                    selected_inputs -= self.SAMPLES_DF_DESC[self.DATAFRAME_DESCRIPTOR].keys()
                    if selected_inputs != set(self.selected_inputs):
                        self.selected_inputs = list(selected_inputs)
                        self.eval_in_list = [
                            f'{self.get_disc_full_name()}.{element}' for element in self.selected_inputs]
                        dataframe_descriptor = self.SAMPLES_DF_DESC['dataframe_descriptor'].copy()
                        for key, var_f_name in zip(self.selected_inputs, self.eval_in_list):
                            if var_f_name in self.ee.dm.data_id_map:
                                var = tuple([self.ee.dm.get_data(
                                    var_f_name, self.TYPE), None, True])
                                dataframe_descriptor[key] = var
                            elif self.MULTIPLIER_PARTICULE in var_f_name:
                                # for multipliers assume it is a float
                                dataframe_descriptor[key] = ('float', None, True)
                            else:
                                raise KeyError(f'Selected input {var_f_name} is not in the Data Manager')

    def configure_driver(self):
        if len(self.proxy_disciplines) > 0:
            # CHECK USECASE IMPORT AND IMPORT IT IF NEEDED
            # Manage usecase import
            ref_discipline_full_name = f'{self.ee.study_name}.Eval'
            self.manage_import_inputs_from_sub_process(
                ref_discipline_full_name)
            # SET EVAL POSSIBLE VALUES
            self.set_eval_possible_values()

    def set_wrapper_attributes(self, wrapper):
        super().set_wrapper_attributes(wrapper)
        if self.selected_inputs is not None:
            # specific to mono-instance
            eval_attributes = {
                'eval_out_list': self.eval_out_list,
                'eval_out_names': self.eval_out_names,
                'driver_name': self.get_disc_full_name(),
                'reduced_dm': self.ee.dm.reduced_dm,  # for conversions
                'selected_inputs': self.selected_inputs,
                'selected_outputs': self.selected_outputs,
            }
            wrapper.attributes.update(eval_attributes)

    def prepare_build(self):
        '''
        Get the builder of the single subprocesses in mono-instance builder mode.
        '''
        if self.get_data_in() and self.eval_process_builder is None:
            self._set_eval_process_builder()
        sub_builders = [self.eval_process_builder] if self.eval_process_builder else []
        # add sample generator
        sub_builders.extend(super().prepare_build())
        return sub_builders

    def update_reference(self):
        return bool(self.get_data_in())

    def is_configured(self):
        return super().is_configured() and self.sub_proc_import_usecase_status == 'No_SP_UC_Import'

    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary, which will allow mono-instance builds.
        '''
        updated_ns_list = self.update_sub_builders_namespaces()
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        elif len(self.cls_builder) == 1:
            # Note no distinction is made whether the builder is executable or not; old implementation used to put
            # scatter builds under a coupling automatically too.
            disc_builder = self.cls_builder[0]
        else:
            # If eval process is a list of builders then we build a coupling
            # containing the eval process

            disc_builder = self.create_sub_builder_coupling(
                self.SUBCOUPLING_NAME, self.cls_builder)
            self.hide_coupling_in_driver_for_display(disc_builder)

        self.eval_process_builder = disc_builder

        self.eval_process_builder.add_namespace_list_in_associated_namespaces(
            updated_ns_list)

    def update_sub_builders_namespaces(self):
        '''
        Update sub builders namespaces with the driver name in monoinstance case
        '''

        ns_ids_list = []
        extra_name = f'{self.sos_name}'
        after_name = self.father_executor.get_disc_full_name()

        for ns_name in self.sub_builder_namespaces:
            old_ns = self.ee.ns_manager.get_ns_in_shared_ns_dict(ns_name)
            updated_value = self.ee.ns_manager.update_ns_value_with_extra_ns(
                old_ns.get_value(), extra_name, after_name=after_name)
            display_value = old_ns.get_display_value_if_exists()
            ns_id = self.ee.ns_manager.add_ns(
                ns_name, updated_value, display_value=display_value, add_in_shared_ns_dict=False)
            ns_ids_list.append(ns_id)

        return ns_ids_list

    def hide_coupling_in_driver_for_display(self, disc_builder):
        '''
        Set the display_value of the sub coupling to the display_value of the driver
        (if no display_value filled the display_value is the simulation value)
        '''
        driver_display_value = self.ee.ns_manager.get_local_namespace(
            self).get_display_value()
        self.ee.ns_manager.add_display_ns_to_builder(
            disc_builder, driver_display_value)

    