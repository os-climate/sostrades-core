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
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.execution_engine.proxy_sample_generator import ProxySampleGenerator
from sostrades_core.tools.gather.gather_tool import gather_selected_outputs


class ProxyMonoInstanceDriverException(Exception):
    pass


class ProxyMonoInstanceDriver(ProxyDriverEvaluator):
    _ontology_data = {
        'label': ' Mono-Instance Driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    SUBCOUPLING_NAME = 'subprocess'
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

    def __init__(self, sos_name, ee, cls_builder,
                 driver_wrapper_cls=None,
                 associated_namespaces=None,
                 map_name=None
                 ):
        super().__init__(sos_name, ee, cls_builder, driver_wrapper_cls, associated_namespaces=associated_namespaces, map_name=map_name)
        self.driver_eval_mode = self.DRIVER_EVAL_MODE_MONO

    def setup_sos_disciplines(self):
        disc_in = self.get_data_in()
        dynamic_outputs = {}
        if disc_in and self.GATHER_OUTPUTS in disc_in:
            gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
            selected_outputs_dict = gather_selected_outputs(gather_outputs, self.GATHER_DEFAULT_SUFFIX)
            self.selected_outputs = selected_outputs_dict.keys()
            if len(selected_outputs_dict) > 0:
                self.eval_out_list = self._compose_with_driver_ns(selected_outputs_dict.keys())
                self.eval_out_names = selected_outputs_dict.values()
                # setting dynamic outputs. One output of type dict per selected output
                dynamic_outputs.update(
                    {out_name: {self.TYPE: 'dict'}
                     for out_name in selected_outputs_dict.values()})
                dynamic_outputs.update({'samples_outputs_df': {self.TYPE: 'dataframe'}})

                self.add_outputs(dynamic_outputs)

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

    def check_data_integrity(self):
        '''
        Check the data integrity of the driver (from super) and there should be at least one trades variables
        and at least one gather output should be selected
        '''
        super().check_data_integrity()
        disc_in = self.get_data_in()

        if self.SAMPLES_DF in disc_in:
            value_check = True
            # if we are at run time no need to check the samples and output
            if self.sample_generator_disc is not None:
                sampling_generation_mode = self.sample_generator_disc.sampling_generation_mode
                if sampling_generation_mode == ProxySampleGenerator.AT_RUN_TIME:
                    value_check = False
            if value_check:
                # check that there is at least one trade variables
                # (the trades variables are column with variable names in samples_df)
                samples_df = self.get_sosdisc_inputs(self.SAMPLES_DF)
                variables_column = [col for col in samples_df.columns if col not in self.SAMPLES_DF_COLUMNS_LIST]
                if len(variables_column) == 0:
                    warning_msg = 'There should be at least one trade variable column in samples_df'
                    self.check_integrity_msg_list.append(warning_msg)
                    # save inetrgity message on samples_df
                    self.driver_data_integrity = False
                    data_integrity_msg = '\n'.join(self.check_integrity_msg_list)
                    self.dm.set_data(
                        self.get_var_full_name(self.SAMPLES_DF, disc_in),
                        self.CHECK_INTEGRITY_MSG, data_integrity_msg)

            # check that there is at least one gather output selected
            gather_outputs = self.get_sosdisc_inputs(self.GATHER_OUTPUTS)
            selected_outputs_dict = gather_selected_outputs(gather_outputs, self.GATHER_DEFAULT_SUFFIX)
            if selected_outputs_dict is None or len(selected_outputs_dict) == 0:
                self.dm.set_data(
                    self.get_var_full_name(self.GATHER_OUTPUTS, disc_in),
                    self.CHECK_INTEGRITY_MSG, "There should be at least one selected output")
