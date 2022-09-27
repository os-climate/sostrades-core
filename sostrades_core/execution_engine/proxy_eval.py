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
from sostrades_core.tools.base_functions.compute_len import compute_len

'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np
from pandas.core.frame import DataFrame

from sostrades_core.api import get_sos_logger
from sostrades_core.execution_engine.ns_manager import NS_SEP
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_discipline_driver import ProxyDisciplineDriver


class ProxyEvalException(Exception):
    pass


class ProxyEval(ProxyDisciplineDriver):
    '''
        SOSEval class which creates a sub process to evaluate
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

    def __init__(self, sos_name, ee, cls_builder):
        '''
        Constructor
        '''
        super().__init__(sos_name, ee, cls_builder)
        self.eval_in_base_list = None
        self.eval_in_list = None
        self.eval_out_base_list = None
        self.eval_out_list = None
        # Needed to reconstruct objects from flatten list
        self.eval_out_type = []
        self.eval_out_list_size = []
        self.logger = get_sos_logger(f'{self.ee.logger.name}.Eval')
        # self.cls_builder = cls_builder
        # Create the eval process builder associated to SoSEval
        self.eval_process_builder = self._set_eval_process_builder()
        self.eval_process_disc = None

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
                ProxyCoupling.DESC_IN.keys())
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
        # children of SoSEval
        old_current_discipline = self.ee.factory.current_discipline
        self.ee.factory.current_discipline = self

        # if we want to build an eval coupling containing eval process,
        # we have to remove SoSEval name in current_ns to build eval coupling
        # at the same node as SoSEval
        if len(self.cls_builder) == 0:  # added condition for proc build
            pass
        elif self.cls_builder[0] != self.eval_process_builder:
            current_ns = self.ee.ns_manager.current_disc_ns
            self.ee.ns_manager.set_current_disc_ns(
                current_ns.split(f'.{self.sos_name}')[0])
            self.build_eval_process()
            # reset current_ns after build
            self.ee.ns_manager.set_current_disc_ns(current_ns)
        else:
            self.build_eval_process()

        # If the old_current_discipline is None that means that it is the first build of a coupling then self is the high
        # level coupling and we do not have to restore the current_discipline
        if old_current_discipline is not None:
            self.ee.factory.current_discipline = old_current_discipline

    def build_eval_process(self):
        # build coupling containing eval process if self.cls_builder[0] != self.eval_process_builder
        # or build and store eval process in the children of SoSEval
        self.eval_process_disc = self.eval_process_builder.build()
        # store coupling in the children of SoSEval
        if self.eval_process_disc not in self.proxy_disciplines:
            self.ee.factory.add_discipline(self.eval_process_disc)

    def configure_driver(self):
        # Extract variables for eval analysis
        if len(self.proxy_disciplines) > 0:
            self.set_eval_possible_values()

    def set_eval_possible_values(self):
        '''
            Once all disciplines have been run through,
            set the possible values for eval_inputs and eval_outputs in the DM
        '''
        # the eval process to analyse is stored as the only child of SoSEval
        # (coupling chain of the eval process or single discipline)
        analyzed_disc = self.proxy_disciplines[0]

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
        if len(disc.proxy_disciplines) != 0:
            for sub_disc in disc.proxy_disciplines:
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
        return x0 #Removed cast to array

    def _set_eval_process_builder(self):
        '''
        Create the eval process builder, in a coupling if necessary
        '''
        if len(self.cls_builder) == 0:  # added condition for proc build
            disc_builder = None
        else:
            # If eval process is a list of builders or a non executable builder,
            # then we build a coupling containing the eval process
            # In the case of a single sub-disc for sos_eval, although len(self.cls_builder) = 1 or it is an
            # executable discipline, a coupling is also wanted to contain the eval process:
            disc_builder = self.ee.factory.create_builder_coupling(
                self.sos_name)
            disc_builder.set_builder_info('cls_builder', self.cls_builder)

        return disc_builder

    def set_wrapper_attributes(self, wrapper):
        """ set the attribute attributes of wrapper
        """
        # ProxyDisciplineDriver attributes (sub_mdo_discipline)
        super().set_wrapper_attributes(wrapper)
        eval_attributes = {'eval_in_list': self.eval_in_list,
                          'eval_out_list': self.eval_out_list,
                          'reference_scenario': self.get_x0(),
                          'activated_elems_dspace_df': [[True, True]
                                                        if self.ee.dm.get_data(var, 'type') == 'array' else [True]
                                                        for var in self.eval_in_list], # TODO: Array dimensions greater than 2???
                          'study_name': self.ee.study_name,
                          'reduced_dm': self.ee.dm.reduced_dm, #for conversions
                          }
        wrapper.attributes.update(eval_attributes)

    # def set_discipline_attributes(self, discipline):
    #     """ set the attribute attributes of gemseo object
    #     """
    #     # TODO : attribute has been added to SoSMDODiscipline __init__, use sos_disciplines rather ?
    #     discipline.disciplines = [self.proxy_disciplines[0].mdo_discipline_wrapp.mdo_discipline]

#     def prepare_execution(self):
#         '''
#         Preparation of the GEMSEO process, including GEMSEO objects instanciation
#         '''
#         # prepare_execution of proxy_disciplines as in coupling
#         # TODO: move to builder ?
#         sub_mdo_disciplines = []
#         for disc in self.proxy_disciplines:
#             disc.prepare_execution()
#             # Exclude non executable proxy Disciplines
#             if disc.mdo_discipline_wrapp is not None:
#                 sub_mdo_disciplines.append(disc.mdo_discipline_wrapp.mdo_discipline)
#
#         # FIXME : cache mgmt?
#         super().prepare_execution()
# #         '''
# #         GEMSEO objects instanciation
# #         '''
# #         if self.mdo_discipline_wrapp.mdo_discipline is None:
# #             # init gemseo discipline if it has not been created yet
# #             self.mdo_discipline_wrapp.create_gemseo_discipline(proxy=self,
# #                                                                reduced_dm=self.ee.dm.reduced_dm,
# #                                                                cache_type=self.get_sosdisc_inputs(self.CACHE_TYPE),
# #                                                                cache_file_path=self.get_sosdisc_inputs(
# #                                                                    self.CACHE_FILE_PATH),
# #                                                                disciplines=sub_mdo_disciplines)
# #
# #         else:
# #             # TODO : this should only be necessary when changes in structuring variables happened?
# #             self.set_wrapper_attributes(self.mdo_discipline_wrapp.wrapper)
# #
# #             if self._reset_cache:
# #                 # set new cache when cache_type have changed (self._reset_cache == True)
# #                 self.set_cache(self.mdo_discipline_wrapp.mdo_discipline, self.get_sosdisc_inputs(self.CACHE_TYPE),
# #                                self.get_sosdisc_inputs(self.CACHE_FILE_PATH))
# # #             if self._reset_debug_mode:
# # #                 # update default values when changing debug modes between executions
# # #                 to_update_debug_mode = self.get_sosdisc_inputs(self.DEBUG_MODE, in_dict=True, full_name=True)
# # #                 self.mdo_discipline_wrapp.update_default_from_dict(to_update_debug_mode)
# #             # set the status to pending on GEMSEO side (so that it does not stay on DONE from last execution)
# #             self.mdo_discipline_wrapp.mdo_discipline.status = MDODiscipline.STATUS_PENDING
# #         self.status = self.mdo_discipline_wrapp.mdo_discipline.status
# #         self._reset_cache = False
# #         self._reset_debug_mode = False