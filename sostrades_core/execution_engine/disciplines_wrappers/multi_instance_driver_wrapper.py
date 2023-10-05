'''
Copyright (c) 2023 Capgemini

All rights reserved

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or mother materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND OR ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import copy
import re
import numpy as np

import platform
from tqdm import tqdm
import time

from sostrades_core.tools.base_functions.compute_len import compute_len
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_new_type_into_array, convert_array_into_new_type

from numpy import array, ndarray, delete, NaN

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.tools.proc_builder.process_builder_parameter_type import ProcessBuilderParameterType


'''
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import logging

from sostrades_core.execution_engine.disciplines_wrappers.driver_evaluator_wrapper import DriverEvaluatorWrapper
import pandas as pd
from collections import ChainMap
from gemseo.core.parallel_execution import ParallelExecution


class MultiInstanceDriverWrapper(DriverEvaluatorWrapper):
    # pass
    def multi_instance_run(self):
        """
        Run in the multi instance case.
        """
        pass
        # # very simple ms only
        # self._init_input_data()
        # subpr_to_eval = self.subprocesses_to_eval or range(self.n_subprocs)
        # gather_names = self.attributes['gather_names']
        # gather_out_keys = self.attributes['gather_out_keys']
        # # TODO: if an output does not exist in a scenario, it will not be in the dict. Add entry {sc_name: None} ?
        # gather_output_dict = {key: {} for key in gather_out_keys}
        # # gather_output_dict = {key: {sc: None for sc in self.attributes['scenario_names']} for key in gather_out_keys}
        #
        # for i_subprocess in subpr_to_eval:
        #     self.subprocess_evaluation({}, i_subprocess)
        #     # save data of execution i.e. scenario values
        #     subprocess_outputs = {key: self.attributes['sub_mdo_disciplines'][i_subprocess].local_data[key]
        #                           for key in self.attributes['sub_mdo_disciplines'][i_subprocess].output_grammar.get_data_names()}
        #     self.store_sos_outputs_values(
        #         subprocess_outputs, full_name_keys=True)
        #
        #     # the keys of gather_names correspond to the full names of the vars to gather
        #     gathered_in_subprocess = self._select_output_data(subprocess_outputs, gather_names)
        #     for _gathered_var_name, _gathered_var_value in gathered_in_subprocess.items():
        #         # the values of gather_names are tuples out_key, scenario_name which allow mapping to global_dict_output
        #         out_key, scenario_name = gather_names[_gathered_var_name]
        #         gather_output_dict[out_key][scenario_name] = _gathered_var_value
        # self.store_sos_outputs_values(gather_output_dict)
