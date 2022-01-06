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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from gemseo.core.discipline import MDODiscipline


class CheckJacobianDiscipline(SoSDiscipline):
    _maturity = 'Research'

    SUB_BUILDER_NAME = None
    JAC = 'jacobian'
    JAC_APPROX = 'approximate_jacobian'
    JAC_IS_VALID = 'jacobian_is_valid'

    DESC_IN = {'repo_name': {'type': 'string', 'structuring': True, 'structuring': True},
               'process_name': {'type': 'string', 'structuring': True, 'structuring': True},
               'derr_approx': {'type': 'string', 'default': MDODiscipline.FINITE_DIFFERENCES,
                               'possible_values': MDODiscipline.APPROX_MODES, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'step': {'type': 'float', 'default': 1e-7, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'threshold': {'type': 'float', 'default': 1e-5, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'checkjac_parallel': {'type': 'bool', 'default': False, SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'checkjac_n_processes': {'type': 'int', 'default': MDODiscipline.N_CPUS, 'unit': '[-]', SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'checkjac_use_threading': {'type': 'bool', 'default': False, SoSDiscipline.POSSIBLE_VALUES: [True, False], SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'input_column': {'type': 'string_list', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'output_column': {'type': 'string_list', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'load_jac_path': {'type': 'string', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'dump_jac_path': {'type': 'string', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'inputs': {'type': 'string_list', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               'outputs': {'type': 'string_list', 'unit': '[-]', SoSDiscipline.OPTIONAL: True, SoSDiscipline.NUMERICAL: True, SoSDiscipline.USER_LEVEL: 3},
               }
    DESC_OUT = {JAC: {'type': 'dict', 'unit': '[-]', SoSDiscipline.USER_LEVEL: 3},
                JAC_APPROX: {'type': 'dict', 'unit': '[-]', SoSDiscipline.USER_LEVEL: 3},
                JAC_IS_VALID: {'type': 'bool', 'unit': '[-]', SoSDiscipline.USER_LEVEL: 3}}

    def setup_sos_disciplines(self):
        if 'repo_name' in self._data_in and self.get_sosdisc_inputs('repo_name') is not None:
            if len(self.ee.factory.sos_disciplines) == 1:
                repo_name = self.get_sosdisc_inputs('repo_name')
                process_name = self.get_sosdisc_inputs('process_name')
                builder = self.ee.factory.get_builder_from_process(
                    repo_name, process_name)
                full_name = f'{self.get_disc_full_name()}.{builder.sos_name}'
                self.SUB_BUILDER_NAME = full_name
                builder.set_disc_name(full_name)
                for ns in self.ee.ns_manager.shared_ns_dict.values():
                    self.ee.ns_manager.update_namespace_with_extra_ns(
                        ns, self.name, self.ee.study_name)
                disc = builder.build()
                self.ee.factory.add_discipline(disc)

    def run(self):

        self.logger.info(
            "Check jacobian mode of discipline %s" % self.get_disc_full_name())

        # get inputs
        linearization_mode = self.get_sosdisc_inputs('linearization_mode')
        mode_jac_keys = list(self.DESC_IN.keys())
        jac_inputs = self.get_sosdisc_inputs(mode_jac_keys, in_dict=True)
        optim_disc = self.ee.dm.get_disciplines_with_name(
            self.SUB_BUILDER_NAME)[0]

        # check jacobian
        flag = optim_disc.check_jacobian(input_data=None, derr_approx=jac_inputs['derr_approx'],
                                         step=jac_inputs['step'], threshold=jac_inputs['threshold'],
                                         linearization_mode=linearization_mode,
                                         inputs=jac_inputs['inputs'], outputs=jac_inputs['outputs'],
                                         n_processes=jac_inputs['checkjac_n_processes'], parallel=jac_inputs['checkjac_parallel'],
                                         # wait_time_between_fork=jac_inputs['wait_time_between_fork'],
                                         use_threading=jac_inputs['checkjac_use_threading'],
                                         input_column=jac_inputs['input_column'], output_column=jac_inputs['output_column'],
                                         dump_jac_path=jac_inputs['dump_jac_path'], load_jac_path=jac_inputs['load_jac_path'])

        # store outputs
        output_dict = {self.JAC: optim_disc.jac,
                       self.JAC_APPROX: optim_disc.approx.approx_jac_complete,
                       self.JAC_IS_VALID: flag}
        self.store_sos_outputs_values(output_dict)
