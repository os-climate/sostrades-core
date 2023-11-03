'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/04-2023/11/03 Copyright 2023 Capgemini

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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
from copy import deepcopy
import logging
#from sostrades_core.execution_engine.gemseo_addon.mda.gs_purenewton_mda import GSPureNewtonMDA

"""
A chain of MDAs to build hybrids of MDA algorithms sequentially
***************************************************************
"""

from sostrades_core.execution_engine.gemseo_addon.mda.gauss_seidel import SoSMDAGaussSeidel
from gemseo.api import create_mda

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.sequential_mda import MDASequential


LOGGER = logging.getLogger("gemseo.addons.mda.purenewton_or_gs")


class GSPureNewtonorGSMDA(MDASequential):
    """
    Perform some GaussSeidel iterations and then NewtonRaphson iterations.
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        max_mda_iter=10,  # type: int
        relax_factor=0.99,  # type: float
        linear_solver="DEFAULT",  # type: str
        tolerance_gs=10.0,
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        linear_solver_options=None,  # type: Mapping[str,Any]
        log_convergence=False,  # type: bool
        **newton_mda_options  # type: float,  # type: Mapping[str,Any]
):
        """
        Constructor

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param name: name
        :type name: str
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str
        :param tolerance: tolerance of the iterative direct coupling solver,
            norm of the current residuals divided by initial residuals norm
            shall be lower than the tolerance to stop iterating
        :type tolerance: float
        :param max_mda_iter: maximum number of iterations
        :type max_mda_iter: int
        :param relax_factor: relaxation factor
        :type relax_factor: float
        :param linear_solver: type of linear solver to be used to solve
            the Newton problem
        :type linear_solver: str
        :param max_mda_iter_gs: maximum number of iterations of the GaussSeidel
            solver
        :type max_mda_iter_gs: int
        :param warm_start: if True, the second iteration and ongoing
            start from the previous coupling solution
        :type warm_start: bool
        :param linear_solver_tolerance: Tolerance of the linear solver
            in the adjoint equation
        :type linear_solver_tolerance: float
        :param use_lu_fact: if True, when using adjoint/forward
            differenciation, store a LU factorization of the matrix
            to solve faster multiple RHS problem
        :type use_lu_fact: bool
        :param newton_mda_options: options passed to the MDANewtonRaphson
        :type newton_mda_options: dict
        """
        mda_gs = SoSMDAGaussSeidel(disciplines, max_mda_iter=max_mda_iter,
                                   name=None, grammar_type=grammar_type)
        mda_gs.tolerance = tolerance

        mda_newton = create_mda(
            'GSPureNewtonMDA',disciplines,  max_mda_iter=max_mda_iter,
                                 name=None, grammar_type=grammar_type,
                                 linear_solver=linear_solver,
                                 linear_solver_options=linear_solver_options,
                                 tolerance_gs=tolerance_gs,
                                 use_lu_fact=use_lu_fact, tolerance=tolerance,
                                 relax_factor=relax_factor,
                                 ** newton_mda_options
        )

        # mda_newton = GSPureNewtonMDA(disciplines,  max_mda_iter=max_mda_iter,
        #                          name=None, grammar_type=grammar_type,
        #                          linear_solver=linear_solver,
        #                          linear_solver_options=linear_solver_options,
        #                          tolerance_gs=tolerance_gs,
        #                          use_lu_fact=use_lu_fact, tolerance=tolerance,
        #                          relax_factor=relax_factor,
        #                          ** newton_mda_options)

        sequence = [mda_gs, mda_newton]
        super(GSPureNewtonorGSMDA,
              self).__init__(disciplines, sequence, name=name,
                             grammar_type=grammar_type,
                             max_mda_iter=max_mda_iter,
                             tolerance=tolerance,
                             linear_solver_options=linear_solver_options,
                             linear_solver_tolerance=linear_solver_tolerance,
                             warm_start=warm_start)

    def _run(self):
        """Runs the MDAs in a sequential way

        :returns: the local data
        """
        self._couplings_warm_start()
        # execute MDAs in sequence
        if self.reset_history_each_run:
            self.residual_history = []
        # initialize dm_values
        dm_values = {}
        try:
            mda_i = self.mda_sequence[1]
            mda_i.reset_statuses_for_run()
            dm_values = deepcopy(self.disciplines[0].dm.get_data_dict_values())
            self.local_data = mda_i.execute(self.local_data)
        except:
            LOGGER.warning(
                'The GSPureNewtonMDA has not converged try with MDAGaussSeidel')
            mda_i = self.mda_sequence[0]
            mda_i.reset_statuses_for_run()
            dm = self.disciplines[0].ee.dm
            # set values directrly in dm to avoid reconfigure of disciplines
            dm.set_values_from_dict(dm_values)
            # self.disciplines[0].ee.load_study_from_input_dict(dm_values)
            self.local_data = mda_i.execute(self.local_data)

        self.residual_history += mda_i.residual_history
