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
# pylint: skip-file
"""A Gauss Seidel algorithm for solving MDAs."""


from gemseo.core.discipline import MDODiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from numpy import array


class SoSMDAGaussSeidel(MDAGaussSeidel):
    """ Overload of GEMSEO's MDA GaussSeidel
    (overload introduces warm_start_threshold option)
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        name=None,  # type: Optional[str]
        max_mda_iter=10,  # type: int
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        over_relax_factor=1.0,  # type: float
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False,  # type: bool
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
        warm_start_threshold=-1,  # type: int
    ):  # type: (...) -> None
        """
        Args:
            over_relax_factor: The relaxation coefficient,
                used to make the method more robust,
                if ``0<over_relax_factor<1`` or faster if ``1<over_relax_factor<=2``.
                If ``over_relax_factor =1.``, it is deactivated.
        """
        # Not possible to parallelize MDAGaussSeidel execution
        self.n_processes = 1
        MDAGaussSeidel.__init__(
            self,
            disciplines,
            name=name,
            max_mda_iter=max_mda_iter,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            over_relax_factor=over_relax_factor,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
        )
        self.warm_start_threshold = warm_start_threshold

    def _run(self):
        # Run the disciplines in a sequential way
        # until the difference between outputs is under tolerance.
        if self.warm_start:
            self._couplings_warm_start()
        # sostrades modif to support array.size for normalization
        current_couplings = array([0.0])

        relax = self.over_relax_factor
        use_relax = relax != 1.0

        # -- SoSTrades modif
        # stores cache history if residual_start filled
        if self.warm_start_threshold != -1:
            self.store_state_for_warm_start()
        # -- end of SoSTrades modif

        # store initial residual
        current_iter = 0
        while not self._termination(current_iter) or current_iter == 0:
            for discipline in self.disciplines:
                discipline.execute(self.local_data)
                outs = discipline.get_output_data()
                if use_relax:
                    # First time this output is computed, update directly local
                    # data
                    self.local_data.update(
                        {k: v for k, v in outs.items() if k not in self.local_data}
                    )
                    # The couplings already exist in the local data,
                    # so the over relaxation can be applied
                    self.local_data.update(
                        {
                            k: relax * v + (1.0 - relax) * self.local_data[k]
                            for k, v in outs.items()
                            if k in self.local_data
                        }
                    )
                else:
                    self.local_data.update(outs)

            # build new_couplings: concatenated strong couplings, converted into arrays
            new_couplings = self._current_strong_couplings()
            
            self._compute_residual(
                current_couplings,
                new_couplings,
                current_iter,
                first=current_iter == 0,
                log_normed_residual=self.log_convergence,
            )
            
            # store current residuals
            current_iter += 1
            current_couplings = new_couplings

            # -- SoSTrades modif
            # stores cache history if residual_start filled
            if self.warm_start_threshold != -1:
                self.store_state_for_warm_start()
            # -- end of SoSTrades modif

        for discipline in self.disciplines:  # Update all outputs without relax
            self.local_data.update(discipline.get_output_data())
