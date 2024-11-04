# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

# pylint: skip-file
"""A Gauss Seidel algorithm for solving MDAs."""

from __future__ import annotations

from gemseo.mda.gauss_seidel import MDAGaussSeidel
from numpy import array

SOS_GRAMMAR_TYPE = "SoSSimpleGrammar"
class SoSMDAGaussSeidel(MDAGaussSeidel):
    """Overload of GEMSEO's MDA GaussSeidel
    (overload introduces warm_start_threshold option)
    """

    def _run(self):
        # Run the disciplines in a sequential way
        # until the difference between outputs is under tolerance.
        if self.warm_start:
            self._couplings_warm_start()
        # sostrades modif to support array.size for normalization
        current_couplings = array([0.0])

        relax = self.over_relaxation_factor
        use_relax = relax != 1.0

        # store initial residual
        current_iter = 0
        while not self._termination(current_iter) or current_iter == 0:
            for discipline in self._disciplines:
                discipline.execute(self.io.data)
                outs = discipline.get_output_data()
                if use_relax:
                    # First time this output is computed, update directly local
                    # data
                    self.io.data.update({k: v for k, v in outs.items() if k not in self.io.data})
                    # The couplings already exist in the local data,
                    # so the over relaxation can be applied
                    self.io.data.update({
                        k: relax * v + (1.0 - relax) * self.io.data[k]
                        for k, v in outs.items()
                        if k in self.io.data
                    })
                else:
                    self.io.data.update(outs)

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

        for discipline in self._disciplines:  # Update all outputs without relax
            self.io.data.update(discipline.get_output_data())
