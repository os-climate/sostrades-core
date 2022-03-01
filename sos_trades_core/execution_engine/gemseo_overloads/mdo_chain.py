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

from gemseo.core.chain import MDOChain


def reverse_chain_rule(cls, chain_outputs, discipline):
    """Chains derivatives of self, with a new discipline in the chain in reverse
    mode.

    Performs chain ruling:
    (notation: D is total derivative, d is partial derivative)

    D out    d out      dinpt_1    d output      dinpt_2
    -----  = -------- . ------- + -------- . --------
    D new_in  d inpt_1  d new_in   d inpt_2   d new_in


    D out    d out        d out      dinpt_2
    -----  = -------- + -------- . --------
    D z      d z         d inpt_2     d z


    D out    d out      [dinpt_1   d out      d inpt_1    dinpt_2 ]
    -----  = -------- . [------- + -------- . --------  . --------]
    D z      d inpt_1   [d z       d inpt_1   d inpt_2     d z    ]

    :param discipline: new discipline to compose in the chain
    :param chain_outputs: the chain_outputs to linearize
    """
    # TODO : only linearize wrt needed inputs/inputs
    # use coupling_structure graph path for that
    last_cached = discipline.cache.get_last_cached_inputs()
    discipline.linearize(last_cached, force_no_exec=True, force_all=True)

    for output in chain_outputs:
        if output in cls.jac:
            # This output has already been taken from previous disciplines
            # Derivatives must be composed using the chain rule

            # Make a copy of the keys because the dict is changed in the
            # loop
            existing_inputs = cls.jac[output].keys()
            common_inputs = set(existing_inputs) & set(discipline.jac)
            for input_name in common_inputs:
                # Store reference to the current Jacobian
                curr_j = cls.jac[output][input_name]
                for new_in, new_jac in discipline.jac[input_name].items():
                    # Chain rule the derivatives
                    # TODO: sum BEFORE dot
                    loc_dot = curr_j.dot(new_jac)
                    # when input_name==new_in, we are in the case of an
                    # input being also an output
                    # in this case we must only compose the derivatives
                    if new_in in cls.jac[output] and input_name != new_in:
                        # The output is already linearized wrt this
                        # input_name. We are in the case:
                        # d o     d o    d o     di_2
                        # ----  = ---- + ----- . -----
                        # d z     d z    d i_2    d z
                        cls.jac[output][new_in] += loc_dot
                    else:
                        # The output is not yet linearized wrt this
                        # input_name.  We are in the case:
                        #  d o      d o     di_1   d o     di_2
                        # -----  = ------ . ---- + ----  . ----
                        #  d x      d i_1   d x    d i_2    d x
                        cls.jac[output][new_in] = loc_dot

        elif output in discipline.jac:
            # Output of the chain not yet filled in jac,
            # Take the jacobian dict of the current discipline to
            # Initialize. Make a copy !
            cls.jac[output] = MDOChain.copy_jacs(discipline.jac[output])


setattr(MDOChain, "reverse_chain_rule", reverse_chain_rule)
