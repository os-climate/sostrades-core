'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/19-2023/11/03 Copyright 2023 Capgemini

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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding: iso-8859-1 -*-

import numpy
from sympy import sympify, lambdify, Symbol, factor
# from sympy.core.compatibility import StringIO
# from sympy.parsing.sympy_tokenize import untokenize, generate_tokens
from functools import reduce
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from tokenize import NAME, OP

DOT_CHAR = '_00d00_'


# def my_stringify_expr(s, local_dict, global_dict, transformations):
#     """
#     Converts the string ``s`` to Python code, in ``local_dict``
#     Generally, ``parse_expr`` should be used.
#     """
#     tokens = []
#     input_code = StringIO(s.strip())
#     dots_in = False
#     for toknum, tokval, _, _, _ in generate_tokens(input_code.readline):
#         tokens.append((toknum, tokval))
#         if tokval == '.':
#             dots_in = True
#
#
#     if dots_in:
#         token_out = []
#         previous_toknum = 1
#         while(len(tokens) > 0):
#             toknum, tokval = tokens.pop(0)
#             cat = tokval
#             while(len(tokens) > 1 and tokens[0][1][0] == '.' ):
#                 if tokens[0][1] == '.':
#                     tokens.pop(0)  # remove the next dot
#                     toknum, tokval2 = tokens.pop(0)  # Update toknum value
#                 else:
#                     toknum, tokval2 = tokens.pop(0)  # Update toknum value
#                     tokval2 = tokval2[1:]
#                 # remove math. and numpy. which are useeless with sympy
#                 if tokval == 'math' or tokval == 'numpy':
#                     cat = tokval2
#                 else:
#                     cat += DOT_CHAR + tokval2
#                 toknum=1
#             tokval = cat
#             token_out.append((toknum, tokval))
#         tokens = token_out
#
#     for transform in transformations:
#         tokens = transform(tokens, local_dict, global_dict)
#
#     return untokenize(tokens)

# from sympy.parsing import sympy_parser
# sympy_parser.stringify_expr = my_stringify_expr


def my_transformation(tokens, local_dict, global_dict):
    #     print(tokens)
    new_tokens = []

    i = 0
    while (i < len(tokens)):
        current_token = tokens[i]

        # remove math. and numpy. which are useeless with sympy
        if current_token[1] in ['numpy', 'math']:
            i += 2
            current_token = tokens[i]

        if current_token[0] == NAME and i + 1 < len(tokens):
            while i + 1 < len(tokens) and tokens[i + 1][1] != '' and tokens[i + 1][1][0] == '.':

                if tokens[i + 1][0] == OP:
                    current_token = (NAME, DOT_CHAR.join(
                        [current_token[1], tokens[i + 2][1]]))
                    i += 2
                else:
                    current_token = (NAME, DOT_CHAR.join(
                        [current_token[1], tokens[i + 1][1][1:]]))
                    i += 1
        new_tokens.append(current_token)
        i += 1
    #     print(new_tokens)

    return new_tokens


class SympyFormula():
    """
    Class for mathematical interpretation of functions and their differential forms based on Sympy
    """

    def __init__(self, fexpr, fgrad=True):
        """
        Formula constructor
        @param fexpr : formula expression
        @param fgrad : formula gradient should be calculated ir True
        """
        # Formula Expressions
        self.__fgrad = False
        self.__fexpr = str(fexpr)
        self.__fgradexpr = None

        # Evaluation Expressions and related data
        self.__var_dict = {}
        self.__var_dict_keys = None

        # Evaluation results
        self.__value = None
        self.__gradient = None
        self.__fexpr_sympy = None
        self.__fgradexpr_sympy = None
        self.__sympy_function = None
        self.__sympy_gradient_function = None

        self.__numexpr_f = None
        self.__numexpr_fgrad = None
        # Initialise Expressions from input expression
        self.__init_expressions()

    def differentiate_expr_partial(
            self, atom, simplify_expr=False, simpl_func=None):
        """
        Builds the differential form of the expression for atom "atom" : atom + b*atom ->datom + b*datom
        @param atom : the atom
        @param simplify_expr : if True the expression will be simplified using "simpl_func" sympy function
        @param simpl_func: the simplification function
        """
        if simplify_expr:
            if simpl_func is not None:
                "(" + str(simpl_func(self.__fexpr_sympy.diff(atom))
                          ) + ")*d" + str(atom)
            else:
                return "(" + str(factor(self.__fexpr_sympy.diff(atom))
                                 ) + ")*d" + str(atom)
        return "(" + str(self.__fexpr_sympy.diff(atom)) + ")*d" + str(atom)

    def build_differential_form(self, simplify_expr=False, simpl_func=None):
        """
        Builds the differential form of the expression : a + b*a ->da + b*da + a*db
        @param atom : the atom
        @param simplify_expr : if True the expression will be simplified using "simpl_func" sympy function
        @param simpl_func: the simplification function
        """

        def mapped_func(atom): return self.differentiate_expr_partial(
            atom, simplify_expr, simpl_func)

        def reduced_func(x, y): return x + "+" + y

        return reduce(reduced_func, map(mapped_func, self.__fexpr_symbs))

    def get_symbols(self, sympy_expr, replace_dots=True):
        """
        Accessor for the atomic elements of the expression in the Sympy sense
        @param sympy_expr : the sympy expression
        """

        return sympy_expr.atoms(Symbol)

    # Private methods
    def __repr__(self):
        tokenlist = self.get_symbols(self.__fexpr_sympy)

        info_string = '\n--o0 Formula Information 0o--'
        info_string += '\n  Active gradient     : %s' % self.__fgrad
        info_string += '\n  Formula             : %s' % self.__fexpr
        info_string += '\n  Gradient formula    : %s' % self.__fgradexpr
        info_string += '\n  Token list          : %s' % tokenlist
        info_string += '\n--o0 ------------------- 0o--'
        return info_string

    def __init_expressions(self):
        """
        Initilizes sympy expressions
        """
        self.__fexpr_sympy = parse_expr(self.__fexpr, transformations=(
                                                                          my_transformation,) + standard_transformations)
        self.__fexpr_symbs = self.get_symbols(self.__fexpr_sympy)
        self.__sympy_function = lambdify(
            self.__fexpr_symbs, self.__fexpr_sympy, "numpy", dummify=False)

        # Gradient formula
        if self.__fgrad:
            self.__fgradexpr_sympy = sympify(self.build_differential_form())
            self.__fgradexpr_symbs = set(
                list(
                    self.get_symbols(
                        self.__fgradexpr_sympy)) +
                list(
                    self.__fexpr_symbs))
            self.__sympy_gradient_function = lambdify(
                self.__fgradexpr_symbs,
                self.__fgradexpr_sympy,
                "numpy",
                dummify=False)

    def str_to_sympy_argslist(self, str_dict, symbols):
        """
        Converts, at evaluation time, the symbols dicts of values to sympy arguments
        """

        sympy_args = []
        for atom in symbols:
            tok_name = str(atom).replace(DOT_CHAR, '.')

            sympy_args.append(str_dict[tok_name])
            if not str(atom).startswith("d"):

                if isinstance(sympy_args[-1], numpy.ndarray):
                    sympy_args[-1] = str_dict[str(atom)][0]
                # Checks if the value is a float or is convertible to a float
                # by sympy, otherwise probably an expression
                # TBD find a cleaner way to check failure of test a priori
                try:
                    complex(sympy_args[-1])
                except:
                    # Recursively creates a sub expression
                    subexpr = SympyFormula(sympy_args[-1], fgrad=True)
                    subexpr.evaluate(str_dict)
                    sympy_args[-1] = subexpr.get_value()
                    if self.__fgrad:
                        str_dict['d' + str(atom)] = subexpr.get_gradient()

        return sympy_args

    def set_grad(self, fgrad=True):
        """
        set __fgrad attribute
        @param fgrad: option to compute gradient (B{True}: gradient active, B{False}: gradient inactive)
        @type fgrad: Boolean
        """
        if self.__fgrad != fgrad:
            self.__fgrad = False
            self.__init_expressions()

    def get_token_list(self):
        """
        Gets the token list for the expression
        """
        token_list = list()

        for token in map(str, self.get_symbols(self.__fexpr_sympy)):
            token_list.append(token.replace(DOT_CHAR, '.'))

        return token_list

    def evaluate(self, var_dict):
        """
        Evaluates the expression for given values and gradients
        @param var_dict  : the dictionary of values and gradients
        """
        #         new_var_dict = dict()
        #         for k,v in var_dict.items():
        #             new_var_dict[k.replace('.',DOT_CHAR)]=v
        #         var_dict = new_var_dict

        sympy_args = self.str_to_sympy_argslist(var_dict, self.__fexpr_symbs)
        self.__value = self.__sympy_function(*sympy_args)
        if self.__fgrad:
            sympy_args = self.str_to_sympy_argslist(
                var_dict, self.__fgradexpr_symbs)
            self.__gradient = self.__sympy_gradient_function(*sympy_args)

    def get_formula(self):
        """
        Accessor for the formula expression as a string
        @return the formula expression
        """
        return self.__fexpr

    def get_grad_formula(self):
        """
        Accessor for the formula gradient expression as a string
        @return the formula gradient expression
        """
        if not self.__fgrad:
            print("Warning: gradient formula not generated!")
        return self.__fgradexpr

    def get_value(self):
        """
        Accessor for the evaluated value of the expression
        @return the value
        """
        return self.__value

    def get_gradient(self):
        """
        Accessor for the evaluated gradient of the expression
        @return the value
        """
        return self.__gradient
