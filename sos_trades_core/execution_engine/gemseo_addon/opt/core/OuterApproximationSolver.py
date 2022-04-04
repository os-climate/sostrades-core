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
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt.opt_lib import OptimizationLibrary

from numpy import array, append, arange, zeros, matmul, int32, atleast_2d, dot
from copy import deepcopy
import cvxpy as cp

class OuterApproximationSolver(object):
    '''
    Implementation of Outer Approximation solver
    '''
    ETA = "eta"
    UPPER_BOUND = "U"
    FULL_PROBLEM_DV_NAME = "x"
    MILP_DV_NAME_INT = FULL_PROBLEM_DV_NAME + '_int'
    MILP_DV_NAME_FLOAT = FULL_PROBLEM_DV_NAME + '_float'
    ALGO_OPTIONS_MILP = "algo_options_MILP"
    ALGO_OPTIONS_NLP = "algo_options_NLP"
    ALGO_NLP = "algo_NLP"
    NORMALIZE_DESIGN_SPACE_OPTION = DriverLib.NORMALIZE_DESIGN_SPACE_OPTION
    MAX_ITER = OptimizationLibrary.MAX_ITER
    F_TOL_ABS = OptimizationLibrary.F_TOL_ABS

    def __init__(self, problem):
        '''
        Constructor
        '''
        self.full_problem = problem
        if not problem.minimize_objective:
            msg = "Problem defined as an objective maximization instead of minimization"
            raise ValueError(msg)
        self.dual_problem = None
        self.primal_problem = None
        self.epsilon = 1e-3
        self.upper_bounds_candidates = []
        self.upper_bounds = []
        self.lower_bounds = []
        self.cont_solution = []
        self.int_solution = []
        self.x_solution_history = []
        self.ind_by_varname, self.size_by_varname = None, None
    
    def set_options(self, **options):
        
        self.differentiation_method = self.full_problem.differentiation_method
        self.max_iter = options[self.MAX_ITER]
        self.ftol_abs = options[self.F_TOL_ABS]
        
        self.algo_options_MILP = options[self.ALGO_OPTIONS_MILP]
        self.algo_NLP = options[self.ALGO_NLP]
        self.algo_options_NLP = options[self.ALGO_OPTIONS_NLP]
    
    def init_solver(self):
        
        dspace = self.full_problem.design_space
        self._compute_xvect_indices_and_sizes(dspace)
        
        iv_ind = array([], dtype=int32)
        for iv in dspace.variables_names:
            if iv in self.int_varnames:
                iv_ind = append(iv_ind, dspace.get_variables_indexes([iv]))
        self.integer_indices = iv_ind
        
        fv_ind = array([], dtype=int32)
        for fv in dspace.variables_names:
            if fv in self.float_varnames:
                fv_ind = append(fv_ind, dspace.get_variables_indexes([fv]))
        self.float_indices = fv_ind

    def _compute_xvect_indices_and_sizes(self, dspace):
        '''Computes the indices for each variable name
        '''
        float_varnames = []
        int_varnames = []
        for name in dspace.variables_names:
            if dspace.get_type(name) == [DesignSpace.INTEGER.value]: # pylint: disable=E0602
                int_varnames.append(name)
            else:
                float_varnames.append(name)
                
        self.ind_by_varname = dspace.get_variables_indexes(dspace.variables_names)
        self.size_by_varname = dspace.variables_sizes
        self.int_varnames = int_varnames
        self.float_varnames = float_varnames
            
#     def _check(self, dspace):
#         
#         if len(dspace.get_type(v)) > 1:
#             msg = 'The design variable <%s> has several types instead of one for all components.\n' %v
#             msg += '(different types for each component of the variable is not handled for now)'
#             raise ValueError(msg)
#         
#         if len(self.full_problem.objective.outvars) > 1:
#             raise ValueError("Several outputs in MDOFunction is not allowed")
#         
#         for c in self.full_problem.constraints:
#             if len(c.outvars) > 1:
#                 raise ValueError("Several outputs in MDOFunction is not allowed")
                
    
    def _get_integer_variables_indices(self, dspace):
        ''' returns integer variables indices in xvect defined by the design space
        '''
        return self._get_x_indices_by_type(dspace, 
                                           DesignSpace.INTEGER.value)
        
    def _get_float_variables_indices(self, dspace):
        ''' returns float variables indices in xvect defined by the design space
        '''
        return self._get_x_indices_by_type(dspace, 
                                           DesignSpace.FLOAT.value)
    
    def _build_full_vect(self, float_vals, int_vals):
        ''' builds the global xvect with continuous and integer values
        '''
        fdspace = self.full_problem.design_space
        
        x = fdspace.get_current_x()
        
        if len(x) != len(self.float_indices) + len(self.integer_indices):
            msg = 'Sum of Integer and Float design variables '
            msg += 'components is not equal to the design space full size.'
            raise ValueError(msg)
        
        x[self.integer_indices] = int_vals
        x[self.float_indices] = float_vals
        
        return x
        
    
    #- primal problem definition
    
    def build_primal_pb(self):
        ''' build primal problem without hyperplanes
        (will be updated at other iterations)
        '''
        # design variables definition
        eta = cp.Variable(name=self.ETA)

        # objective definition
        obj = cp.Minimize(eta)
        
        # constraints definition
        ub = cp.Parameter(name=self.UPPER_BOUND)
        constraints = [eta <= ub - self.epsilon] # eta <= U^(k) - eps
        
        # problem definition
        prob = cp.Problem(obj, constraints)
        
        return prob
    
    def update_primal_pb(self, old_primal_pb, dual_pb, upper_bnd, x0):
        ''' update primal problem with new upper bound value U^{(k)}
        and supporting hyperplanes (linearizations of objecgives and constraints
        of the NLP(x_int)^{(k)} )
        '''
        x0_dict = self.full_problem.design_space.array_to_dict(x0)
        # gather eta design variable
        eta = old_primal_pb.var_dict[self.ETA]
        # gather x design variable if exists (for iterations > 0)
        bounds_cst = []
        if len(old_primal_pb.var_dict) > 1:
            all_vars = old_primal_pb.var_dict
        else:
            bounds_cst = []
            # if not, x is created with associated bounds constraints
            # create float variables and associated bound constraints
            float_vars = {}
            for v in self.float_varnames:
                v_shape = self.full_problem.design_space.get_current_x_dict()[v].shape
                fv = cp.Variable(v_shape, v, integer=False)
                float_vars[v] = fv
                lb = self.full_problem.design_space.get_lower_bounds([v])
                bounds_cst.append(lb <= fv)
                ub = self.full_problem.design_space.get_upper_bounds([v])
                bounds_cst.append(fv <= ub)
            # create int variables and associated bound constraints
            int_vars = {}
            for v in self.int_varnames:
                v_shape = self.full_problem.design_space.get_current_x_dict()[v].shape
                iv = cp.Variable(v_shape, v, integer=True)
                int_vars[v] = iv
                lb = self.full_problem.design_space.get_lower_bounds([v])
                bounds_cst.append(lb <= iv)
                ub = self.full_problem.design_space.get_upper_bounds([v])
                bounds_cst.append(fv <= iv)
            
            all_vars = {}
            all_vars.update(float_vars)
            all_vars.update(int_vars)
                
        # primal problem constraint :
        # build dual pb objective linearization
        obj_jac = atleast_2d(self.full_problem.objective.jac(x0))
        data_size = deepcopy(self.size_by_varname)
        data_size.update({self.full_problem.objective.outvars[0]: obj_jac.shape[0]})
        obj_jac_dict = split_array_to_dict_of_arrays(obj_jac,
                                         data_size,
                                         self.full_problem.objective.outvars, 
                                         self.full_problem.design_space.variables_names,
                                         )
        obj_f = self.full_problem.objective.func(x0)
        obj_lin = obj_f
        for v in self.full_problem.design_space.variables_names:
            x_v = all_vars[v]
            x0_v = x0_dict[v]
            dx = x_v - x0_v
            obj_lin += obj_jac_dict[self.full_problem.objective.outvars[0]][v] @ dx
        obj_lin = obj_lin <= eta
        
        # build dual pb constraints linearization
        cst_linearized = []
        for c in self.full_problem.constraints:
            c_jac = atleast_2d(c.jac(x0))
            data_size = deepcopy(self.size_by_varname)
            data_size.update({c.outvars[0]: c_jac.shape[0]})
            c_jac_dict = split_array_to_dict_of_arrays(c_jac, 
                                            data_size,
                                             c.outvars, 
                                             self.full_problem.design_space.variables_names,
                                             )
        
            cst_f = c.func(x0)
            cst_lin = cst_f
            for v in self.full_problem.design_space.variables_names:
                x_v = all_vars[v]
                x0_v = x0_dict[v]
                dx = x_v - x0_v
#                 cst_jac = atleast_2d(c.jac(x0))
                cst_lin += c_jac_dict[c.outvars[0]][v] @ dx
                
            cst_linearized.append(cst_lin <= 0)
        
        # problem re-definition (cvxpy does not allow in-memory problem updates, excepted parameter values)
        hyperplanes = [obj_lin] + cst_linearized
        primal_pb = cp.Problem(old_primal_pb.objective, 
                               old_primal_pb.constraints + hyperplanes + bounds_cst)
        
        print('primal_pb', primal_pb.var_dict)
        
        # update upper bound parameter value
        ub = primal_pb.param_dict[self.UPPER_BOUND]
        print("upper_bnd", upper_bnd)
        ub.value = upper_bnd
        
        return primal_pb
    
    def solve_primal(self, problem):
        ''' solve the primal problem
        '''
        problem.solve(solver=cp.CBC, verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Optimal value: %s" % problem.value)
            sol_int = array([])
            for dv in self.full_problem.design_space.variables_names:
                if dv in self.int_varnames:
                    val = problem.var_dict[dv].value
                    sol_int = append(sol_int, val)
            self.int_solution.append(sol_int)
            self.lower_bounds.append(problem.value)
        else:
            sol_int = None
        
        
        for variable in problem.variables():
            print("Variable %s: value %s" % (variable.name(), variable.value))
        
        print ("status:", problem.status)
        print ("optimal value", problem.value)
        print ("optimal var", [(k, v) for k, v in problem.var_dict.items()])
        
        return sol_int

    #- dual problem definition
    
    def build_dual_pb(self, integer_values):
        ''' Build the dual problem
        '''
        # retrieve full problem
        full_pb = self.full_problem
        
        # original design space filtered without integer variables
        cont_vars = []
        for v in full_pb.design_space:
            if full_pb.design_space.get_type(v) == DesignSpace.FLOAT.value: # pylint: disable=E0602
                cont_vars.append(v)
        dspace = full_pb.design_space.filter(cont_vars, copy=True)
        
        input_dim = sum(full_pb.design_space.variables_sizes.values())
        
        # build restriction of original constraint functions
        print("integer_indices", self.integer_indices)
        print("integer_values", integer_values)
        print("input_dim", input_dim)
            
        cst_restricted = []
        for c in full_pb.nonproc_constraints:
            new_c_name = c.name + '_restricted'
            new_c = c.restrict(self.integer_indices, #frozen indexes
                               integer_values, #frozen values
                               input_dim,
                               name=new_c_name,
                               f_type=MDOFunction.TYPE_INEQ,
                               #expr=f"{f.name}(%s)",
                               args=None)
            cst_restricted.append(new_c)
        
        # build restriction of original objective functions
        new_o_name = full_pb.objective.name + '_restricted'
        new_o = full_pb.nonproc_objective.restrict(self.integer_indices, #frozen indexes
                                           integer_values, #frozen values
                                           input_dim,
                                           name=new_o_name,
                                           f_type=MDOFunction.TYPE_OBJ,
                                           #expr=f"{f.name}(%s)",
                                           args=None)
        
        # build dual problem
        pb = OptimizationProblem(dspace)
        # objective setup
        pb.objective = new_o
        # constraints setup
        for c in cst_restricted:
            pb.add_constraint(c, cstr_type=MDOFunction.TYPE_INEQ)
        pb.differentiation_method = self.differentiation_method#OptimizationProblem.USER_GRAD# FINITE_DIFFERENCES USER_GRAD
        
        return pb
    
     
    def update_dual(self, nlp, integer_values):
        ''' Updates frozen values of NLP problem with those provided
        '''
        nlp.objective.set_frozen_value(integer_values)
         
        for f in nlp.constraints:
            f.set_frozen_value(integer_values)
        
        return nlp
    
    def solve_dual(self, nlp):
        ''' Solves the dual problem
        '''
        print("self.algo_options_NLP", self.algo_options_NLP)
        cont_sol = OptimizersFactory().execute(nlp, self.algo_NLP,
                          **self.algo_options_NLP#normalize_design_space=False,
                          )
        
        self.cont_solution.append(cont_sol.f_opt)
        
        return cont_sol
    
            
    # main Outer Approximation algorithm
    def _termination_criteria(self, ite_nb, mip):
        ''' termination criteria computation
        '''
        if ite_nb == 0:
            _continue = True
        else:
            if mip.status == "infeasible":
                _continue = False
                msg = "Tolerance reached"
                print(msg)
            elif self.upper_bounds[ite_nb-1] - self.lower_bounds[ite_nb-1] <= self.epsilon:
                _continue = False
                msg = "Tolerance reached"
                print(msg)
            else:
                _continue = True
        
        return _continue
    
    def update_history(self, pb):
        
        self.upper_bounds_candidates.append(pb.f_opt)
        
        uk = min(self.upper_bounds_candidates)
        self.upper_bounds.append(uk)
        
    def get_upper_bound(self, it_nb):
        return self.upper_bounds[it_nb]
    
    def solve(self):
        ''' Solve the optimization problem
        '''
        msg = """***\nOuterApproximation Initialization\n***"""
#         self.logger.info(msg)
        iter_nb = 0

        x0 = self.full_problem.design_space.get_current_x()
        self.x0_integer = x0[self.integer_indices]
        
        # build NLP(x0_integer)
        nlp = self.build_dual_pb(self.x0_integer)
        
        # compute argmin NLP(x0)
        nlp_sol = self.solve_dual(nlp)
        xopt_cont = nlp_sol.x_opt
        
        self.update_history(nlp_sol)
        
        xsol = self._build_full_vect(xopt_cont, self.x0_integer)
        
        # initialize primal problem
        mip = self.build_primal_pb()
        
        print("xsol", xsol)
        while self._termination_criteria(iter_nb, mip):
            # update primal problem
            uk = self.get_upper_bound(iter_nb)
            mip = self.update_primal_pb(mip, nlp, uk, xsol)
            
            # solve primal problem
            xopt_int = self.solve_primal(mip)
            print("xopt_int", xopt_int)

            if xopt_int is None:
                break
            
            # build NLP(integer solution iteration k)
#             nlp = self.build_dual_pb(xopt_int)
            nlp = self.update_dual(nlp, xopt_int)
        
            # compute argmin NLP(integer solution iteration k)
            nlp_sol = self.solve_dual(nlp)
            xopt_cont = nlp_sol.x_opt
            xsol = self._build_full_vect(xopt_cont, xopt_int)
            
            self.x_solution_history.append(xsol)
            
#             data = {""}
#             self.full_problem.database.store()
            print("UPPER BOUNDS")
            print(self.upper_bounds)
            print("LOWER BOUNDS")
            print(self.lower_bounds)
            self.update_history(nlp_sol)
            
            iter_nb +=1
            
        print("Integer solution is", self.int_solution)
        print("Continuous solution is", self.cont_solution)
        

#     def _build_primal_objective(self, dspace):
#         ''' build primal problem objective function
#         '''
#         # gather indices of objective of relaxed problem in xvect
#         indices = self._compute_xvect_indices(dspace)
#         i_min, i_max = indices[self.primal_obj_name]
#         
#         # relaxed objective function
#         def primal_obj(xvect):
#             return xvect[i_min : i_max]
#         
#         # relaxed objective jacobian
#         def primal_obj_jac(xvect):
#             xlen = len(xvect)
#             jac = zeros((xlen, xlen))
#             jac[i_min : i_max, i_min : i_max] = array([[1]]) # identity(i_max-i_min)
#             return jac
#         
#         return MDOFunction(primal_obj,
#                             self.primal_obj_name,
#                             args="x",
#                             expr=self.primal_obj_name,
#                             jac=primal_obj_jac,
#                             outvars=self.primal_obj_name,
#                             f_type=MDOFunction.TYPE_INEQ,
#                             dim=1)
#         
#     def build_primal(self):
#         ''' build the primal problem
#         '''
#         # retrieve full problem
#         full_pb = self.full_problem
#         # relaxed objective name added to original design space
#         dspace = deepcopy(full_pb.dspace)
#         dspace.add_variable(self.relaxed_obj_name)
#         # build dual problem
#         pb = OptimizationProblem(dspace)
#         # objective setup
#         obj = self._build_primal_objective(dspace)
#         pb.objective = obj
#             
#         return pb
#     
#     def update_primal_pb(self, primal_pb, xvect, iter_nb):
#         ''' Update primal with supporting hyperplanes 
#         of objective and constraints
#         '''
#         full_pb = self.full_problem
# 
#         # build supporting hyperplanes of constraints functions
#         cst_restricted = []
#         for c in full_pb.get_right_sign_constraints():
#             new_c_name = c.name + '_linearized_ite#' + str(iter_nb)
#             new_c = c.linear_approximation(xvect,
#                                            name=new_c_name,
#                                            f_type=MDOFunction.TYPE_INEQ,
#                                            #expr=f"{f.name}(%s)",
#                                            args=None)
#             cst_restricted.append(new_c)
#             
#         # build supporting hyperplanes of objective functions
#         new_o_name = full_pb.objective.name + '_linearized_ite#' + str(iter_nb)
#         new_o = full_pb.objective.linear_approximation(xvect,
#                                                        name=new_o_name,
#                                                        f_type=MDOFunction.TYPE_OBJ,
#                                                        #expr=f"{f.name}(%s)",
#                                                        args=None)
# 
#         # add linearizations as primal pb constraints
#         primal_pb.add_constraint(new_o, cstr_type=MDOFunction.TYPE_INEQ)
#         for c in cst_restricted:
#             primal_pb.add_constraint(c, cstr_type=MDOFunction.TYPE_INEQ)        





#     def _store_x_indices_by_type(self, dspace, vtype):
#         ''' get integer variables indices in 
#         xvect associated to the provided design space
#         '''
#         ind_dict = self._compute_xvect_indices(dspace)
#         int_indices = array([], dtype=int32)
#         for v in ind_dict:
#             if len(dspace.get_type(v)) > 1:
#                 msg = 'The design variable <%s> has several types instead of one for all components.\n' %v
#                 msg += '(different types for each component of the variable is not handled for now)'
#                 raise ValueError(msg)
#             if dspace.get_type(v) == [vtype]: # pylint: disable=E0602
#                 i_min, i_max = ind_dict[v]
#                 curr_indices = arange(i_min, i_max + 1)
#                 int_indices = append(int_indices, curr_indices)
#         return int_indices
#
#
#     def get_integer_varsize_dict(self):
#         '''
#         '''
#         for v in self.int_var_names:
#             if len(dspace.get_type(v)) > 1:
#                 msg = 'The design variable <%s> has several types instead of one for all components.\n' %v
#                 msg += '(different types for each component of the variable is not handled for now)'
#                 raise ValueError(msg)
#             if dspace.get_type(v) == [vtype]: # pylint: disable=E0602
                