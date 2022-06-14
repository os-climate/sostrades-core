"""
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
"""
import unittest

from gemseo.algos.design_space import DesignSpace

from numpy import array, ones
from numpy.testing import assert_array_equal, assert_almost_equal

from pandas import DataFrame
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for optimization scenario
"""


class TestSoSOptimScenario(unittest.TestCase):
    """SoSOptimScenario test class for the Pymoo plugin.

    Most of the tests use the Knapsack Problem.
    """

    def setUp(self):
        # Names (for the Knapsack Problem).
        self.study_name = "optim"
        self.scenario_name = "KnapsackOptimScenario"
        self.discipline_name = "knapsack_problem"

        # Namespaces (for the Knapsack Problem).
        self.study_level = f"{self.study_name}"
        self.scenario_level = f"{self.study_name}.{self.scenario_name}"
        self.discipline_level = f"{self.study_name}.{self.scenario_name}.{self.discipline_name}"

        # Paths.
        self.repo = "sos_trades_core.sos_processes.test"
        self.process_name = "test_pymoo"

        # Knapsack variables' name.
        self.objective_name = "value"
        self.constraint_items_name = "excess_items"
        self.constraint_weight_name = "excess_weight"

        # Integer options.
        self.integer_operators = {
            "sampling": "int_lhs",
            "crossover": "int_sbx",
            "mutation": ["int_pm", dict(prob=1.0, eta=3.0)],
        }
        self.integer_options = {"normalize_design_space": False, "stop_crit_n_x": 99}

        # Mixed variables options.
        self.mixed_operators = {
            "sampling": dict(integer="int_random", float="real_random"),
            "crossover": dict(
                integer=["int_sbx", dict(prob=1.0, eta=3.0)], float=["real_sbx", dict(prob=1.0, eta=3.0)],
            ),
            "mutation": dict(int=["int_pm", dict(eta=3.0)], float=["real_pm", dict(eta=3.0)])
        }

        self.set_up_design_space = lambda n_items, lb, ub: {
            "variable": ["x"],
            "value": [[1.] * n_items],
            "lower_bnd": [[lb] * n_items],
            "upper_bnd": [[ub] * n_items],
            "enable_variable": [True],
            "activated_elem": [[True] * n_items],
            "variable_type": [[DesignSpace.INTEGER] * n_items]
        }

    def test_so_ga_weight_constraint(self):
        """Test GA algo using the Knapsack problem with weight constraint only."""

        # Knapsack characteristics.
        values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
        weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
        capacity_weight = 269.
        n_items = len(values)

        solution_x = array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        solution_f = sum(solution_x * values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "value",
            f"{self.scenario_level}.maximize_objective": True,
            f"{self.scenario_level}.ineq_constraints": ["excess_weight"],
            f"{self.scenario_level}.algo": "PYMOO_GA",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options,
                **self.integer_operators,
                "max_gen": 2 ** 11,
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_weight": capacity_weight,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        assert_array_equal(solution_x, optim_scenario.optimization_result.x_opt)
        self.assertEqual(solution_f, optim_scenario.optimization_result.f_opt)

    def test_so_nsga2_items_constraint(self):
        """Test NSGA2 algo using the Knapsack problem with items constraint only."""

        # Knapsack characteristics.
        values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
        weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
        capacity_items = 5
        n_items = len(values)

        solution_x = array([1, 0, 0, 0, 0, 1, 0, 1, 1, 1])
        solution_f = sum(solution_x * values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "value",
            f"{self.scenario_level}.maximize_objective": True,
            f"{self.scenario_level}.ineq_constraints": ["excess_items"],
            f"{self.scenario_level}.algo": "PYMOO_NSGA2",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options, **self.integer_operators, "max_gen": 20
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_items": capacity_items,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        assert_array_equal(solution_x, optim_scenario.optimization_result.x_opt)
        self.assertEqual(solution_f, optim_scenario.optimization_result.f_opt)

    def test_so_nsga2_weight_constraint(self):
        """Test NSGA2 algo using the Knapsack problem with weight constraint only."""

        # Knapsack characteristics.
        values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
        weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
        capacity_weight = 269.
        n_items = len(values)

        solution_x = array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        solution_f = sum(solution_x * values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "value",
            f"{self.scenario_level}.maximize_objective": True,
            f"{self.scenario_level}.ineq_constraints": ["excess_weight"],
            f"{self.scenario_level}.algo": "PYMOO_NSGA2",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options, **self.integer_operators, "max_gen": 20
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_weight": capacity_weight,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        assert_array_equal(solution_x, optim_scenario.optimization_result.x_opt)
        self.assertEqual(solution_f, optim_scenario.optimization_result.f_opt)

    def test_so_nsga3_weight_constraint(self):
        """Test NSGA3 algo using the Knapsack problem with weight constraint only."""

        # Knapsack characteristics.
        values = array([44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75, 29, 75, 63])
        weights = array([92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58])
        capacity_weight = 878.
        n_items = len(values)

        solution_x = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        solution_f = sum(solution_x * values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "value",
            f"{self.scenario_level}.maximize_objective": True,
            f"{self.scenario_level}.ineq_constraints": ["excess_weight"],
            f"{self.scenario_level}.algo": "PYMOO_NSGA3",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options,
                **self.integer_operators,
                "max_gen": 2 ** 11,
                "crossover": ["int_sbx", dict(eta=3, prob=1.0)],
                "mutation": ["int_pm", dict(prob=1.0, eta=3)],
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_weight": capacity_weight,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        assert_array_equal(solution_x, optim_scenario.optimization_result.x_opt)
        self.assertEqual(solution_f, optim_scenario.optimization_result.f_opt)

    def test_so_unsga3_weight_constraint(self):
        """Test UNSGA3 algo using the Knapsack problem with weight constraint only."""

        # Knapsack characteristics.
        values = array([44, 46, 90, 72, 91, 40, 75, 35, 8, 54, 78, 40, 77, 15, 61, 17, 75, 29, 75, 63])
        weights = array([92, 4, 43, 83, 84, 68, 92, 82, 6, 44, 32, 18, 56, 83, 25, 96, 70, 48, 14, 58])
        capacity_weight = 878.
        n_items = len(values)

        solution_x = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1])
        solution_f = sum(solution_x * values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "value",
            f"{self.scenario_level}.maximize_objective": True,
            f"{self.scenario_level}.ineq_constraints": ["excess_weight"],
            f"{self.scenario_level}.algo": "PYMOO_UNSGA3",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options,
                **self.integer_operators,
                "max_gen": 2 ** 11,
                "ref_points": array([[1.0], [2.0], [3.0]]),
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_weight": capacity_weight,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        assert_array_equal(solution_x, optim_scenario.optimization_result.x_opt)
        self.assertEqual(solution_f, optim_scenario.optimization_result.f_opt)

    def test_mo_nsga2(self):
        """Test NSGA2 algo using a Multi Objective Knapsack problem."""

        # Knapsack characteristics.
        values = array([55.0, 10.0, 47.0, 5.0, 4.0, 50.0, 8.0, 61.0, 85.0, 87.0])
        weights = array([95.0, 4.0, 60.0, 32.0, 23.0, 72.0, 80.0, 62.0, 65.0, 46.0])
        capacity_items = 10
        capacity_weight = 269.0
        n_items = len(values)

        # Set-up design space.
        design_space = self.set_up_design_space(n_items, 0., 1.)

        # Set-up scenario.
        scenario_settings = {
            f"{self.scenario_level}.design_space": DataFrame(design_space),
            f"{self.scenario_level}.formulation": "DisciplinaryOpt",
            f"{self.scenario_level}.objective_name": "items_and_-value",
            # f"{self.scenario_level}.objective_name": ["nvalue", "n_items"],
            f"{self.scenario_level}.maximize_objective": False,
            f"{self.scenario_level}.ineq_constraints": ["excess_items", "excess_weight"],
            f"{self.scenario_level}.algo": "PYMOO_NSGA2",
            f"{self.scenario_level}.max_iter": 800,
            f"{self.scenario_level}.algo_options": {
                **self.integer_options, **self.integer_operators, "max_gen": 20
            },
            f"{self.discipline_level}.x": ones(n_items, dtype=int),
            f"{self.discipline_level}.items_value": values,
            f"{self.discipline_level}.items_weight": weights,
            f"{self.discipline_level}.capacity_items": capacity_items,
            f"{self.discipline_level}.capacity_weight": capacity_weight,
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(repo=self.repo, mod_id=self.process_name)
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{self.study_name}.{self.scenario_name}")[0]
        pareto = optim_scenario.optimization_result.pareto

        # Known solutions (anchor points).
        anchor_x = array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        anchor_f = array([6, -295.0])
        assert anchor_x in pareto.set
        assert anchor_f in pareto.front

        anchor_x = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        anchor_f = array([0, -0.0])
        assert anchor_x in pareto.set
        assert anchor_f in pareto.front

        # Best compromise.
        comp_x = array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1])
        comp_f = array([5, -293.0])
        assert comp_x in pareto.set
        assert comp_f in pareto.front

    def test_so_nsga2_mixed_variables_linear(self):
        """Test NSGA2 algo using the DiscMixedOptLinearFeasible problem."""

        # Set-up design space.
        design_space = {
            "variable": ["x1", "x2"],
            "value": [[5.], [5.]],
            "lower_bnd": [[0], [0.]],
            "upper_bnd": [[10], [10.]],
            "enable_variable": [True, True],
            "activated_elem": [[True], [True]],
            "variable_type": [[DesignSpace.INTEGER], [DesignSpace.FLOAT]]
        }

        # Set-up scenario.
        scenario_level = f"{self.study_name}.MixedOptimScenario"
        discipline_level = f"{scenario_level}.MixedCoupling.DiscMixedOpt"
        scenario_settings = {
            f"{scenario_level}.design_space": DataFrame(design_space),
            f"{scenario_level}.formulation": "DisciplinaryOpt",
            f"{scenario_level}.objective_name": "obj",
            f"{scenario_level}.maximize_objective": False,  # There is a "-" sign in the objective function !
            f"{scenario_level}.ineq_constraints": ["constr"],
            f"{scenario_level}.algo": "PYMOO_NSGA2",
            f"{scenario_level}.max_iter": 800,
            f"{scenario_level}.algo_options": {
                **self.integer_options, **self.mixed_operators, "max_gen": 20
            },
            f"{discipline_level}.x1": array([1], dtype=int),
            f"{discipline_level}.x2": array([1.], dtype=float),
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(
            repo="sos_trades_core.sos_processes.test", mod_id="test_mixedopt_linear"
        )
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{scenario_level}")[0]
        assert_almost_equal(array([5, 3.1]), optim_scenario.optimization_result.x_opt, decimal=3)
        self.assertAlmostEqual(-30.5, optim_scenario.optimization_result.f_opt, delta=1e-2)

    def test_so_nsga2_mixed_variables_non_linear(self):
        """Test NSGA2 algo using the DiscMixedOptNonLinearFeasible problem."""

        # Set-up design space.
        design_space = {
            "variable": ["x1", "x2"],
            "value": [[100], [100.]],
            "lower_bnd": [[0], [0]],
            "upper_bnd": [[200], [200]],
            "enable_variable": [True, True],
            "activated_elem": [[True], [True]],
            "variable_type": [[DesignSpace.INTEGER], [DesignSpace.INTEGER]]
        }

        # Set-up scenario.
        scenario_level = f"{self.study_name}.MixedOptimScenario"
        discipline_level = f"{scenario_level}.MixedCoupling.DiscMixedOpt"
        scenario_settings = {
            f"{scenario_level}.design_space": DataFrame(design_space),
            f"{scenario_level}.formulation": "DisciplinaryOpt",
            f"{scenario_level}.objective_name": "obj",
            f"{scenario_level}.maximize_objective": False,
            f"{scenario_level}.ineq_constraints": ["constr"],
            f"{scenario_level}.algo": "PYMOO_NSGA2",
            f"{scenario_level}.max_iter": 1000,
            f"{scenario_level}.algo_options": {
                **self.integer_options, **self.integer_operators, "max_gen": 20
            },
            f"{discipline_level}.x1": array([100], dtype=int),
            f"{discipline_level}.x2": array([100], dtype=int),
        }

        # Set-up execution engine.
        exec_eng = ExecutionEngine(self.study_name)
        opt_builder = exec_eng.factory.get_builder_from_process(
            repo="sos_trades_core.sos_processes.test", mod_id="test_mixedopt_nonlinear"
        )
        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        exec_eng.configure()

        # Set-up study.
        exec_eng.load_study_from_input_dict(scenario_settings)
        exec_eng.configure()

        # Execute.
        exec_eng.execute()

        # Retrieve scenario to check the result.
        optim_scenario = exec_eng.dm.get_disciplines_with_name(f"{scenario_level}")[0]
        assert_array_equal(array([4, 2]), optim_scenario.optimization_result.x_opt)
        self.assertEqual(16., optim_scenario.optimization_result.f_opt)


if "__main__" == __name__:
    cls = TestSoSOptimScenario()
    cls.setUp()
    cls.test_so_nsga2_weight_constraint()
