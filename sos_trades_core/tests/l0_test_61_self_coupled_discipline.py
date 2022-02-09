'''
Created on 25/08/2021
'''
import unittest

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TestSelfCoupledDiscipline(unittest.TestCase):

    def setUp(self):
        self.name = 'SoSDisc'
        self.ee = ExecutionEngine('Test')
        self.factory = self.ee.factory

    def test_01_execute_self_coupled_discipline(self):
        model_name = 'SelfCoupledDiscipline'

        # Get Disc1 builder using path
        mod_path = 'sos_trades_core.sos_wrapping.test_discs.disc_self_coupled.SelfCoupledDiscipline'
        builder = self.factory.get_builder_from_module(
            model_name, mod_path)

        # Set builder in factory and configure
        self.factory.set_builders_to_coupling_builder(builder)
        self.ee.load_study_from_input_dict({})
        root_coupling = self.ee.dm.get_disciplines_with_name('Test')[0]
        print(root_coupling.mdo_chain)

        self.ee.execute()

        # Get output value
        x = self.ee.dm.get_value('Test.SelfCoupledDiscipline.x')

        # Test output value
        self.assertEqual(x, 0.5)
        self.assertEqual(type(root_coupling.mdo_chain.disciplines[0]).__name__, 'SelfCoupledDiscipline')

        self.ee.load_study_from_input_dict({'Test.authorize_self_coupled_disciplines': True})
        print(root_coupling.mdo_chain)
        self.ee.execute()

        # Get output value
        x = self.ee.dm.get_value('Test.SelfCoupledDiscipline.x')

        # Test output value
        self.assertEqual(x, 0.0)
        self.assertEqual(type(root_coupling.mdo_chain.disciplines[0]).__name__, 'MDAJacobi')
