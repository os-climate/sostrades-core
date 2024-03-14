import logging
import unittest

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class MyCustomWrapTest(unittest.TestCase):
    def setUp(self):
        self.name = 'MyCustomWrapTest'
        self.ee = ExecutionEngine(self.name)
        self.wrap_name = "MyCustomWrap"
        self.wrap_path = f"sostrades_core.sos_wrapping.tuto_wrap.my_custom_wrap.{self.wrap_name}"

    def test_01_wrap_execution(self):
        # Get the wrap builder
        wrap_builder = self.ee.factory.get_builder_from_module(self.wrap_name, self.wrap_path )
        
        # Set it to be built directly under the root coupling node
        self.ee.factory.set_builders_to_coupling_builder(wrap_builder)

        # associate namespaces
        self.ee.ns_manager.add_ns('ns_one', self.name)

        # Configure the discipline
        self.ee.configure()

        # Display the treeview
        logging.info(self.ee.display_treeview_nodes(display_variables=True))

        a, b, x = 1.0, 2.0, 1.0
        values_dict = {
            self.name + ".x": a,
            self.name + ".MyCustomWrap.a": b,
            self.name + ".MyCustomWrap.b": x,
        }

        # Load input values for the study
        self.ee.load_study_from_input_dict(values_dict)

        # Execute
        self.ee.execute()

        # Check status
        for disc_id in self.ee.dm.disciplines_dict.keys():
            self.assertEqual(self.ee.dm.get_discipline(disc_id).status, 'DONE')

        # Check output
        y = self.ee.dm.get_value(self.name + ".y")
        self.assertEqual(y, a * x + b)
