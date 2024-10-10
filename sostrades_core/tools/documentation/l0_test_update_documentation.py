'''
Copyright 2024 Capgemini
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
import importlib
import inspect
import os
import textwrap
import unittest

from sostrades_core.tools.documentation.update_documentation import DocGenerator


# example of class to be used for the tests
class A:
    """This is the docstring for class A"""
    DESC_IN = {'var_in1': {'unit': 'G$', 'type': 'float', 'description': 'input var1'},
               'var_in2': {'unit': 'G$', 'type': 'float', 'description': 'input var2'},
               }
    DESC_OUT = {'var_out1': {'unit': 'G$', 'type': 'float', 'description': 'output var1'},
               'var_out2': {'unit': 'G$', 'type': 'float', 'description': 'output var2'},
               }
    def method1(self):
        """This is the docstring for method1"""
        pass
    def method2(self, x:[float]) -> [float]:
        """this is the docstring for method2"""
        y = x**2 + 1.
        return y


# Function to write the class to a file
def write_class_to_file(cls, filename):
    # Get the source code of the class
    source = inspect.getsource(cls)
    # Remove any leading indentation
    source = textwrap.dedent(source)
    with open(filename, 'w') as file:
        file.write(source)

class UpdatedDocumentation(unittest.TestCase):

    MARKDOWN_REF = "# Model Data\n ## Static inputs\n- var_in1, unit=G$, type=float, description=input var1\n- var_in2, unit=G$, type=float, description=input var2\n ## Static outputs\n- var_out1, unit=G$, type=float, description=output var1\n- var_out2, unit=G$, type=float, description=output var2"
    def test_get_discipline_class(self):
        doc = DocGenerator()
        # no class_name defined
        doc.get_discipline_class()
        self.assertEqual(doc.discipline_class, None)
        # with class defined but no python file
        doc.class_name = "MacroeconomicsDiscipline"
        doc.get_discipline_class()
        self.assertEqual(doc.discipline_class, None)
        # with pythonfile defined
        platform_path_abs = os.path.dirname(os.path.abspath(__file__)).split(os.sep + "platform")[0]
        doc.pythonfile = os.path.join(platform_path_abs, r'models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\macroeconomics\macroeconomics_discipline.py')
        doc.get_discipline_class()
        self.assertEqual(doc.discipline_class.__name__, doc.class_name)

    def test_convert_discipline_desc_to_markdown(self):
        doc = DocGenerator()
        doc.discipline_class = A
        markdown_str = doc.convert_discipline_desc_to_markdown()
        self.assertEqual(markdown_str, self.MARKDOWN_REF)

    def test_convert_class_method_docstring_to_markdown(self):
        doc = DocGenerator()
        doc.discipline_class = A
        markdown_str = doc.convert_class_method_docstring_to_markdown("method1")
        self.assertEqual(markdown_str, "<p>This is the docstring for method1</p>\n")

    def test_update_markdown_section(self):
        doc = DocGenerator()
        doc.discipline_class = A
        section_to_replace = "# Model data"
        new_content = self.MARKDOWN_REF + "\n- var_out3, unit=G$, type=float, description=output var3"
        next_section = "\n#Model description \nThis model does this and that"
        initial_markdown_str = self.MARKDOWN_REF + next_section
        updated_markdown_content = doc.update_markdown_section(initial_markdown_str,
                                                               section_to_replace,
                                                               new_content)
        assert(new_content in updated_markdown_content and next_section in updated_markdown_content)

    def test_generate_markdown_str_of_model(self):
        doc = DocGenerator()
        doc.discipline_class = A
        markdown_str = doc.generate_markdown_str_of_model()
        self.assertEqual(markdown_str, "# A\n\nThis is the docstring for class A\n\n## method1\n\nThis is the docstring for method1\n\n## method2\n\nthis is the docstring for method2\n\n")
    def test_write_markdown_file(self):
        doc = DocGenerator()
        doc.discipline_class = A
        doc.markdown_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_write_markdown.md")
        doc.write_markdown_file(self.MARKDOWN_REF)
        with open(doc.markdown_file , "r") as f:
            markdown_read = f.read()
        self.assertEqual(markdown_read, self.MARKDOWN_REF)
        os.remove(doc.markdown_file)

    def test_generate_docstring(self):
        doc = DocGenerator()
        doc.discipline_class = A
        api_key = "E2QSy0VXlI7MaalEcc6z98hCyUT7UOmn1IfxXI1o"
        docstring = doc.generate_docstring("method2", api_key)
        print()

    def test_update_code_docstring(self):
        method_name = "method1"
        doc = DocGenerator()
        doc.pythonfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_class_A.py")
        # write initial code of class A to file
        write_class_to_file(A, doc.pythonfile)
        doc.class_name = "A"
        doc.get_discipline_class()
        # modify class A code
        new_docstring = "This is the new docstring for method1"
        doc.update_code_docstring(method_name, new_docstring)
        #reload module and discipline as the code and file have changed.
        module = doc.load_module(doc.pythonfile)
        doc.get_discipline_class()
        method = getattr(doc.discipline_class, method_name)
        self.assertEqual(method.__doc__, new_docstring)
        os.remove(doc.pythonfile)

    def test_extract_markdown_from_method(self):
        doc = DocGenerator()
        platform_path_abs = os.path.dirname(os.path.abspath(__file__)).split(os.sep + "platform")[0]
        doc.pythonfile = os.path.join(platform_path_abs,
                                               r"models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\macroeconomics\macroeconomics_discipline.py")
        doc.class_name = "MacroeconomicsDiscipline"
        doc.get_discipline_class()
        api_key = "E2QSy0VXlI7MaalEcc6z98hCyUT7UOmn1IfxXI1o"
        markdown_str = doc.extract_markdown_from_method("setup_sos_disciplines", api_key)
        print()

    def test_run(self):
        doc = DocGenerator()
        discipline_py_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_class_A.py")
        discipline_class_name = "A"
        model_py_file_path = discipline_py_file_path
        model_class_name = discipline_class_name
        markdown_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_disc.md")
        api_key = "E2QSy0VXlI7MaalEcc6z98hCyUT7UOmn1IfxXI1o"

        # markdown and temp_class do not exist => create them
        write_class_to_file(A, discipline_py_file_path)
        with open(markdown_file_path, "w") as f:
            f.write(r"#Test documentation \n")

        doc.run(discipline_py_file_path,
            discipline_class_name,
            model_py_file_path,
            model_class_name,
            markdown_file_path,
            api_key)
        for file in [discipline_py_file_path,
                     markdown_file_path,
                     markdown_file_path.replace(".md", "_for_dev.md"),
            ]:
            # since docstrings are not generated in a deterministic manner, they can't be crosschecked with a reference
            self.assertTrue(os.path.exists(file))
            os.remove(file)
