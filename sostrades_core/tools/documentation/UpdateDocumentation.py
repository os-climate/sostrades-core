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
import importlib.util
import os
import logging
import mistune
import re
import inspect
import ast
import astor
from sostrades_core.tools.gen_ai.gen_engine_services import GenerativeEngineService


class DocGenerator():
    """Generates or updates documentation"""

    def __init__(self):
        self.markdown_file = None #path to the markdown file where the documentation is saved
        self.pythonfile = None #path to the source code where docgenerator methods are applied
        self.class_name = None #name of the python class
        self.markdown_str = None #documentation content in markdown format
        self.discipline_class = None #if dealing with the documentation of a discipline

    def get_discipline_class(self):
        """
        get the discipline class defined by self.class_name in the self.pythonfile
        to access its attributes
        """
        if self.class_name is None:
            logging.info('No discipline class defined')
            self.discipline_class = None

        elif self.pythonfile is not None:
            module_path = os.path.splitext(self.pythonfile)[0].replace(os.sep, '.')  # convert to python module format
            module = importlib.import_module(module_path)
            self.discipline_class = getattr(module, self.class_name)
        else:
            self.discipline_class = None

    def convert_discipline_desc_to_markdown(self) -> [str]:
        """
        gets the DESC_IN and DESC_OUT of a discipline and generates a string in markdown format that contains:
        variable name, unit, type, definition
        Args:
        Returns:
            markdown_str: DESC_IN and DESC_OUT in markdown format
        """
        if self.discipline_class is None:
            self.get_discipline_class()
            logging.debug(f"extracting DESC_IN and DESC_OUT of discipline {self.discipline_class.__name__}")
        markdown_str = "# Model Data"
        for desc_type in ["DESC_IN", "DESC_OUT"]:
            do_process = False
            if desc_type == "DESC_IN":
                markdown_str += "\n ## Static inputs"
                if hasattr(self.discipline_class, "DESC_IN"):
                    do_process = True
                    DESC = self.discipline_class.DESC_IN
                else:
                    markdown_str += "\n None"
            else:
                markdown_str += "\n ## Static outputs"
                if hasattr(self.discipline_class, "DESC_OUT"):
                    do_process = True
                    DESC = self.discipline_class.DESC_OUT
                else:
                    markdown_str += "\n None"
            if do_process:
                for var, val in DESC.items():
                    markdown_str += f"\n- {var}"
                    for info in ['unit', 'type', 'description']:
                        if info in val.keys():
                            markdown_str += f", {info}={val[info]}"

        return markdown_str

    def convert_class_method_docstring_to_markdown(self, method_name:[str]) -> [str]:
        """
        Convert the docstring of the class.method_name into markdown format
        If the method does not exist, returns "dynamic variables: N/A"
        Args:
            method_name: the name of the method which docstring needs be extracted.
                If method_name = 'class', it extracts docstring of the class description
        Returns:
            markdown_str: the docstring of the setup_sos_disciplines in markdown format

        """
        if self.discipline_class is None:
            self.get_discipline_class()
            logging.debug(f"Generating markdown from docstring of {self.discipline_class.__name__}.{method_name}")
        if method_name == 'class':
            docstring = self.discipline_class.__doc__
        else:
            method = getattr(self.discipline_class, method_name)
            docstring = method.__doc__

        markdown_str = mistune.markdown(docstring)

        return markdown_str

    def update_markdown_section(self, initial_markdown_str:[str],
                                        section_to_replace:[str],
                                        new_content:[str]) -> [str]:
        """
        replaces the content of a section within a markdown
        if section does not exist, it adds it at the beginning of the markdown
        Assumes that a section starts with a title #SOMETHING and ends at the beginning of another section that also 
        starts with a title #SOMETHINGELSE
        Args:
            initial_markdown_str: documentation content in markdown syntax (not the file, the actual content)
            section_to_replace: Name of the section to replace
            new_content: the new content that replaces the initial one
        Returns:
            updated_markdown_content
        """
        section_pattern = f"{section_to_replace}" + r'\n(.*?)\n#'
        match = re.search(section_pattern, initial_markdown_str, re.DOTALL)

        if match:
            content = match.group(1)

            # Replace content
            updated_markdown_content = re.sub(section_pattern, f"{section_to_replace}\n{new_content}\n#", initial_markdown_str, flags=re.DOTALL)

        else:
            updated_markdown_content = section_to_replace + "\n" + new_content + "\n" + initial_markdown_str

        return updated_markdown_content


    def write_markdown_file(self, markdown_str:[str]):
        """
        writes a markdown file
        Args:
            markdown_str: content of the markdown file
        """
        with open(self.markdown_file, "w") as f:
            f.write(markdown_str)

    def generate_docstring(self, method_name:[str], api_key:[str]) -> [str]:
        """
        Generates automatically the docstring of a method or a class following rules/prompt generated
        with https://generative.engine.capgemini.com/studio/chatbot/prompt-playground and put in workspaceid
        Therefore, only the python code needs be provided
        Args:
            method_name: name of the specific method that requires a new docstring
            api_key: general api key generated by Capgemini for a given account

        Returns:
            Docstring
        """
        method = getattr(self.discipline_class, method_name)
        method_code = inspect.getsource(method)
        url = "https://api.generative.engine.capgemini.com"
        workspace_ID ="ad32a1ad-858a-48a1-b071-de8026990577" # workspace generate_docstrings dedicated to doscstring generation https://generative.engine.capgemini.com/studio/rag/workspaces/ad32a1ad-858a-48a1-b071-de8026990577
        genai = GenerativeEngineService(url, api_key, session_name=None, workspace_id=workspace_ID)
        generic_prompt = r"Your task is to generate a Google-style do cstring for a given Python method. Here are the steps to follow:" \
                         r"1. I will provide you with a Python method code in the following format:" \
                         r"<python_code>{$PYTHON_CODE}</python_code>" \
                         r"2. Remove any existing docstring from the provided Python code." \
                         r"3. Generate a new Google-style docstring for the method, following these guidelines:" \
                         r"   - Start with a brief one-line summary of what the method does." \
                         r"   - Leave a blank line after the summary." \
                         r"   - Add a more detailed description of the method's purpose and behavior." \
                         r"   - Leave another blank line." \
                         r"   - List the arguments of the method using the following format:" \
                         r"      Args:" \
                         r"        arg1 (type): Description of arg1." \
                         r"        arg2 (type): Description of arg2." \
                         r"        ..." \
                         r"   - If the method has a return value, add a 'Returns' section:" \
                         r"      Returns:" \
                         r"        type: Description of the return value." \
                         r"   - If the method raises any exceptions, add a 'Raises' section:" \
                         r"      Raises:" \
                         r"        Exception1: Description of when Exception1 is raised." \
                         r"        Exception2: Description of when Exception2 is raised." \
                         r"        ..." \
                         r"   - If you need to provide examples, add an 'Examples' section with code samples." \
                         r"4. Do not include any other information or comments beyond the docstring." \
                         r"5. Ensure that the docstring follows the Google style guide for Python docstrings." \
                         r"Your output should exclusively contain the generated docstring, without any additional text or code. Here's an example of the expected output format:" \
                         r"<docstring>" \
                         r"One line summary of the method." \
                         r"Detailed description of the method's purpose and behavior." \
                         r"Args:" \
                         r"    arg1 (type): Description of arg1." \
                         r"    arg2 (type): Description of arg2." \
                         r"Returns:" \
                         r"    type: Description of the return value." \
                         r"Raises:" \
                         r"    Exception1: Description of when Exception1 is raised." \
                         r"    Exception2: Description of when Exception2 is raised." \
                         r"Examples:" \
                         r"    Example usage of the method." \
                         r"</docstring>" \
                         r"Remember, your output should be exclusively the generated Google-style docstring for that method."
        prompt = generic_prompt.replace("$PYTHON_CODE", method_code)
        answ = genai.run(prompt, "azure", "openai.gpt-4")
        # genai can add other comments => extract the docstring only
        content_str = answ.data.content
        pattern = r'"""([\s\S]*?)"""'
        matches = re.findall(pattern, content_str)
        if matches:  # The docstring is the first (and should be only) match
            docstring = matches[0].strip()
        else:
            docstring = """Not generated"""

        return docstring

    def generate_markdown_of_model(self):
        """
        Generates a markdown string for a python model. Assumes that Google style dosctring is used for the method and class
        docstring
        """
        markdown_str = f"# {self.discipline_class.__name__}\n\n"
        markdown_str += self.discipline_class.__doc__ + "\n\n"

        for name, obj in inspect.getmembers(self.discipline_class):
            if inspect.isfunction(obj):
                markdown_str += f"## {name}\n\n"
                doc = obj.__doc__
                if doc is not None:
                    markdown_str += doc + "\n\n"

        return markdown_str

    def update_code_docstring(self, method_name:[str], docstring:[str]):
        """
        updates (or adds if does not exist) the docstring of a method or class
        Args:
            method_name: name of the method to update
            docstring: new docstring value
        Returns
        """
        with open(self.pythonfile, 'r') as file:
            tree = ast.parse(file.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == self.class_name:
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) and sub_node.name == method_name:
                        sub_node.body[0] = ast.Expr(ast.Str(docstring))
                        break
                break

        modified_code = astor.to_source(tree)

        with open(self.pythonfile, 'w') as file:
            file.write(modified_code)


    def run(self, discipline_py_file_path: [str],
            discipline_class_name: [str],
            model_py_file_path: [str],
            model_class_name: [str],
            markdown_file_path: [str],
            api_key: [str]):
        """
        For a discipline, updates the markdown with the DESC_IN and DESC_OUT, generates the docstring and corresponding
        markdown for the discipline model
        Args:
            discipline_py_file_path: relative path (from root) to the python file hosting the discipline
            discipline_class_name: name of the class of the discipline
            model_py_file_path: relative path (from root) to the python file hosting the model of the discipline
            model_class_name: name of the class of the model
            markdown_file_path: relative (path (from root) to the markdown file of the discipine's documentation
            api_key: general api key generated by Capgemini for a given account

        Returns:
        """
        self.markdown_file = markdown_file_path
        self.pythonfile = discipline_py_file_path
        self.class_name = discipline_class_name  # name of the python class

        # Updating the discipline's markdown DESC_IN and DESC_OUT
        section_to_replace = "## Model data"
        self.get_discipline_class()
        markdown_str = self.convert_discipline_desc_to_markdown()
        with open(self.markdown_file, "r") as f:
            markdown_initial = f.read()
        updated_markdown_content = self.update_markdown_section(markdown_initial,
                                                                section_to_replace,
                                                                markdown_str)
        self.write_markdown_file(updated_markdown_content)

        # generating the model markdown for all the methods of the model class
        self.markdown_file = markdown_file_path.replace('_disc', '_model')
        self.pythonfile = model_py_file_path
        self.class_name = model_class_name  # name of the python class
        self.get_discipline_class()
        all_attributes = dir(self.discipline_class)
        methods = [attr for attr in all_attributes if inspect.isfunction(getattr(self.discipline_class, attr))] #isfunction instead of ismethod because handling the class, not an instance
        # ToDo: understand why when self.update_code_docstring is with self.generate_docstring in the loop
        # method_code = inspect.getsource(method) returns the method1 source code when method2 is requested
        # then make one loop instead of two
        docstring_dict = {}
        for method_name in methods:
            docstring_dict[method_name] = self.generate_docstring(method_name, api_key)
        for method_name in methods:
            self.update_code_docstring(method_name, docstring_dict[method_name])
        # since the python file of the model has changed, it must be reloaded
        module = importlib.import_module(os.path.splitext(self.pythonfile)[0].replace(os.sep, '.'))
        importlib.reload(module)
        self.get_discipline_class()
        markdown_str = self.generate_markdown_of_model()
        self.write_markdown_file(markdown_str)




