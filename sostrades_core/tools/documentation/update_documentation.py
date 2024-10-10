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

import ast
import importlib
import importlib.util
import inspect
import logging
import os
import re
import codecs
import astor
from docstring_to_markdown import convert

from sostrades_core.tools.gen_ai.gen_engine_services import GenerativeEngineService


class DocGenerator():
    """Generates or updates documentation"""

    def __init__(self):
        self.markdown_file = None #absolute path to the markdown file where the documentation is saved
        self.pythonfile = None #absolute path to the source code where docgenerator methods are applied
        self.class_name = None #name of the python class
        self.markdown_str = None #documentation content in markdown format
        self.discipline_class = None #if dealing with the documentation of a discipline

    @staticmethod
    def load_module(module_abs_path: [str]):
        """
        Loads a python module defined by the absolute path of its file
        Args:
            module_abs_path
        Returns:
            module
        """
        module_name = os.path.splitext(os.path.basename(module_abs_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
    def get_discipline_class(self):
        """
        get the discipline class defined by self.class_name in the self.pythonfile
        to access its attributes
        Args:
            python_file: must be the absolute path
        """
        if self.class_name is None:
            logging.info('No discipline class defined')
            self.discipline_class = None

        elif self.pythonfile is not None:
            # Load the module from the file
            module = self.load_module(self.pythonfile)
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
        markdown_str = "# Model data"
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

        #markdown_str = mistune.markdown(docstring)
        markdown_str = convert(docstring, style="google")

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
            section_to_replace: Name of the section to replace. Assumes that the section starts with one or several # sign
            new_content: the new content that replaces the initial one
        Returns:
            updated_markdown_content
        """
        # the section to be replaced starts at section_to_replace and ends when the same number of # signs are encountered
        match = re.match(r'^#+', section_to_replace) # Find all consecutive '#' characters at the start of the string
        next_section_must_start_with = None
        if match:
            next_section_must_start_with = match.group() + " "
        else:
            raise ValueError(f"A section to replace must be a markdown section starting with the # sign. Here, section_to_replace = {section_to_replace}")
        # because of using codecs, a robust search method must be used
        match = False
        lines = initial_markdown_str.split('\n')
        in_section = False
        modified_lines = []
        for line in lines:
            if line.strip() == section_to_replace:
                in_section = True
                match = True
                modified_lines.extend([new_content])
            elif in_section and (line.startswith(next_section_must_start_with) or not line):
                in_section = False
                modified_lines.append(line)
            elif not in_section:
                modified_lines.append(line)

            updated_markdown_content = '\n'.join(modified_lines)

        if not match:
            updated_markdown_content = section_to_replace + "\n" + new_content + "\n" + initial_markdown_str

        return updated_markdown_content


    def write_markdown_file(self, markdown_str:[str]):
        """
        writes a markdown file
        Args:
            markdown_str: content of the markdown file
        """
        with codecs.open(self.markdown_file, "w", encoding='utf-8') as f:
            f.write(markdown_str)

    def extract_markdown_from_method(self, method_name: [str], api_key:[str]) -> [str]:
        """
        """
        method = getattr(self.discipline_class, method_name)
        method_code = inspect.getsource(method)
        url = "https://api.generative.engine.capgemini.com"
        workspace_ID ="ad32a1ad-858a-48a1-b071-de8026990577" # workspace generate_docstrings dedicated to doscstring generation https://generative.engine.capgemini.com/studio/rag/workspaces/ad32a1ad-858a-48a1-b071-de8026990577
        genai = GenerativeEngineService(url, api_key, session_name=None, workspace_id=workspace_ID)
        generic_prompt = r"You will be analyzing the provided source code for a class method. Your task is to:" \
                         r"1. Read through the source code carefully and understand the conditions under which the dynamic_inputs and dynamic_outputs dictionaries are updated." \
                         r"2. Provide a short description in markdown format explaining these conditions, like this:" \
                         r"<description>" \
                         r"[A brief explanation of when the dynamic_inputs and dynamic_outputs dictionaries are updated based on the input values.]" \
                         r"</description>" \
                         r"3. Then, list all the possible keys that could exist in the dynamic_inputs dictionary under a 'Dynamic Inputs' section, with each key on a new bullet point line, like this:" \
                         r"### Dynamic Inputs" \
                         r"- [key1]" \
                         r"- [key2]" \
                         r"- ..." \
                         r"4. Next, list all the possible keys that could exist in the dynamic_outputs dictionary under a 'Dynamic Outputs' section, with each key on a new bullet point line, like this: " \
                         r"### Dynamic Outputs" \
                         r"- [key1]" \
                         r"- [key2]" \
                         r"- ..." \
                         r"5. Make sure to format the entire output using valid markdown syntax." \
                         r"Here is the source code you need to analyze:" \
                         r"{$PYTHON_CODE}" \
                         r"Please begin your response, and make sure to follow the formatting instructions carefully."
        prompt = generic_prompt.replace("$PYTHON_CODE", method_code)
        [hw, engine] = ["bedrock", "anthropic.claude-v2"] #["azure", "openai.gpt-4"]
        answ = genai.run(prompt, hw, engine)
        # genai can add other comments in various format depending on the engine used => extract the docstring only
        markdown_str = answ.data.content

        return markdown_str
    def generate_docstring(self, method_name:[str], api_key:[str]) -> [str]:
        """
        Generates automatically the docstring of a method or a class following rules/prompt generated
        with https://generative.engine.capgemini.com/studio/chatbot/prompt-playground and put in workspaceid
        Therefore, only the python code needs be provided
        NB: using gpt4, it seems like we are limited in terms of token available. Therefore, the method may crash from time
        to time. Workaround: either relaunch after 30s or try with antropic-claude engine
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
        generic_prompt = r"Your task is to generate a Google-style docstring for a given Python method. Here are the steps to follow:" \
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
        [hw, engine] = ["bedrock", "anthropic.claude-v2"] #["azure", "openai.gpt-4"]
        answ = genai.run(prompt, hw, engine)
        # genai can add other comments in various format depending on the engine used => extract the docstring only
        content_str = answ.data.content
        pattern = r'<docstring>([\s\S]*?)</docstring>'  # pattern also used by claude
        matches = re.findall(pattern, content_str)
        if not matches:
            pattern = r'"""([\s\S]*?)"""'  # pattern for gpt_4
            matches = re.findall(pattern, content_str)
        if matches:  # The docstring is the first (and should be only) match
            docstring = matches[0].strip()
        else:
            logging.info("Could not extract docstring from the gen ai answer. Review the pattern used")
            docstring = """Not generated"""

        return docstring

    def generate_markdown_str_of_model(self):
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

    def generate_markdown_file_of_model(self):
        """
        Generates a markdown file for a python model
        This method is an alternative to the combined methods generate_markdown_str_of_model and write_markdown_file
        """
        os.system(f"pydoc-markdown -I {os.path.dirname(self.pythonfile)} -m {self.pythonfile.split(os.sep)[-1].split('.py')[0]} --render-toc > {self.markdown_file}")

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

        # Updating the discipline's markdown DESC_IN and DESC_OUT including the dynamic input and outputs
        section_to_replace = "# Model data"
        self.get_discipline_class()
        markdown_str = self.convert_discipline_desc_to_markdown()

        if hasattr(self.discipline_class, "setup_sos_disciplines"):
            markdown_str_dynamic_in_out = self.extract_markdown_from_method("setup_sos_disciplines", api_key)
            markdown_str += "\n" + markdown_str_dynamic_in_out

        # markdown may contain symbols that require treatment with codecs
        with codecs.open(self.markdown_file, "r", encoding='utf-8') as f:
            markdown_initial = f.read()
        updated_markdown_content = self.update_markdown_section(markdown_initial,
                                                                section_to_replace,
                                                                markdown_str)
        self.write_markdown_file(updated_markdown_content)

        # generating the model markdown for all the methods of the model class
        name_split = markdown_file_path.split('.')
        self.markdown_file = name_split[0] + '_for_dev.' + name_split[1]
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
            logging.info(f"using genai to generate docstring for method {method_name}")
            docstring_dict[method_name] = self.generate_docstring(method_name, api_key)
        for method_name in methods:
            self.update_code_docstring(method_name, docstring_dict[method_name])
        # since the python file of the model has changed, it must be reloaded
        module = self.load_module(self.pythonfile)
        self.get_discipline_class()
        self.generate_markdown_file_of_model()

if '__main__' == __name__:
    """
    Example of execution of the run method that will update the damage model discipline markdown with the DESC_IN and DESC_OUT, 
    generate the docstring of the damage_model and create its markdown
    """
    doc = DocGenerator()
    model = "macroeconomics"
    platform_path_abs = os.path.dirname(os.path.abspath(__file__)).split(os.sep + "platform")[0]
    if model == "damage":
        discipline_py_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\damagemodel\damagemodel_discipline.py")
        discipline_class_name = "DamageDiscipline"
        model_py_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\core\core_witness\damage_model.py")
        model_class_name = "DamageModel"
        markdown_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\damagemodel\documentation\damagemodel_discipline.md")
    elif model == "macroeconomics":
        discipline_py_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\macroeconomics\macroeconomics_discipline.py")
        discipline_class_name = "MacroeconomicsDiscipline"
        model_py_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\core\core_witness\macroeconomics_model_v1.py")
        model_class_name = "MacroEconomics"
        markdown_file_path = os.path.join(platform_path_abs, r"models\witness-core\climateeconomics\sos_wrapping\sos_wrapping_witness\macroeconomics\documentation\macroeconomics_discipline.md")
    api_key = "E2QSy0VXlI7MaalEcc6z98hCyUT7UOmn1IfxXI1o"
    doc.run(discipline_py_file_path,
            discipline_class_name,
            model_py_file_path,
            model_class_name,
            markdown_file_path,
            api_key)