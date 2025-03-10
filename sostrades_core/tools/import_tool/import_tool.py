'''
Copyright 2025 Capgemini

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
from __future__ import annotations

import inspect
import os
from importlib import import_module
from typing import Any


def get_class_from_path(class_path: str) -> type[Any]:
    """Get a class object from its class path i.e. module_poth.ClassName."""
    module_struct_list = class_path.split('.')
    import_name = '.'.join(module_struct_list[:-1])
    m = import_module(import_name)
    return getattr(m, module_struct_list[-1])


def get_module_class_path(class_name, folder_list):
    """
    Return the module path of a class in a list of directories
    Return the first found for now ..
    """
    module_class_path = None
    for folder in folder_list:
        # Get the module of the folder
        try:
            module = import_module(folder)
            folder_path = os.path.dirname(module.__file__)
        except:
            raise Warning(f'The folder {folder} is not a module')

        # Get all files in the folder_path
        file_list = os.listdir(folder_path)
        # Find all submodules in the path
        sub_module_list = [
            import_module('.'.join([folder, file.split('.')[0]]))
            for file in file_list
        ]

        for sub_module in sub_module_list:
            # Find all members of each submodule which are classes
            # belonging to the sub_module
            class_list = [
                value
                for value, cls in inspect.getmembers(sub_module)
                if inspect.isclass(getattr(sub_module, value))
                   and cls.__module__ == sub_module.__name__
            ]
            # CHeck if the following class is in the list
            if class_name in class_list:
                module_class_path = '.'.join(
                    [sub_module.__name__, class_name])
                break
        else:
            continue
        break

    return module_class_path
