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
import os
import shutil
import time


def rmtree_safe(path: str, retry: int = 100):
    """
    Remove a directory tree, but wait until it is really gone.
    Useful to fix some issues with shutil.rmtree.
    
    Parameters:
    path (str): The directory tree to remove.
    retry (int): The maximum number of loop iterations to check if the directory is removed. Default is 20.
    """
    shutil.rmtree(path)

    attempts = 0
    while os.path.exists(path) and attempts < retry:
        time.sleep(0.05)
        attempts += 1

    if os.path.exists(path):
        raise Exception(f"Failed to remove the directory {path} after {retry} retries.")
