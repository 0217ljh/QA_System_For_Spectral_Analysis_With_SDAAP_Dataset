"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
import os
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default template here.
            template_name = "alpaca"
        #file_name = osp.join(".\templates", f"{template_name}.json") 
        ROOT_DIR = os.path.dirname(os.path.realpath('__file__'))
        ROOT_DIR = ROOT_DIR[:ROOT_DIR.find('new_verson') + len('new_verson')]
        project_path = ROOT_DIR + "/templates/"  # Absolute path under the project
        file_name = osp.join(project_path, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}" # print the description of the template
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            print('has input')
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
