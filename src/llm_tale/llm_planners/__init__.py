"""For the prompt structure, we took inspiration from https://github.com/Stanford-ILIAD/droc"""

import os

proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_folder = os.path.join(proj_folder, "llm_planners")
aff_folder = os.path.join(proj_folder, "llm_planners/mid_level/")
