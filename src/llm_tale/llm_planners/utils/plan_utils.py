import os
import openai

from llm_tale.llm_planners.utils.modular_prompt import modular_prompt
from llm_tale.llm_planners.utils.io_utils import read_py


def prompt_task_planner(Instruction, object_state, folder_path, verbose=False, model="gpt-4o"):
    high_level_folder = folder_path + "/task_level/"
    prompt = modular_prompt(high_level_folder + "tl_backbone.txt", high_level_folder + "tl_content.txt")

    updated_prompt = prompt.get_prompt()
    updated_prompt = updated_prompt.replace("Instruction:#", "Instruction: " + Instruction)
    updated_prompt = updated_prompt.replace("Objects:#", f"{object_state}")
    if verbose:
        print(updated_prompt)

    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "system", "content": updated_prompt}], temperature=0, max_tokens=200
    )
    response.text = response.choices[0].message["content"]
    if verbose:
        print(response.text)
    return response


def prompt_code_translator(plans, object_state, folder_path, verbose=False, model="gpt-4o"):
    prompt_path = os.path.join(folder_path, "task_level/code_translator.txt")
    prompt = read_py(prompt_path)
    prompt = prompt.replace("Objects:#", f"{object_state}")
    prompt = prompt.replace("Plans:#", plans)
    if verbose:
        print(prompt)

    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "system", "content": prompt}], temperature=0, top_p=1, max_tokens=200
    )
    response.text = response.choices[0].message["content"]
    if verbose:
        print(response.text)
    return response


def prompt_affordance_identifier(step_instru, object_state, folder_path, verbose=False, model="gpt-4o"):
    as_prompt_path = os.path.join(folder_path, "affordance_level/affordance_identifier.txt")
    as_prompt = read_py(as_prompt_path)
    as_prompt = as_prompt.replace("Instruction:#", "Instruction: " + step_instru)
    as_prompt = as_prompt.replace("Objects:#", f"{object_state}")
    if verbose:
        print(as_prompt)

    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "system", "content": as_prompt}], temperature=0, top_p=1, max_tokens=200
    )
    response.text = response.choices[0].message["content"]
    if verbose:
        print(response.text)
    return response


def prompt_affordance_planner(step_instru, mode, object_state, folder_path, verbose=False, model="gpt-4o"):
    if mode == "pick":
        aff_prompt_path = os.path.join(folder_path, "affordance_level/pick_planner.txt")
    elif mode == "transport":
        aff_prompt_path = os.path.join(folder_path, "affordance_level/transport_planner.txt")
    as_prompt = read_py(aff_prompt_path)
    as_prompt = as_prompt.replace("Instruction:#", "Instruction: " + step_instru)
    as_prompt = as_prompt.replace("Objects:#", f"{object_state}")
    if verbose:
        print(as_prompt)

    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "system", "content": as_prompt}], temperature=0, top_p=1, max_tokens=200
    )
    response.text = response.choices[0].message["content"]
    if verbose:
        print(response.text)
    return response
