from llm_tale.llm_planners.utils.io_utils import read_py
from llm_tale.llm_planners.utils.string_utils import str_to_dict, dict_to_str


class modular_prompt:
    def __init__(self, backbone_path, content_path) -> None:
        self.backbone = read_py(backbone_path)
        self.content_str = read_py(content_path)
        self.content_dict = str_to_dict(self.content_str)
        self.prompt = None
        self.form_prompt()

    def form_prompt(self):
        self.prompt = self.backbone.replace("{}", self.content_str)

    def update_content(self, new_content):
        ind = list(self.content_dict.keys())[-1] + 1
        self.content_dict.update({ind: new_content})
        self.content_str = dict_to_str(self.content_dict)
        self.form_prompt()

    def get_prompt(self):
        return self.prompt
