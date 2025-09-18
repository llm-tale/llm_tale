def str_to_dict(string):
    paragraphs = string.split("\n\n")
    paragraph_dict = {index: paragraph.strip() for index, paragraph in enumerate(paragraphs)}
    return paragraph_dict


def dict_to_str(dictionary):
    ret_str = ""
    for k, v in dictionary.items():
        if k != len(dictionary) - 1:
            ret_str = ret_str + v + "\n\n"
        else:
            ret_str = ret_str + v
    return ret_str
