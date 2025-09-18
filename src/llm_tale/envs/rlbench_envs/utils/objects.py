class Object:
    def __init__(self, object_handler):
        self.object_handler = object_handler

    def get_name(self):
        return self.object_handler.get_name()

    def get_position(self):
        return self.object_handler.get_position()

    def get_pose(self):
        return self.object_handler.get_pose()

    def parser_attribute(self, attributes):
        att = attributes.split(".")
        if att[0] not in self.attrib.keys():
            raise ValueError("Attribute not found")
        return self.attrib[att[0]].parser_attribute(att[1])
