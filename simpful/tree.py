# thisdict = {
#   "root": "child"
#   }

#   x = thisdict["root"]


class Node:
    """This is a dumb implementation"""
    def __init__(self, left, right, operator=None, value=None):
        self.left = left
        self.right = right
        self.operator = operator

        if operator is None:
            self.value = value
        else:
            self.value = self.eval()

    def eval(self):
        if self.operator == "sum":
            result = sum([self.left.value, self.right.value])
        return result

leaf1 = Node(None, None, value=5)
leaf2 = Node(None, None, value=3)
root = Node(leaf1, leaf2, operator="sum")




class ShorterNode:
    """Somewhat better approach"""
    def __init__(self, value):
        self.value = value

leaf1 = ShorterNode(5)
leaf2 = ShorterNode(3)
root = ShorterNode(sum(leaf1.value, leaf2.value))

string = "leaf1 SUM leaf2"



# class2 Node:
#     def __init__(self, **args):
#         for argument in **args:
#             self.argument = argument

# # node = {"left": , "right": }
