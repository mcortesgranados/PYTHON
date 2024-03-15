# Question 12/18-Python3 00:07 /1500

# Instructions

# © You can change the programming language at the top right of the screen.

# """
# Given a graph, write a program that search and print the path from
# the root node to the specified node using a BFS (Breadth First Search)

# A graph can be represented as a dictionary, so the following graph:

# A
# |\
# B C
# |\ \
# D E F

# is represented like this:

# graph = {
#  "A": ["B", "C"],
#  "B": ["D", "E"],
#  "C": ["F"],
#  "D": [],
#  "E": [],
#  "F": []  
#}  
  
# where root_node is "A"

# This is a list of examples for the previous graph:
# - wanted_node="F", result=["A", "B", "C", "D", "E", "F"]
# - wanted_node="D", result=["A", "B", "C", "D"]
# - wanted_node="C", result=["A", "B", "C"]
# """

from collections import deque

def bfs(graph, root_node, wanted_node):
    visited = set()
    queue = deque([(root_node, [])])

    while queue:
        node, path = queue.popleft()
        if node == wanted_node:
            return path + [node]
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                queue.append((neighbor, path + [node]))

# Example graph
graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F"],
    "D": [],
    "E": [],
    "F": []  
}

# Example usage
print(bfs(graph, "A", "F"))  # Output: ['A', 'C', 'F']
print(bfs(graph, "A", "D"))  # Output: ['A', 'B', 'D']
print(bfs(graph, "A", "C"))  # Output: ['A', 'C']
