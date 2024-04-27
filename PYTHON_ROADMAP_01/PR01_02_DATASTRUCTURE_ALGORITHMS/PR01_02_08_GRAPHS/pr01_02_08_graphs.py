# Python Graphs Example

class Graph:
    """
    A class representing an undirected graph.

    Attributes:
    - vertices (dict): A dictionary to store vertices and their adjacent vertices.
    """

    def __init__(self):
        """Initialize an empty graph."""
        self.vertices = {}

    def add_vertex(self, vertex):
        """
        Add a vertex to the graph.

        Args:
        - vertex: The vertex to be added to the graph.
        """
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2):
        """
        Add an undirected edge between two vertices.

        Args:
        - vertex1: The first vertex.
        - vertex2: The second vertex.
        """
        if vertex1 in self.vertices and vertex2 in self.vertices:
            self.vertices[vertex1].append(vertex2)
            self.vertices[vertex2].append(vertex1)

    def get_adjacent_vertices(self, vertex):
        """
        Get the adjacent vertices of a given vertex.

        Args:
        - vertex: The vertex whose adjacent vertices are to be retrieved.

        Returns:
        - list: The list of adjacent vertices.
        """
        if vertex in self.vertices:
            return self.vertices[vertex]
        else:
            return []

    def __str__(self):
        """Return a string representation of the graph."""
        result = ""
        for vertex, adjacent_vertices in self.vertices.items():
            result += f"{vertex}: {', '.join(map(str, adjacent_vertices))}\n"
        return result

# Example: Using the Graph Class
# Let's demonstrate the usage of the Graph class with various operations.

# Create a graph
my_graph = Graph()

# Add vertices to the graph
my_graph.add_vertex(1)
my_graph.add_vertex(2)
my_graph.add_vertex(3)
my_graph.add_vertex(4)

# Add edges to the graph
my_graph.add_edge(1, 2)
my_graph.add_edge(1, 3)
my_graph.add_edge(2, 3)
my_graph.add_edge(3, 4)

# Display the graph
print("Graph:")
print(my_graph)

# Get adjacent vertices of a vertex
adjacent_vertices = my_graph.get_adjacent_vertices(3)
print("Adjacent Vertices of Vertex 3:", adjacent_vertices)  # Output: [1, 2, 4]

# Documenting the Graph Class:
def graph_documentation():
    """
    This function demonstrates the Graph class in Python.

    Graph Class:
    - __init__(): Initialize an empty graph.
    - add_vertex(vertex): Add a vertex to the graph.
    - add_edge(vertex1, vertex2): Add an undirected edge between two vertices.
    - get_adjacent_vertices(vertex): Get the adjacent vertices of a given vertex.
    - __str__(): Return a string representation of the graph.
    """
    pass

# End of example
