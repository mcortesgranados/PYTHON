# Python Linked List Example

class Node:
    """A class representing a node in a linked list."""

    def __init__(self, data):
        """Initialize the node with data."""
        self.data = data
        self.next = None

class LinkedList:
    """A class representing a linked list."""

    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None

    def is_empty(self):
        """Check if the linked list is empty."""
        return self.head is None

    def append(self, data):
        """Append a new node with data to the end of the linked list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def prepend(self, data):
        """Insert a new node with data at the beginning of the linked list."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        """Delete the first occurrence of a node with the given data."""
        if self.head is None:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current_node = self.head
        while current_node.next:
            if current_node.next.data == data:
                current_node.next = current_node.next.next
                return
            current_node = current_node.next

    def display(self):
        """Display the elements of the linked list."""
        current_node = self.head
        while current_node:
            print(current_node.data, end=" ")
            current_node = current_node.next
        print()

# Example: Using the LinkedList Class
# Let's demonstrate the usage of the LinkedList class with various operations.

# Create an empty linked list
my_linked_list = LinkedList()

# Append elements to the linked list
my_linked_list.append(1)
my_linked_list.append(2)
my_linked_list.append(3)

# Prepend an element to the linked list
my_linked_list.prepend(0)

# Display the linked list
print("Linked List:")
my_linked_list.display()  # Output: 0 1 2 3

# Delete an element from the linked list
my_linked_list.delete(2)

# Display the updated linked list
print("Updated Linked List:")
my_linked_list.display()  # Output: 0 1 3

# Documenting the LinkedList Class:
def linked_list_documentation():
    """
    This function demonstrates the LinkedList class in Python.

    LinkedList Class:
    - is_empty(): Check if the linked list is empty.
    - append(data): Append a new node with data to the end of the linked list.
    - prepend(data): Insert a new node with data at the beginning of the linked list.
    - delete(data): Delete the first occurrence of a node with the given data.
    - display(): Display the elements of the linked list.
    """
    pass

# End of example
