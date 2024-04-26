# Python Stacks Example

class Stack:
    """A class representing a stack."""

    def __init__(self):
        """Initialize an empty stack."""
        self.items = []

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0

    def push(self, item):
        """Push an item onto the stack."""
        self.items.append(item)

    def pop(self):
        """Remove and return the top item from the stack."""
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("pop from an empty stack")

    def peek(self):
        """Return the top item from the stack without removing it."""
        if not self.is_empty():
            return self.items[-1]
        else:
            return None

    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)

# Example: Using the Stack Class
# Let's demonstrate the usage of the Stack class with various operations.

# Create an empty stack
my_stack = Stack()

# Push elements onto the stack
my_stack.push(1)
my_stack.push(2)
my_stack.push(3)

# Display the stack
print("Stack:", my_stack.items)  # Output: [1, 2, 3]

# Peek at the top element of the stack
print("Top Element (Peek):", my_stack.peek())  # Output: 3

# Pop elements from the stack
popped_element = my_stack.pop()
print("Popped Element:", popped_element)  # Output: 3

# Display the updated stack
print("Updated Stack:", my_stack.items)  # Output: [1, 2]

# Documenting the Stack Class:
def stack_documentation():
    """
    This function demonstrates the Stack class in Python.

    Stack Class:
    - is_empty(): Check if the stack is empty.
    - push(item): Push an item onto the stack.
    - pop(): Remove and return the top item from the stack.
    - peek(): Return the top item from the stack without removing it.
    - size(): Return the number of items in the stack.
    """
    pass

# End of example
