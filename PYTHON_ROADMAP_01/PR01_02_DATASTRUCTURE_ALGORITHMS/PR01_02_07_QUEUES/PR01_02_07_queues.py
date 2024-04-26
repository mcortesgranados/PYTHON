# Python Queues Example

class Queue:
    """
    A class representing a queue.

    Attributes:
    - items (list): The list to store elements of the queue.
    """

    def __init__(self):
        """Initialize an empty queue."""
        self.items = []

    def is_empty(self):
        """
        Check if the queue is empty.

        Returns:
        - bool: True if the queue is empty, False otherwise.
        """
        return len(self.items) == 0

    def enqueue(self, item):
        """
        Add an item to the rear of the queue.

        Args:
        - item: The item to be added to the queue.
        """
        self.items.append(item)

    def dequeue(self):
        """
        Remove and return the item at the front of the queue.

        Returns:
        - item: The item removed from the front of the queue.
        """
        if not self.is_empty():
            return self.items.pop(0)
        else:
            raise IndexError("dequeue from an empty queue")

    def peek(self):
        """
        Return the item at the front of the queue without removing it.

        Returns:
        - item: The item at the front of the queue.
        """
        if not self.is_empty():
            return self.items[0]
        else:
            return None

    def size(self):
        """
        Return the number of items in the queue.

        Returns:
        - int: The number of items in the queue.
        """
        return len(self.items)

# Example: Using the Queue Class
# Let's demonstrate the usage of the Queue class with various operations.

# Create a queue
my_queue = Queue()

# Add elements to the queue
my_queue.enqueue(1)
my_queue.enqueue(2)
my_queue.enqueue(3)

# Display the queue
print("Queue:", my_queue.items)  # Output: [1, 2, 3]

# Remove an element from the queue
removed_item = my_queue.dequeue()
print("Removed Item:", removed_item)  # Output: 1

# Display the updated queue
print("Updated Queue:", my_queue.items)  # Output: [2, 3]

# Documenting the Queue Class:
def queue_documentation():
    """
    This function demonstrates the Queue class in Python.

    Queue Class:
    - __init__(): Initialize an empty queue.
    - is_empty(): Check if the queue is empty.
    - enqueue(item): Add an item to the rear of the queue.
    - dequeue(): Remove and return the item at the front of the queue.
    - peek(): Return the item at the front of the queue without removing it.
    - size(): Return the number of items in the queue.
    """
    pass

# End of example
