class NumberIterator:
    """
    An iterator class that generates a sequence of numbers.

    This class implements the iterator protocol by defining the `__iter__` 
    and `__next__` methods.
    """

    def __init__(self, start, stop):
        """
        Initializes the iterator with a starting and stopping value.

        Args:
            start (int): The starting number of the sequence.
            stop (int): The ending number (not inclusive) of the sequence.
        """
        self.current = start - 1  # Initialize before start for first __next__ call
        self.stop = stop

    def __iter__(self):
        """
        Returns the iterator object itself.

        This method is required by the iterator protocol to allow usage in for loops.
        """
        return self

    def __next__(self):
        """
        Returns the next element in the sequence.

        This method is called repeatedly in a for loop until it raises a 
        StopIteration exception, signaling the end of the sequence.
        """
        self.current += 1
        if self.current < self.stop:
            return self.current
        else:
            raise StopIteration

# Create an iterator object
number_iterator = NumberIterator(1, 6)

# Use the iterator in a for loop
for num in number_iterator:
    print(num)
