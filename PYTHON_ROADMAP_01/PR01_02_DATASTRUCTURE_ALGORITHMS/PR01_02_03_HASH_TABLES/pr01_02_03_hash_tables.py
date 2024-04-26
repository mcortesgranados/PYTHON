# Python Hash Tables Example

class HashTable:
    """A class representing a hash table."""

    def __init__(self, size=10):
        """Initialize a hash table with a given size."""
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        """Generate a hash value for a given key."""
        return hash(key) % self.size

    def put(self, key, value):
        """Insert a key-value pair into the hash table."""
        index = self._hash(key)
        for pair in self.table[index]:
            if pair[0] == key:
                pair[1] = value  # Update value if key exists
                return
        self.table[index].append([key, value])

    def get(self, key):
        """Retrieve the value associated with a given key."""
        index = self._hash(key)
        for pair in self.table[index]:
            if pair[0] == key:
                return pair[1]  # Return value if key exists
        raise KeyError(f"Key '{key}' not found")

    def remove(self, key):
        """Remove a key-value pair from the hash table."""
        index = self._hash(key)
        for i, pair in enumerate(self.table[index]):
            if pair[0] == key:
                del self.table[index][i]  # Remove pair if key exists
                return
        raise KeyError(f"Key '{key}' not found")

# Example: Using the HashTable Class
# Let's demonstrate the usage of the HashTable class with various operations.

# Create a hash table
my_hash_table = HashTable()

# Insert key-value pairs into the hash table
my_hash_table.put("apple", 5)
my_hash_table.put("banana", 10)
my_hash_table.put("orange", 7)

# Retrieve values from the hash table
print("Value for 'apple':", my_hash_table.get("apple"))  # Output: 5

# Remove a key-value pair from the hash table
my_hash_table.remove("banana")

# Retrieve values after removal
try:
    print("Value for 'banana':", my_hash_table.get("banana"))
except KeyError as e:
    print(e)  # Output: Key 'banana' not found

# Documenting the HashTable Class:
def hash_table_documentation():
    """
    This function demonstrates the HashTable class in Python.

    HashTable Class:
    - __init__(size): Initialize a hash table with a given size.
    - put(key, value): Insert a key-value pair into the hash table.
    - get(key): Retrieve the value associated with a given key.
    - remove(key): Remove a key-value pair from the hash table.
    """
    pass

# End of example
