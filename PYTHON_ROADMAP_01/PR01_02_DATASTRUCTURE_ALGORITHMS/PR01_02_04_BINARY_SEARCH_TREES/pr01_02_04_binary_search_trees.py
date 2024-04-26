# Python Binary Search Trees Example

class TreeNode:
    """A class representing a node in a binary search tree."""

    def __init__(self, key):
        """Initialize a node with a key."""
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    """A class representing a binary search tree."""

    def __init__(self):
        """Initialize an empty binary search tree."""
        self.root = None

    def insert(self, key):
        """Insert a key into the binary search tree."""
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, root, key):
        """Recursively insert a key into the binary search tree."""
        if root is None:
            return TreeNode(key)
        if key < root.key:
            root.left = self._insert_recursive(root.left, key)
        elif key > root.key:
            root.right = self._insert_recursive(root.right, key)
        return root

    def search(self, key):
        """Search for a key in the binary search tree."""
        return self._search_recursive(self.root, key)

    def _search_recursive(self, root, key):
        """Recursively search for a key in the binary search tree."""
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self._search_recursive(root.left, key)
        return self._search_recursive(root.right, key)

    def delete(self, key):
        """Delete a key from the binary search tree."""
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, root, key):
        """Recursively delete a key from the binary search tree."""
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete_recursive(root.left, key)
        elif key > root.key:
            root.right = self._delete_recursive(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            min_node = self._find_min(root.right)
            root.key = min_node.key
            root.right = self._delete_recursive(root.right, min_node.key)
        return root

    def _find_min(self, node):
        """Find the node with the minimum key in a subtree."""
        while node.left is not None:
            node = node.left
        return node

    def inorder_traversal(self):
        """Perform an inorder traversal of the binary search tree."""
        result = []
        self._inorder_traversal_recursive(self.root, result)
        return result

    def _inorder_traversal_recursive(self, root, result):
        """Recursively perform an inorder traversal of the binary search tree."""
        if root is not None:
            self._inorder_traversal_recursive(root.left, result)
            result.append(root.key)
            self._inorder_traversal_recursive(root.right, result)

# Example: Using the BinarySearchTree Class
# Let's demonstrate the usage of the BinarySearchTree class with various operations.

# Create a binary search tree
bst = BinarySearchTree()

# Insert keys into the binary search tree
bst.insert(10)
bst.insert(5)
bst.insert(15)
bst.insert(3)
bst.insert(7)
bst.insert(12)
bst.insert(20)

# Search for keys in the binary search tree
print("Search Results:")
print("Key 5:", bst.search(5) is not None)  # Output: True
print("Key 8:", bst.search(8) is not None)  # Output: False

# Perform an inorder traversal of the binary search tree
print("Inorder Traversal:", bst.inorder_traversal())  # Output: [3, 5, 7, 10, 12, 15, 20]

# Delete a key from the binary search tree
bst.delete(10)

# Perform an inorder traversal after deletion
print("Inorder Traversal after Deletion:", bst.inorder_traversal())  # Output: [3, 5, 7, 12, 15, 20]

# Documenting the BinarySearchTree Class:
def binary_search_tree_documentation():
    """
    This function demonstrates the BinarySearchTree class in Python.

    BinarySearchTree Class:
    - __init__(): Initialize an empty binary search tree.
    - insert(key): Insert a key into the binary search tree.
    - search(key): Search for a key in the binary search tree.
    - delete(key): Delete a key from the binary search tree.
    - inorder_traversal(): Perform an inorder traversal of the binary search tree.
    """
    pass

# End of example
