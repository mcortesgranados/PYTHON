
# Recursion is a programming technique where a function calls itself in order to solve smaller instances of the same problem. It is particularly useful for solving problems that can be broken down into smaller, similar subproblems. Recursion involves two main components: a base case and a recursive case.

# Here are the key concepts of recursion:

# Base Case:

# The base case is a condition that determines when the recursion should stop.
#It provides the terminating condition for the recursive calls and prevents infinite recursion.
# Without a base case, the recursion would continue indefinitely, leading to a stack overflow.
# Recursive Case:

# The recursive case defines how the problem is broken down into smaller subproblems.
# It involves calling the function recursively with modified input parameters, moving towards the base case.
# Each recursive call should make progress towards reaching the base case.
# Recursion is commonly used to solve problems in which the solution depends on solutions to smaller instances of the same problem. Some classic examples of problems that are often solved using recursion include:

# Computing factorial of a number.
# Calculating Fibonacci numbers.
# Traversing tree-like data structures such as binary trees.
# Searching and sorting algorithms like binary search and quicksort.
# Here's a simple example of a recursive function to calculate the factorial of a number:

def factorial(n):
    # Base case: factorial of 0 or 1 is 1
    if n == 0 or n == 1:
        return 1
    # Recursive case: n! = n * (n-1)!
    else:
        return n * factorial(n - 1)

# Example usage
print(factorial(5))  # Output: 120 (5! = 5*4*3*2*1)


# Recursion provides an elegant and concise solution to certain types of problems but can be less efficient than iterative approaches due to 
# the overhead of function calls and the risk of stack overflow for deep recursion. It's important to understand when to use 
# recursion and when to use iterative solutions based on the specific problem and its constraints.

# Computing factorial of a number:

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# Example usage
print(factorial(5))  # Output: 120 (5! = 5*4*3*2*1)


# Calculating Fibonacci numbers:

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
for i in range(10):
    print(fibonacci(i), end=" ")  # Output: 0 1 1 2 3 5 8 13 21 34


# Traversing binary trees:

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.value, end=" ")
        inorder_traversal(root.right)

# Example usage
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print("Inorder traversal:", end=" ")
inorder_traversal(root)  # Output: 4 2 5 1 3


# Binary search algorithm:


def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target = 5
print("Index of", target, ":", binary_search(arr, target))  # Output: 4


# Quicksort algorithm:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Example usage
arr = [3, 6, 8, 10, 1, 2, 1]
print("Sorted array:", quicksort(arr))  # Output: [1, 1, 2, 3, 6, 8, 10]


