# Python Binary Recursion Example

def binary_search(arr, target):
    """
    Perform binary search on a sorted array to find the target element.

    Args:
    - arr (list): The sorted array to search.
    - target (int): The target element to find.

    Returns:
    - int: The index of the target element if found, else -1.
    """
    return _binary_search_recursive(arr, target, 0, len(arr) - 1)

def _binary_search_recursive(arr, target, low, high):
    """
    Recursively perform binary search on a sorted array.

    Args:
    - arr (list): The sorted array to search.
    - target (int): The target element to find.
    - low (int): The lowest index of the current subarray.
    - high (int): The highest index of the current subarray.

    Returns:
    - int: The index of the target element if found, else -1.
    """
    if low > high:
        return -1

    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return _binary_search_recursive(arr, target, mid + 1, high)
    else:
        return _binary_search_recursive(arr, target, low, mid - 1)

# Example: Using Binary Recursion for Binary Search
# Let's demonstrate the usage of binary recursion for performing binary search.

# Sorted array for binary search
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Target elements for searching
targets = [5, 8, 12]

# Perform binary search for each target
for target in targets:
    index = binary_search(arr, target)
    if index != -1:
        print(f"Target {target} found at index {index}")
    else:
        print(f"Target {target} not found in the array")

# Documenting the Binary Recursion Function:
def binary_recursion_documentation():
    """
    This function demonstrates binary recursion in Python for binary search.

    Binary Recursion Function:
    - binary_search(arr, target): Perform binary search on a sorted array to find the target element.
    - _binary_search_recursive(arr, target, low, high): Recursively perform binary search on a sorted array.
    """
    pass

# End of example
