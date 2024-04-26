# Python Sorting Algorithms Example

def bubble_sort(arr):
    """
    Sort an array using the Bubble Sort algorithm.

    Args:
    - arr (list): The unsorted array.

    Returns:
    - list: The sorted array.
    """
    n = len(arr)
    for i in range(n):
        # Flag to optimize the sorting process
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                # Swap elements if they are in the wrong order
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # If no two elements were swapped in the inner loop, array is sorted
        if not swapped:
            break
    return arr

def insertion_sort(arr):
    """
    Sort an array using the Insertion Sort algorithm.

    Args:
    - arr (list): The unsorted array.

    Returns:
    - list: The sorted array.
    """
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """
    Sort an array using the Merge Sort algorithm.

    Args:
    - arr (list): The unsorted array.

    Returns:
    - list: The sorted array.
    """
    if len(arr) <= 1:
        return arr
    # Divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    # Recursively sort each half
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left_half, right_half):
    """
    Merge two sorted arrays into a single sorted array.

    Args:
    - left_half (list): The sorted left half.
    - right_half (list): The sorted right half.

    Returns:
    - list: The merged and sorted array.
    """
    result = []
    left_index, right_index = 0, 0
    while left_index < len(left_half) and right_index < len(right_half):
        if left_half[left_index] < right_half[right_index]:
            result.append(left_half[left_index])
            left_index += 1
        else:
            result.append(right_half[right_index])
            right_index += 1
    result.extend(left_half[left_index:])
    result.extend(right_half[right_index:])
    return result

# Example: Using Sorting Algorithms
# Let's demonstrate the usage of each sorting algorithm with an example.

# Unsorted array
arr = [64, 34, 25, 12, 22, 11, 90]

# Sorting using Bubble Sort
sorted_arr_bubble = bubble_sort(arr.copy())
print("Bubble Sort:", sorted_arr_bubble)

# Sorting using Insertion Sort
sorted_arr_insertion = insertion_sort(arr.copy())
print("Insertion Sort:", sorted_arr_insertion)

# Sorting using Merge Sort
sorted_arr_merge = merge_sort(arr.copy())
print("Merge Sort:", sorted_arr_merge)

# Documenting the Sorting Algorithms:
def sorting_algorithms_documentation():
    """
    This function demonstrates three sorting algorithms in Python.

    Sorting Algorithms:
    - bubble_sort(arr): Sort an array using the Bubble Sort algorithm.
    - insertion_sort(arr): Sort an array using the Insertion Sort algorithm.
    - merge_sort(arr): Sort an array using the Merge Sort algorithm.
    """
    pass

# End of example
