import heapq

# Python Heaps Example

# Example 1: Creating a Min-Heap
# A min-heap is a binary tree where the parent node is smaller than its children.
# In Python, min-heaps are commonly implemented using the heapq module.
# Here's an example of creating a min-heap:

heap = [4, 1, 7, 3, 8, 5]
heapq.heapify(heap)
# Explanation: The heapq.heapify() function converts the list 'heap' into a min-heap in-place.

print("Min-Heap:", heap)

# Example 2: Adding Elements to a Heap
# You can add elements to a heap using the heapq.heappush() function.
# Here's an example of adding elements to the min-heap:

heapq.heappush(heap, 2)
# Explanation: This adds the element 2 to the min-heap while maintaining the heap property.

print("Min-Heap after adding 2:", heap)

# Example 3: Removing Elements from a Heap
# You can remove the smallest element from a min-heap using the heapq.heappop() function.
# Here's an example of removing the smallest element from the min-heap:

smallest = heapq.heappop(heap)
# Explanation: This removes the smallest element from the min-heap and returns it.

print("Smallest Element Removed:", smallest)
print("Min-Heap after removing smallest element:", heap)

# Example 4: Retrieving the Smallest Element from a Heap
# You can retrieve the smallest element from a min-heap without removing it using the heapq.heappop() function.
# Here's an example of retrieving the smallest element from the min-heap:

smallest = heap[0]
# Explanation: The smallest element in a min-heap is always at index 0.

print("Smallest Element (without removal):", smallest)

# Example 5: Creating a Max-Heap
# A max-heap is a binary tree where the parent node is larger than its children.
# In Python, max-heaps can be simulated by using the negative of values.
# Here's an example of creating a max-heap:

max_heap = [4, 1, 7, 3, 8, 5]
max_heap = [-x for x in max_heap]
heapq.heapify(max_heap)
max_heap = [-x for x in max_heap]
# Explanation: To create a max-heap, negate each element of the list, apply heapq.heapify(),
# and then negate the elements back.

print("Max-Heap:", max_heap)

# Documenting the Heaps:
def heaps_documentation():
    """
    This function demonstrates various aspects of heaps in Python using the heapq module.

    Example 1:
    - Creating a Min-Heap: How to create a min-heap using the heapq.heapify() function.

    Example 2:
    - Adding Elements to a Heap: How to add elements to a heap using the heapq.heappush() function.

    Example 3:
    - Removing Elements from a Heap: How to remove the smallest element from a heap using the heapq.heappop() function.

    Example 4:
    - Retrieving the Smallest Element from a Heap: How to retrieve the smallest element from a heap without removing it.

    Example 5:
    - Creating a Max-Heap: How to simulate a max-heap using negative values and the heapq module.
    """
    pass

# End of examples
