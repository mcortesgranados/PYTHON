
# itertools is a powerful module in Python's standard library that provides various functions for working with iterators. It offers tools for creating and combining iterators to solve common iteration-related tasks efficiently. Let's explore some of the functions provided by itertools:

# Infinite Iterators:

# itertools.count(start=0, step=1): Returns an iterator that generates numbers starting from start with a step size of step.
# itertools.cycle(iterable): Returns an iterator that cycles indefinitely over the elements of iterable.
# itertools.repeat(elem, times=None): Returns an iterator that repeatedly yields elem times number of times, or indefinitely if times is None.

# Finite Iterators:

# itertools.chain(*iterables): Chains together multiple iterables, yielding elements from each one consecutively.
# itertools.zip_longest(*iterables, fillvalue=None): Zips together multiple iterables, filling missing values with fillvalue if provided.
# itertools.product(*iterables, repeat=1): Computes the Cartesian product of the provided iterables.
# itertools.permutations(iterable, r=None): Generates all possible permutations of length r from the elements of iterable.
# itertools.combinations(iterable, r): Generates all possible combinations of length r from the elements of iterable.
# itertools.combinations_with_replacement(iterable, r): Generates all possible combinations with replacement of length r from the elements of iterable.

# Iterators Terminating on the Shortest Input Sequence:

# itertools.islice(iterable, start, stop, step=1): Returns an iterator that yields selected elements from iterable as specified by start, stop, and step.
# itertools.takewhile(predicate, iterable): Returns an iterator that yields elements from iterable as long as the predicate function evaluates to True.
# itertools.dropwhile(predicate, iterable): Returns an iterator that skips elements from iterable as long as the predicate function evaluates to True, then yields the remaining elements.
# itertools.filterfalse(predicate, iterable): Returns an iterator that yields elements from iterable for which the predicate function evaluates to False.

# Combining Multiple Iterators:

# itertools.chain.from_iterable(iterables): Chains together multiple iterables, similar to chain().
# itertools.product() supports repeat keyword which produces the cartesian product of iterable with itself r times.
# itertools.starmap(func, iterable): Applies func to each element of iterable, unpacking the arguments from each element when calling the function.
# itertools.zip_longest() supports arbitrary numbers of input iterators.
# These are just a few examples of the functions provided by the itertools module. By leveraging these functions, you can write more concise and efficient code for various iteration tasks in Python.

import itertools

# Infinite Iterators
# 1. itertools.count(start=0, step=1)
for i in itertools.count(start=1, step=2):
    print(i, end=' ')
    if i > 10:
        break
# Output: 1 3 5 7 9 11

# 2. itertools.cycle(iterable)
cycle_iter = itertools.cycle('ABC')
for i in range(6):
    print(next(cycle_iter), end=' ')
# Output: A B C A B C

# 3. itertools.repeat(elem, times=None)
repeat_iter = itertools.repeat('Hello', times=3)
for i in repeat_iter:
    print(i, end=' ')
# Output: Hello Hello Hello

# Finite Iterators
# 1. itertools.chain(*iterables)
chain_iter = itertools.chain([1, 2, 3], ['a', 'b', 'c'])
print(list(chain_iter))  # Output: [1, 2, 3, 'a', 'b', 'c']

# 2. itertools.zip_longest(*iterables, fillvalue=None)
zip_iter = itertools.zip_longest([1, 2, 3], ['a', 'b'], fillvalue='x')
print(list(zip_iter))  # Output: [(1, 'a'), (2, 'b'), (3, 'x')]

# 3. itertools.product(*iterables, repeat=1)
product_iter = itertools.product([1, 2], ['a', 'b'], repeat=2)
print(list(product_iter))  # Output: [(1, 'a', 1, 'a'), (1, 'a', 1, 'b'), (1, 'a', 2, 'a'), ...]

# 4. itertools.permutations(iterable, r=None)
perm_iter = itertools.permutations('ABCD', 2)
print(list(perm_iter))  # Output: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ...]

# 5. itertools.combinations(iterable, r)
comb_iter = itertools.combinations('ABCD', 2)
print(list(comb_iter))  # Output: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

# 6. itertools.combinations_with_replacement(iterable, r)
comb_rep_iter = itertools.combinations_with_replacement('ABCD', 2)
print(list(comb_rep_iter))  # Output: [('A', 'A'), ('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'B'), ('B', 'C'), ...]

# Iterators Terminating on the Shortest Input Sequence
# 1. itertools.islice(iterable, start, stop, step=1)
islice_iter = itertools.islice('ABCDEFG', 2, None, 2)
print(list(islice_iter))  # Output: ['C', 'E', 'G']

# 2. itertools.takewhile(predicate, iterable)
takewhile_iter = itertools.takewhile(lambda x: x < 5, [1, 3, 5, 7, 2])
print(list(takewhile_iter))  # Output: [1, 3]

# 3. itertools.dropwhile(predicate, iterable)
dropwhile_iter = itertools.dropwhile(lambda x: x < 5, [1, 3, 5, 7, 2])
print(list(dropwhile_iter))  # Output: [5, 7, 2]

# 4. itertools.filterfalse(predicate, iterable)
filterfalse_iter = itertools.filterfalse(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])
print(list(filterfalse_iter))  # Output: [1, 3, 5]
