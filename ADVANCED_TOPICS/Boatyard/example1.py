def add(a: int, b: int) -> int:
    return a + b

result = add(5, '10')  # Type error: 'str' is not compatible with 'int'
print(result)
