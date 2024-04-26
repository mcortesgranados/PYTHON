# Python Conditionals Example

# Example 1: if Statement
# The if statement is used to execute a block of code only if the specified condition is true.
x = 10
if x > 5:
    print("x is greater than 5")

# Example 2: if-else Statement
# The if-else statement is used to execute one block of code if the condition is true, and another block if it's false.
y = 3
if y % 2 == 0:
    print("y is even")
else:
    print("y is odd")

# Example 3: Nested if-else Statements
# Nested if-else statements allow for multiple conditions to be checked sequentially.
grade = 75
if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
elif grade >= 70:
    print("C")
else:
    print("D")

# Example 4: Ternary Conditional Operator (Conditional Expression)
# Python supports a shorthand syntax for conditional expressions using the ternary operator.
a = 10
b = 20
max_value = a if a > b else b
print("Max value:", max_value)

# Example 5: Short-Circuit Evaluation
# Python uses short-circuit evaluation for logical operators (and, or).
# The second operand is evaluated only if the first operand doesn't determine the result.
x = 5
y = 0
if x > 0 and y != 0:
    z = x / y  # Division by zero error will not occur because the second condition is not evaluated.
else:
    z = float('nan')
print("Result:", z)

# Documenting the Conditionals:
def conditionals_documentation():
    """
    This function demonstrates various aspects of conditionals in Python.

    Example 1:
    - if Statement: Executes a block of code if a specified condition is true.

    Example 2:
    - if-else Statement: Executes one block of code if a condition is true, and another block if it's false.

    Example 3:
    - Nested if-else Statements: Allows for checking multiple conditions sequentially.

    Example 4:
    - Ternary Conditional Operator: Provides a shorthand syntax for conditional expressions.

    Example 5:
    - Short-Circuit Evaluation: Demonstrates how Python performs short-circuit evaluation for logical operators.
    """
    pass

# End of examples
