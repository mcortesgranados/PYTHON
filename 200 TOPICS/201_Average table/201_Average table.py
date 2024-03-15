# Instructions

# Write the body of the average(table) function.

# The function should return the average of the values contained in table.table is
# always a defined array, objects in table are always numbers.

# average should return 0 if table is empty.


# Answer
# Python code below
# Use print("messages. . .") to debug your solution.

def average(table):
    if not table:  # Check if the table is empty
        return 0
    
    total = sum(table)  # Calculate the sum of all values in the table
    avg = total / len(table)  # Calculate the average
    return avg

# Test the function with various cases
values1 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
values2 = []  # Empty table
values3 = [5]  # Single value
values4 = [2, 2, 2, 2, 2]  # Identical values
values5 = [1, 2, 3, 4, 5]  # Consecutive values

# Testing
print("Average of values1:", average(values1))  # Output: 5.0
print("Average of values2:", average(values2))  # Output: 0
print("Average of values3:", average(values3))  # Output: 5.0
print("Average of values4:", average(values4))  # Output: 2.0
print("Average of values5:", average(values5))  # Output: 3.0

# © Test code

# values = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
# print (Answer.average(values)) #5.0