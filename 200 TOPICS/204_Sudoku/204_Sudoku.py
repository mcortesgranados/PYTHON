# Question 9/18-Python3 00:01

# Instructions

# You need to check whether a sudoku grid is correct and tell where the first error is
# found.

# A sudoku consists of a 2-dimensional board, 9 cells by 9 cells. Each cell contains a
# digit from 1 to 9.

# All the sudoku cells you get in input are already filled.

# You must check all the following conditions in the order shown, and exit at the first
# error found.

# « Each line must contain every digit from 1 to 9 only once. Check them from top
# to bottom. If a line is not correct, return LINE <y> INVALID, where y is the index
# of the line (counted from 0).

# « Each column must contain every digit from 1 to 9 only once. Check them from
# left to right. If a column is not correct, return COLUMN <x> INVALID, where x is
# the index of the column (counted from 0).

# « A sudoku is also divided into 9 regions, made up of squares of 3 cells by 3
# cells. Each region must contain every digit from 1 to 9 only once. Check them
# from top to bottom, then from left to right. If a region is not correct, retum
# REGION <r> INVALID, where r is the index of the region (counted from 0).

# If all the conditions are fulfilled, return the text VALID.

# ° Implementation

# F  Function
# Implement the function check_sudoku.

# > Parameters
# sudoku (List[List[int]]): A 2-dimensional array containing integers from 1 to 9.

# < Return value
# check_result (str): A string: "LINE <y> INVALID", "COLUMN <x> INVALID", "REGION <r> 
# INVALID" or "VALID".

# ! Contrains 
# Available RAM: 512MB

# Example
# Parameters 									    Return value
# (9) [[. . .], [. . .], [. . .], [. ...]         LINE 6 INVALID


# Answer
# Tests

# The lines of indices 6 and 7 are not correct

# Parameters 									                              Return value
# (9) [[. . .], [. . .], [. . .], [. . .], [. . .], [. . .], [... ]         LINE 6 INVALID

def check_sudoku(sudoku):
    # Check rows
    for i, row in enumerate(sudoku):
        if len(set(row)) != len(row) or any(cell not in range(1, 10) for cell in row):
            return f"LINE {i} INVALID"
    
    # Check columns
    for j in range(9):
        column = [sudoku[i][j] for i in range(9)]
        if len(set(column)) != len(column):
            return f"COLUMN {j} INVALID"
    
    # Check regions
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            region = [sudoku[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            if len(set(region)) != len(region):
                return f"REGION {i//3*3 + j//3} INVALID"
    
    return "VALID"

# Example usage:
sudoku1 = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
]

print(check_sudoku(sudoku1))  # Output: VALID




