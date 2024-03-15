# Question 11/18-Python3  00:01/02:30

# Instructions

# Implement the function solution so that the execution of the following line:

# print(solution(´Hello you !´))

# Gives the following output (one word per line with \n:

# Hello
# you
# !

# Input is always a string.


# Answer
# Python code below
# Use print("messages...") to debug your solution.

def solution(string):
    words = string.split()  # Split the string into words based on whitespace characters
    for word in words:
        print(word)  # Print each word on a separate line

# Test the function
solution('Hello you !')


# © Testcode

# print(Answer.solution(´Hello you !´))