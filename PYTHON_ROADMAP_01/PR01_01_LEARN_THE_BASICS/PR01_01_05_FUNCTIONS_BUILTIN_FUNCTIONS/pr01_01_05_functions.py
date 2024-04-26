# This program demonstrates functions and built-in functions in Python

# Define a function to calculate the area of a circle
def calculate_circle_area(radius):
  """
  This function calculates the area of a circle given its radius.

  Args:
      radius (float): The radius of the circle.

  Returns:
      float: The calculated area of the circle.
  """
  pi = 3.14159  # Pre-defined value for pi (can be replaced with built-in math.pi)
  area = pi * radius * radius
  return area

# Calculate the area of a circle with radius 5 using the function
circle_radius = 5
circle_area = calculate_circle_area(circle_radius)
print("Area of the circle:", circle_area)

# Demonstrate built-in functions: absolute value, power
number = -10
absolute_value = abs(number)  # abs() function for absolute value
squared_value = number**2  # Power operator (**)

print("Absolute value of", number, ":", absolute_value)
print(number, "squared:", squared_value)
