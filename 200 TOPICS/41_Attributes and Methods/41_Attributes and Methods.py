# In this code:

# We define a class named Circle with a class attribute pi, which represents the value of π.
# We define a constructor method __init__ to initialize the radius attribute of the object.
# We define two methods within the Circle class: area() to calculate the area of the circle and circumference() to calculate its circumference.
# We create an object circle1 of the Circle class with a radius of 5.
# We access the attribute radius of circle1 using dot notation.
# We call the area() and circumference() methods of circle1 to calculate the area and circumference of the circle, respectively.
# Attributes and methods are fundamental concepts in object-oriented programming (OOP). They enable the definition of classes that 
# encapsulate data and behavior, providing a modular and organized approach to programming.

# Define a class named Circle
class Circle:
    # Class attribute
    pi = 3.14

    # Constructor method to initialize attributes
    def __init__(self, radius):
        self.radius = radius

    # Method to calculate the area of the circle
    def area(self):
        return self.pi * self.radius ** 2

    # Method to calculate the circumference of the circle
    def circumference(self):
        return 2 * self.pi * self.radius

# Create an object (instance) of the Circle class
circle1 = Circle(5)

# Accessing attributes of the object
print("Radius of circle1:", circle1.radius)

# Calling methods of the object
print("Area of circle1:", circle1.area())
print("Circumference of circle1:", circle1.circumference())
