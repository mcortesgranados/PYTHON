# Inheritance is a fundamental concept in object-oriented programming (OOP) that allows a new class (called a subclass or derived class) to inherit properties and behaviors (methods) from an existing class (called a superclass or base class). Inheritance promotes code reusability and helps to establish a hierarchical relationship between classes.

# Here are some key points about inheritance:

# Superclass and Subclass:

# The class from which properties and behaviors are inherited is called the superclass or base class.
# The class that inherits properties and behaviors from the superclass is called the subclass or derived class.
# Syntax:

# In Python, inheritance is implemented by placing the name of the superclass in parentheses after the name of the subclass in the class definition.
# Subclasses can then access the methods and attributes of the superclass.
# Types of Inheritance:

# Single Inheritance: A subclass inherits from only one superclass.
# Multiple Inheritance: A subclass inherits from more than one superclass.
# Multilevel Inheritance: A subclass inherits from a superclass, and another subclass inherits from the first subclass, forming a chain of inheritance.
# Hierarchical Inheritance: Multiple subclasses inherit from the same superclass.
#Overriding Methods:

# Subclasses can override methods inherited from the superclass by providing a new implementation of the method with the same name.
# This allows subclasses to customize behavior while still reusing the structure of the superclass.
# Access Modifiers:

# Inheritance respects access modifiers like public, protected, and private.
# Subclasses can access public and protected members of the superclass, but not private members.
# Example of inheritance in Python:

class Animal:
    def sound(self):
        print("Animal makes a sound")

class Dog(Animal):  # Dog class inherits from Animal class
    def sound(self):  # Method overriding
        print("Dog barks")

class Cat(Animal):  # Cat class inherits from Animal class
    def sound(self):  # Method overriding
        print("Cat meows")

dog = Dog()
dog.sound()  # Output: Dog barks

cat = Cat()
cat.sound()  # Output: Cat meows
