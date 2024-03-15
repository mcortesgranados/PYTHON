
# Method overriding is a concept in object-oriented programming (OOP) where a subclass provides a specific implementation of a method that is
# already defined in its superclass. This allows the subclass to customize the behavior of inherited methods without modifying the superclass.

# Here are some key points about method overriding:

# Inheritance: Method overriding is closely related to inheritance. When a subclass inherits a method from its superclass, it can choose to override 
# that method with its own implementation.

# Syntax:

# To override a method in Python, simply define a method with the same name and signature (parameters) in the subclass as the one in the superclass.
# The method in the subclass replaces (overrides) the method with the same name in the superclass.

# Access to Superclass Method:

# In the overridden method, if you want to call the superclass method, you can use the super() function followed by dot notation (super().method_name()).
# This allows you to access the superclass's implementation of the method and extend or modify it as needed.
# Purpose:

# Method overriding allows subclasses to provide a more specific implementation of a method defined in the superclass.
# It promotes code reusability and helps to create a more modular and flexible codebase.

# Dynamic Polymorphism:

# Method overriding is one of the key mechanisms that enable dynamic polymorphism, where the appropriate method to call is determined at runtime based
#  on the type of object.
#Example of method overriding in Python:

class Animal:
    def sound(self):
        print("Animal makes a sound")

class Dog(Animal):
    def sound(self):  # Method overriding
        print("Dog barks")

class Cat(Animal):
    def sound(self):  # Method overriding
        print("Cat meows")

dog = Dog()
dog.sound()  # Output: Dog barks

cat = Cat()
cat.sound()  # Output: Cat meows
