
# Polymorphism is a fundamental concept in object-oriented programming (OOP) that allows objects of different classes to be treated as 
# objects of a common superclass. It enables a single interface (method or function) to be used for objects of different types, providing 
# flexibility and extensibility in code design.

# Here are some key points about polymorphism:

# Polymorphic Behavior: Polymorphism allows objects of different classes to respond to the same message (method call) in different ways. 
# This means that the behavior of a method can vary depending on the type of object it is called on.

# Method Overriding: Polymorphism is often achieved through method overriding, where a subclass provides a specific implementation of a 
# method that is already defined in its superclass. This allows the subclass to customize the behavior of inherited methods.

# Dynamic Binding: Polymorphism enables dynamic method binding, where the appropriate method to call is determined at runtime based on 
# the type of object. This allows for flexibility in code execution and facilitates dynamic dispatch.

# Compile-Time and Run-Time Polymorphism: Polymorphism can be categorized into compile-time polymorphism (method overloading) 
# and run-time polymorphism (method overriding). Method overloading involves defining multiple methods with the same name but different parameter
# lists, while method overriding involves providing a specific implementation of a method in a subclass.

# Code Reusability: Polymorphism promotes code reusability by allowing a single interface to be used for objects of different types. 
# This reduces code duplication and promotes modular and flexible code design.

class Animal:
    def sound(self):
        pass

class Dog(Animal):
    def sound(self):  # Method overriding
        print("Dog barks")

class Cat(Animal):
    def sound(self):  # Method overriding
        print("Cat meows")

# Function that demonstrates polymorphism
def make_sound(animal):
    animal.sound()  # Polymorphic behavior

# Creating instances of Dog and Cat
dog = Dog()
cat = Cat()

# Calling make_sound function with different types of animals
make_sound(dog)  # Output: Dog barks
make_sound(cat)  # Output: Cat meows

# In this example, the Animal class serves as a common superclass, and the Dog and Cat classes are subclasses that override the sound()
# method inherited from Animal. The make_sound() function demonstrates polymorphic behavior by accepting objects of different types (Dog and Cat)
# and calling the sound() method on each object, resulting in different outputs based on the type of animal.