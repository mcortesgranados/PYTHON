# In this example:

# We define a class called Person.
# Inside the class, we define the __init__() method, which takes self, name, and age as parameters.
# Inside the constructor, we initialize two attributes name and age with the values passed as arguments during object creation.
# When we create an instance of the Person class (person1), the __init__() method is automatically called with the provided arguments "Alice" and 30.
# We can then access the attributes name and age of the person1 object using dot notation (person1.name, person1.age).
# The constructor can be used to perform any necessary setup or initialization for the object. It can accept any number of parameters, and 
# you can define default values for parameters if needed.

# It's important to note that the first parameter of the constructor method is always self, which refers to the instance of the class itself. 
# It is used to access other attributes and methods within the class.

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Creating an instance of the Person class
person1 = Person("Alice", 30)

# Accessing attributes of the object
print(person1.name)  # Output: Alice
print(person1.age)   # Output: 30
