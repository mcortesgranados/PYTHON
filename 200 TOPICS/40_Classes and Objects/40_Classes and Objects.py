# Define a class named Person
class Person:
    # Constructor method to initialize attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # Method to display information about the person
    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}")

# Create an object (instance) of the Person class
person1 = Person("John", 30)

# Accessing attributes of the object
print("Name of person1:", person1.name)
print("Age of person1:", person1.age)

# Calling a method of the object
person1.display_info()

# Create another object of the Person class
person2 = Person("Alice", 25)
person2.display_info()
