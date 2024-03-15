# Instance variables and class variables are both types of variables in Python, but they serve different purposes and have different scopes.

# Instance Variables:

# Instance variables are variables that are unique to each instance (object) of a class.
# They are defined within methods of the class and are preceded by the self keyword.
# Each instance of the class has its own copy of instance variables, and changes to one instance variable do not affect other instances.
# Instance variables represent the state of individual objects and are used to store data that varies from one object to another.
# Example of instance variables:

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

car1 = Car("Toyota", "Corolla")
car2 = Car("Honda", "Civic")

print(car1.brand)  # Output: Toyota
print(car2.brand)  # Output: Honda
