# Class Variables:

# Class variables are variables that are shared among all instances of a class.
# They are defined outside of any method of the class and are preceded by the class name.
# Class variables are accessed using the class name or through any instance of the class.
# Changes to class variables affect all instances of the class because they are shared across all instances.
# Example of class variables:

class Car:
    wheels = 4  # Class variable

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

car1 = Car("Toyota", "Corolla")
car2 = Car("Honda", "Civic")

print(car1.wheels)  # Output: 4
print(car2.wheels)  # Output: 4

Car.wheels = 3  # Change class variable
print(car1.wheels)  # Output: 3
print(car2.wheels)  # Output: 3
