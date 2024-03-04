# FileName: 08_Singleton.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Singleton Design Pattern in Python

# The Singleton design pattern ensures that a class has only one instance and provides a global point of access to it.

class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Example method of the singleton class
    def show_message(self):
        print("Hello, I am a Singleton!")

# Usage example
if __name__ == "__main__":
    # Creating singleton objects
    singleton1 = Singleton()
    singleton2 = Singleton()

    # Checking if both objects are the same instance
    print("Are singleton1 and singleton2 the same instance?", singleton1 is singleton2)

    # Calling a method on the singleton instance
    singleton1.show_message()
