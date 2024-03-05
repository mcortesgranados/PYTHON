# FileName: 44_Distributed_Computing_with_Pyro.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Distributed Computing with Pyro (Python Remote Objects)

# Pyro is a library for distributed computing in Python, allowing objects to be manipulated remotely across processes or even over a network.

# Installation Instructions:
# Before running this code, you need to install Pyro5. You can install it using pip:
# pip install Pyro5

import Pyro5.api

# Define a remote object
class RemoteCalculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

# Create a Pyro daemon
daemon = Pyro5.api.Daemon()

# Register the remote object
uri = daemon.register(RemoteCalculator)

# Print the URI
print("URI of the remote object:", uri)

# Start the Pyro event loop
daemon.requestLoop()
