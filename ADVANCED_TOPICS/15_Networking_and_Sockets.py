# FileName: 15_Networking_and_Sockets.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Networking and Sockets in Python

# Sockets are a low-level networking interface used to establish connections between computers over a network. 
# Python's socket module provides support for socket programming, allowing you to create client-server applications.

import socket

# Example of a simple server that echoes messages back to clients
def server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a host and port
    server_socket.bind(('localhost', 8888))

    # Listen for incoming connections
    server_socket.listen()

    print("Server listening on port 8888...")

    while True:
        # Accept a connection
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        # Receive data from the client
        data = client_socket.recv(1024)

        if not data:
            break

        # Echo the received data back to the client
        client_socket.sendall(data)

        # Close the connection
        client_socket.close()

    # Close the server socket
    server_socket.close()

# Example of a simple client that sends a message to the server
def client():
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect(('localhost', 8888))

    # Send a message to the server
    message = "Hello, server!"
    client_socket.sendall(message.encode())

    # Receive the echoed message from the se    rver
    echoed_message = client_socket.recv(1024)
    print("Received from server:", echoed_message.decode())

    # Close the connection
    client_socket.close()

# Run the server and client
if __name__ == "__main__":
    import threading

    # Start the server in a separate thread
    server_thread = threading.Thread(target=server)
    server_thread.start()

    # Allow some time for the server to start before running the client
    import time
    time.sleep(1)

    # Run the client
    client()
