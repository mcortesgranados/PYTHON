# FileName: 24_Networking_and_Sockets.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Networking and Sockets in Python

# Python provides robust support for networking and socket programming through the socket module.
# Sockets are endpoints of a bidirectional communication channel in networking.

import socket

# Example: Creating a simple TCP server and client

# TCP server
def tcp_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"TCP server listening on {host}:{port}")
        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected to {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Received data: {data.decode()}")
                conn.sendall(data.upper())

# TCP client
def tcp_client(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        message = "Hello, TCP Server!"
        client_socket.sendall(message.encode())
        data = client_socket.recv(1024)
        print(f"Received response from server: {data.decode()}")

# Test TCP server and client
host = "127.0.0.1"
port = 12345
tcp_server(host, port)
tcp_client(host, port)
