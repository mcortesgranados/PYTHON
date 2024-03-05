# FileName: 32_Concurrency_Control.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Concurrency Control in Python

# Python provides mechanisms like locks and semaphores for controlling access to shared resources in concurrent programming.

import threading

# Example: Concurrency control with locks

# Shared resource
counter = 0

# Create a lock
lock = threading.Lock()

# Function to increment the counter
def increment_counter():
    global counter
    for _ in range(1000000):
        # Acquire the lock before accessing the shared resource
        lock.acquire()
        counter += 1
        # Release the lock after accessing the shared resource
        lock.release()

# Create multiple threads to increment the counter concurrently
threads = []
for _ in range(5):
    thread = threading.Thread(target=increment_counter)
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("Counter:", counter)
