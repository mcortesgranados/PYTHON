import datetime
import threading
import multiprocessing

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Multi-threading and Multiprocessing in Python

# Multi-threading and multiprocessing are techniques used to execute multiple tasks concurrently in Python.
# They are particularly useful for CPU-bound and I/O-bound tasks, respectively.

# Define a function to simulate a CPU-bound task
def cpu_bound_task(task_name, iterations):
    print(f"{task_name} started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    for i in range(iterations):
        result = 0
        for j in range(1000000):
            result += j
    print(f"{task_name} finished at {datetime.datetime.now().strftime('%H:%M:%S')}")

# Define a function to simulate an I/O-bound task
def io_bound_task(task_name, delay):
    print(f"{task_name} started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    import time
    time.sleep(delay)
    print(f"{task_name} finished at {datetime.datetime.now().strftime('%H:%M:%S')}")

# Header for the output
print("Demo code of Multi-threading and Multiprocessing in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Create threads for CPU-bound tasks
thread1 = threading.Thread(target=cpu_bound_task, args=("CPU Task 1", 3))
thread2 = threading.Thread(target=cpu_bound_task, args=("CPU Task 2", 3))

# Start threads
thread1.start()
thread2.start()

# Wait for threads to finish
thread1.join()
thread2.join()

# Create processes for I/O-bound tasks
process1 = multiprocessing.Process(target=io_bound_task, args=("I/O Task 1", 2))
process2 = multiprocessing.Process(target=io_bound_task, args=("I/O Task 2", 1))

# Start processes
process1.start()
process2.start()

# Wait for processes to finish
process1.join()
process2.join()

# Output:
# Demo code of Multi-threading and Multiprocessing in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------
# CPU Task 1 started at ...
# CPU Task 2 started at ...
# CPU Task 1 finished at ...
# CPU Task 2 finished at ...
# I/O Task 2 started at ...
# I/O Task 1 started at ...
# I/O Task 2 finished at ...
# I/O Task 1 finished at ...
