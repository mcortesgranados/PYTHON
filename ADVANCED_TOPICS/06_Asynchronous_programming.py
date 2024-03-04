import asyncio
import datetime

# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Asynchronous Programming with asyncio in Python

# Asynchronous programming allows you to write non-blocking code that can perform multiple tasks concurrently.
# The asyncio module in Python provides a framework for asynchronous programming using coroutines and event loops.

# Define an asynchronous coroutine function to simulate a task with a delay
async def task_with_delay(task_name, delay):
    print(f"{task_name} started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    await asyncio.sleep(delay)
    print(f"{task_name} finished at {datetime.datetime.now().strftime('%H:%M:%S')}")

# Header for the output
print("Demo code of Asynchronous Programming with asyncio in Python")
print("Authored by Manuel Cortés Granados")
print("Date:", current_time)
print("LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/")
print("-" * 50)  # Separator for a nice display

# Create an event loop
async def main():
    # Define tasks with delays
    tasks = [
        task_with_delay("Task 1", 2),
        task_with_delay("Task 2", 1),
        task_with_delay("Task 3", 3)
    ]

    # Run tasks concurrently
    await asyncio.gather(*tasks)

# Run the event loop
asyncio.run(main())

# Output:
# Demo code of Asynchronous Programming with asyncio in Python
# Authored by Manuel Cortés Granados
# Date: 2024-03-04 12:30:45
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
# --------------------------------------------------
# Task 1 started at ...
# Task 2 started at ...
# Task 3 started at ...
# Task 2 finished at ...
# Task 1 finished at ...
# Task 3 finished at ...

# Asynchronous programming with asyncio enables efficient handling of I/O-bound and CPU-bound tasks,
# improving performance and responsiveness of applications.
