# FileName: 46_Interfacing_with_Other_Languages.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Interfacing with Other Languages (using ctypes)

# Python's ctypes module allows interfacing with C libraries by loading dynamic link libraries (DLLs) and accessing functions within them.

import ctypes

# Load the C library
libc = ctypes.CDLL('libc.so.6')

# Example: Call the C library function getpid() to get the process ID
pid = libc.getpid()
print("Process ID:", pid)
