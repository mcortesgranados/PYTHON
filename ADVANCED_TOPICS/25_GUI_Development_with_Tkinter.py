# FileName: 25_GUI_Development_with_Tkinter.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# GUI Development with Tkinter in Python

# Tkinter is Python's standard GUI (Graphical User Interface) toolkit.
# It provides a set of tools for creating desktop applications with graphical user interfaces.

import tkinter as tk

# Example: Creating a simple GUI window with Tkinter

# Create main application window
root = tk.Tk()
root.title("Hello, Tkinter!")

# Create a label widget
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# Run the main event loop
root.mainloop()
