# FileName: 16_GUI_Development_with_Tkinter.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# GUI Development with Tkinter in Python

# Tkinter is Python's standard GUI (Graphical User Interface) toolkit. 
# It provides a set of tools to create simple and complex graphical applications with ease.

import tkinter as tk

# Create a simple GUI window
root = tk.Tk()
root.title("Hello, Tkinter!")

# Add a label widget to the window
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# Start the GUI event loop
root.mainloop()
