# 48_Working with Dates and Times

        
# Working with dates and times in Python is made easy with the datetime module, which provides classes for manipulating dates and times. 
# Here are some key concepts and operations for working with dates and times in Python:

# Date and Time Classes:

# The datetime module provides several classes for representing dates, times, and combined date and time objects.
# The main classes are datetime.date for dates, datetime.time for times, and datetime.datetime for combined date and time objects.
# Additionally, there is a datetime.timedelta class for representing time differences or durations.
# Creating Date and Time Objects:

# You can create date, time, and datetime objects using their respective constructors or by parsing date and time strings.
# For example:

import datetime


# Create a date object
date_obj = datetime.date(2022, 3, 15)
print(date_obj)

# Create a time object
time_obj = datetime.time(9, 30)
print(time_obj)

# Create a datetime object
datetime_obj = datetime.datetime(2022, 3, 15, 9, 30)
print(datetime_obj)

# Parse a date string
parsed_date = datetime.datetime.strptime("2022-03-15", "%Y-%m-%d")
print(parsed_date)

# Formatting and Parsing:

# You can format date and time objects into strings using the strftime() method, which takes a format string specifying the desired format.
# Conversely, you can parse date and time strings into datetime objects using the strptime() function, which takes a string and a format specifier.
# For example:

# Format a datetime object into a string
formatted_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_str)

# Parse a date string into a datetime object
parsed_date = datetime.datetime.strptime("2022-03-15", "%Y-%m-%d")
print(parsed_date)

# Manipulating Dates and Times:

# You can perform arithmetic operations on datetime objects using timedelta objects to add or subtract days, hours, minutes, etc.
#For example:


# Create a timedelta object
delta = datetime.timedelta(days=1)
print(delta)

# Add one day to a datetime object
new_date = datetime_obj + delta
print(new_date)

# Subtract one day from a datetime object
new_date = datetime_obj - delta
print(new_date)