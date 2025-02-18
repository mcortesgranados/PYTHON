# Done by Manuel Cortés Granados
# March 19 2024 10:27 PM
import json # Importing the JSON module for JSON handling
import requests # Importing the requests module for making HTTP requests
from typing import Any, Dict, List, Union

url = "https://coderbyte.com/api/challenges/json/date-list"
response = requests.get(url)


#__define-ocg__
# Function to remove duplicate dictionatires from lists
def remove_duplicates(data: Union[Dict[str,Any],List[Any]])-> Union[Dict[str,Any],List[Any]]:
  """
  This functions removes duplicate dictionaries from lists.
  :param data: List or dictionary to be processed
  :return: List or dictionary without duplicate dictionaries
  """

  # Remove duplicate dictionaries from list
  unique_dicts = [] # Initializing an empty list to store unique dictionaries
  for item in data: # Iterating over each item in the data
    if isinstance(item,dict): # Checking if the item is a dictionary
      if item not in unique_dicts: # Checking if the dictionary is already in the unique_dicts
        unique_dicts.append(item) # Appending the dictionary to the unique_dicts lists
    elif isinstance(item,list): # Checking if the item is a list
      item = remove_duplicates(item) # Recursively call remove_duplicated for nested lists
      if item not in unique_dicts: # Appending the list to the unique_dicts list
        unique_dicts.append(item) # Appending the list to the unique_dicts list
  return unique_dicts # Returning the list without duplicated dictionaries


# Function to remove empty properties from dictionaries
def remove_empty_properties(data):
  """
  This function removes dictionary properties with all values set to an empty string or None
  :param data: Dictionary to be processed
  :return: Dictionary without empty properties
  """
  if isinstance(data,dict): # Checking if the data is a dictionary
    return {k: remove_empty_properties(v) for k,v in data.items() if v or v == 0} # Using dictionary comprehension to iterate over key-value pairs and remove empty properties
  elif isinstance(data,list): # Checking if the data is a list
    return [remove_empty_properties(v) for v in data] # Recursively call remove_empty_properties for nested lists
  else:
    return data # Returning the data as is if it's not a dictionary or a list

# Functions to sort dictionary keys alphabetically in a case-insensitive manner
def sort_dict_keys(data):
  """
  This function sorts dictionary keys alphabitelcally in a case-insensitive manner
  :param data: Dictionary to be processed.
  :return: Dictionary with keys sorted alphabetically
  """
  if isinstance(data,dict): # Checking if the data is a dictionary
    return {k: sort_dict_keys(v) for k,v in sorted(data.items(), key=lambda x:x[0].lower() )} # Sorting dictionary keys alphabetically ignoring cases
  if isinstance(data,list): # Checking if the data is a list
    return [sort_dict_keys(v) for v in data] # Recursively call sort_dict_keys for nested lists
  else:
    return data # Returning the data as is if it's not a dictionary or a list

# The following code performs the following tasks
# 1. Sends a get REQUEST to the specified URL.
# 2. Processes the JSON response data
# 3. Cleans the data by removing duplicate dictionaries and empty remove_empty_properties
# 4. Prints the cleaned data in JSON format

if (response.status_code==200): # Checking if the request was successful
  json_data = response.json() # Parsing the JSON response
  sorted_data = sort_dict_keys(json_data) # Sorting dictionary keys alphabetically
  unique_data = remove_duplicates(sorted_data) # Removing duplicate dictionaries
  cleaned_data = remove_empty_properties(unique_data) # Removing empty properties
  print(json.dumps(cleaned_data,indent=2)) # Printing the cleaned data in JSON format
else:
  print("Failed to fetch data from the API") # Printing an error message if the reuqest fails

