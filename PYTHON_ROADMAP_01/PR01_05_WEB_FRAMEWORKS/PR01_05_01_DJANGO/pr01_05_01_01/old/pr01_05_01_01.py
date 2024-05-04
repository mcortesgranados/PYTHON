"""
01. Creating a Basic Django App: Set up a basic Django project and create an app.

"""

# Step 1: Install Django if you haven't already
# You can install Django using pip (Python's package manager) by running the following command in your terminal or command prompt:
# pip install django

# Step 2: Create a Django project
# To create a Django project, run the following command in your terminal or command prompt:
# django-admin startproject myproject
# This will create a directory named "myproject" with the necessary files and folders for a Django project.

# Step 3: Navigate into the project directory
# Change your current directory to the newly created project directory:
# cd myproject

# Step 4: Create a Django app
# Inside the Django project, you can create one or more apps. Each app represents a distinct component of your project.
# To create a Django app, run the following command in your terminal or command prompt:
# python manage.py startapp myapp
# This will create a directory named "myapp" with the necessary files and folders for a Django app.

# Step 5: Define models (optional)
# If your app requires a database, you can define models to represent the data. Models are typically defined in the "models.py" file within the app directory.
# Here's an example of a simple model:
from django.db import models

class MyModel(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()

# Step 6: Define views
# Views handle the presentation logic of your app. They receive HTTP requests and return HTTP responses. Views are typically defined in the "views.py" file within the app directory.
# Here's an example of a simple view:
from django.http import HttpResponse

def my_view(request):
    return HttpResponse("Hello, Django!")

# Step 7: Define URLs
# URL patterns map URLs to views. They are typically defined in the "urls.py" file within the app directory.
# Here's an example of defining a URL pattern:
from django.urls import path
from . import views

urlpatterns = [
    path('myview/', views.my_view),
]

# Step 8: Register the app
# To use the app in your Django project, you need to register it in the project's settings. Open the "settings.py" file in the project directory and add the name of your app to the "INSTALLED_APPS" list.

# Step 9: Run the development server
# You can now run the development server to test your Django app. Run the following command in your terminal or command prompt:
# python manage.py runserver
# This will start the development server, and you can access your app at http://127.0.0.1:8000/

# Step 10: Test your app
# Open a web browser and navigate to http://127.0.0.1:8000/myview/ to see the output of your view.

# Step 11: Congratulations!
# You have successfully created a basic Django app. You can now continue to build and customize your app further as needed.
