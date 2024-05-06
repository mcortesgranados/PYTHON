"""
19. Unit Testing: Write unit tests for your Flask application.

Explanation:
Test Setup: We define a setUp() method to set up the Flask app for testing. This method creates a test client and sets the testing attribute to True.
Unit Test: We define a test_index() method to test the index route ('/'). We make a GET request to the route using the test client and assert that the response status code is 200 and the response data is 'Hello, World!'.

"""