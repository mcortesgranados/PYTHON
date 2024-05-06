import unittest
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        """Set up the Flask app for testing."""
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        """Test the index route."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Hello, World!')

if __name__ == '__main__':
    unittest.main()
