import os
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
settings.configure()
