# FileName: 30_Cryptography_and_Security.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Cryptography and Security with Python

# Python provides libraries like hashlib and cryptography for cryptography and security tasks.

# Installation Instructions:
# You can install the required libraries using pip.
# For hashlib (usually comes with Python):
# No installation needed.
# For cryptography:
# pip install cryptography

import hashlib
from cryptography.fernet import Fernet

# Example 1: Hashing with hashlib

# Data to be hashed
data = b'Hello, World!'

# Generate a SHA-256 hash
sha256_hash = hashlib.sha256(data).hexdigest()
print("SHA-256 Hash:", sha256_hash)

# Example 2: Symmetric Encryption with cryptography

# Generate a key for symmetric encryption
key = Fernet.generate_key()

# Initialize a Fernet symmetric encryption object with the key
cipher = Fernet(key)

# Data to be encrypted
plaintext = b'Sensitive data'

# Encrypt the data
cipher_text = cipher.encrypt(plaintext)
print("Encrypted:", cipher_text)

# Decrypt the data
decrypted_text = cipher.decrypt(cipher_text)
print("Decrypted:", decrypted_text.decode())
