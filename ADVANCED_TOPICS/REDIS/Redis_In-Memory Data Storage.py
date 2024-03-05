# FileName: 01_redis_example.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04 10:30 (GMT -5:00)
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/
#
# Additional Libraries Required:
# To run this script, you need to install the 'redis' library.
# You can install it via pip by running the following command:
# pip install redis

import redis
import datetime

# Connect to Redis server
# Create a connection to the Redis server running on localhost, port 6379,
# and using the default database (db=0).
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Set a key-value pair in Redis
# Set the key 'my_key' with the value 'Hello Redis!' in Redis.
redis_client.set('my_key', 'Hello Redis!')

# Get the value associated with the key
# Retrieve the value associated with the key 'my_key' from Redis.
value = redis_client.get('my_key')

# Print the retrieved value
# Print the retrieved value after decoding it from bytes to a string.
print(value.decode('utf-8'))  # Decode bytes to string

# Explanation:
# This Python script demonstrates basic usage of the Redis client library.
# It establishes a connection to a Redis server, sets a key-value pair, retrieves
# the value associated with a key, and prints it.

# Item: In-Memory Data Storage
# In-Memory Data Storage:
# Redis stores data primarily in RAM, which allows for extremely fast read and write operations.
# This means that data is held in memory rather than being persisted to disk, making Redis
# an excellent choice for applications that require low-latency access to frequently accessed data.
# Here are some additional details about Redis's in-memory data storage:

# 1. Performance: Storing data in memory enables Redis to achieve incredibly fast read and write speeds,
# making it suitable for use cases where high-performance data access is crucial. With read and write
# operations typically taking just a few microseconds, Redis can handle a large number of requests per
# second, making it ideal for real-time applications and caching.

# 2. Data Durability: While Redis primarily stores data in memory, it also provides options for data
# persistence to ensure durability. Redis offers two main persistence mechanisms: snapshotting and
# append-only file (AOF) persistence. Snapshotting involves periodically writing a snapshot of the
# dataset to disk, while AOF persistence logs every write operation, ensuring that data can be
# recovered in the event of a system failure.

# 3. Cache Use Case: One of the most common use cases for Redis is caching. By storing frequently accessed
# data in memory, Redis can dramatically reduce the latency of read operations compared to fetching data
# from disk-based databases. This makes Redis an excellent choice for caching session data, frequently
# accessed database queries, and other types of data that benefit from low-latency access.

# 4. Data Size Consideration: Since Redis stores data in memory, the available memory on the server limits
# the amount of data that can be stored. It's essential to monitor memory usage carefully and ensure that
# Redis has enough memory to accommodate the data size and workload requirements. Redis also provides
# mechanisms for eviction and expiration to manage memory usage effectively.

# 5. Data Persistence and Durability: While Redis primarily relies on in-memory storage for performance,
# it offers options for data persistence to ensure durability. These persistence mechanisms, such as
# snapshotting and AOF, provide different trade-offs between performance and durability, allowing users
# to choose the approach that best fits their application requirements.
