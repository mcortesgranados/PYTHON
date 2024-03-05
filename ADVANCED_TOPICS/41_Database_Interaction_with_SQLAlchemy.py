# FileName: 41_Database_Interaction_with_SQLAlchemy.py
# Code Written by Manuel Cortés Granados
# Date: 2024-03-04
# Location: Bogota DC, Colombia
# LinkedIn Profile: https://www.linkedin.com/in/mcortesgranados/

# Database Interaction with SQLAlchemy

# SQLAlchemy is a powerful SQL toolkit and Object-Relational Mapping (ORM) library for Python, providing an efficient way to interact with databases.

# Installation Instructions:
# Before running this code, you need to install SQLAlchemy. You can install it using pip:
# pip install SQLAlchemy

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create engine and session
engine = create_engine('sqlite:///example.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

# Define ORM classes
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

# Create tables
Base.metadata.create_all(engine)

# Add data to the database
user1 = User(name='Alice', age=30)
user2 = User(name='Bob', age=25)
session.add(user1)
session.add(user2)
session.commit()

# Query the database
users = session.query(User).all()
for user in users:
    print(f"User: {user.name}, Age: {user.age}")
