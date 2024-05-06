"""
30. GraphQL Integration: Integrate GraphQL with Flask using Graphene.

pip install Flask Graphene Flask-GraphQL

{
  hello(name: "John")
}


{
  "data": {
    "hello": "Hello, John!"
  }
}


"""

from flask import Flask
from flask_graphql import GraphQLView
import graphene

app = Flask(__name__)

# Define a simple GraphQL schema
class Query(graphene.ObjectType):
    hello = graphene.String(name=graphene.String(default_value="World"))

    def resolve_hello(self, info, name):
        return f'Hello, {name}!'

schema = graphene.Schema(query=Query)

# Add a route for the GraphQL API endpoint
app.add_url_rule('/graphql', view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True))

if __name__ == '__main__':
    app.run(debug=True)
