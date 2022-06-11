---
title: Getting started with Python graphene
link: https://www.agiliq.com/blog/2019/08/getting-started-with-python-graphene/
author: Akshar
---

We will write an api endpoint which will respond to graphql queries.

- We will use `graphene` library to create our GraphQL service.
- We will expose this GraphQL service using Flask.
- We will consume the api from a browser or angular/react client.

This post assumes basic familiarity with [graphql](https://graphql.org/).

## Setup

Let's assume our application works with `Person` entity. A Person has a `first_name`
, `last_name` and `age`. Our graphql service will allow querying on `Person`.

Our queries would look like:

```
# Return all attributes of a single person.
http://localhost:5000/?search={ person {firstName lastName age}}

# Return firstName and lastName of a single person.
http://localhost:5000/?search={ person {firstName lastName}}

# Return firstName and lastName of all people in our system.
http://localhost:5000/?search={ people {firstName lastName}}

# Return all attributes of a person identified by key 1.
http://localhost:5000/?search={ person(key: 1) {firstName lastName age}}

```

In a real world application, we would fetch `Persons` from a database which
would be converted to Python class by the ORM. We want to focus on GraphQL
in this post and avoid database setup and interactions. So let's create a
`namedtuple` called `Person` and create few instances of it.

Let's write our code in `hello_graphene.py`.

```python
import collections

Person = collections.namedtuple(
    "Person",
    [
        'first_name',
        'last_name',
        'age'
    ]
)

data = {
    1: Person("Steve", "Jobs", 56),
    2: Person("Bill", "Gates", 63),
    3: Person("Ken", "Thompson", 76),
    4: Person("Guido", "Rossum", 63)
}

```

After this you should be able to do `person.first_name`, `person.last_name`
and `person.age`. Let's verify it from an `ipython` shell.

```repl
In [2]: from hello_graphene import data

In [3]: data
Out[3]:
{1: Person(first_name='Steve', last_name='Jobs', age=56),
 2: Person(first_name='Bill', last_name='Gates', age=63),
 3: Person(first_name='Ken', last_name='Thompson', age=76),
 4: Person(first_name='Guido', last_name='Rossum', age=63)}

In [4]: person = data[1]

In [5]: person.first_name
Out[5]: 'Steve'

In [6]: person.last_name
Out[6]: 'Jobs'

```

## Creating schema

Ensure that `graphene` is installed.

```bash
pip3 install graphene

```

GraphQL expects a root type. We want to make `person` field available on
root type.

We will have to write a `Person` type and the root type to accomplish this.
This should be in `hello_graphene.py`.

```python
from graphene import ObjectType, String, Int, Field

class PersonType(ObjectType):
    first_name = String()
    last_name = String()
    age = Int()

    def resolve_first_name(person, info):
        return person.first_name

    def resolve_last_name(person, info):
        return person.last_name

    def resolve_age(person, info):
        return person.age    

class Query(ObjectType):
    person = Field(PersonType)

    def resolve_person(root, info):
        return data[1]

```

Convention suggests that we name the root type class as `Query`.

Any GraphQL `type` we create must extend from `graphene.ObjectType`. GraphQL
dictates that there must be a resolver function for each field on each type.
That's whey we have `resolve_person` for `person`, `resolve_first_name` for
`first_name` and so on.

For now we have hardcoded the resolver for person to always return details
for person with key 1. We are fixing it soon, hang on.

We need to tell to GraphQL service that the root type is `Query`. The mechanism
to do that is to add a `Schema` instance.

```python
from graphene import Schema
schema = Schema(query=Query)

```

Our full `hello_graphene.py` looks like:

```python
import collections
from graphene import ObjectType, String, Schema, Int, Field

Person = collections.namedtuple(
    "Person",
    [
        'first_name',
        'last_name',
        'age'
    ]
)

data = {
    1: Person("Steve", "Jobs", 56),
    2: Person("Bill", "Gates", 63),
    3: Person("Ken", "Thompson", 76),
    4: Person("Guido", "Rossum", 63)
}

class PersonType(ObjectType):
    first_name = String()
    last_name = String()
    age = Int()

    def resolve_first_name(person, info):
        return person.first_name

    def resolve_last_name(person, info):
        return person.last_name

    def resolve_age(person, info):
        return person.age    

class Query(ObjectType):
    person = Field(PersonType)

    def resolve_person(root, info):
        return data[1]

schema = Schema(query=Query)

```

Our GraphQL service is ready now.

## Executing queries

Let's execute a GraphQL query from the shell.

```repl
In [3]: from hello_graphene import schema

In [7]: query = '{person {firstName lastName age} }'

In [8]: result = schema.execute(query)

In [9]: result.data
Out[9]:
OrderedDict([('person',
              OrderedDict([('firstName', 'Steve'),
                           ('lastName', 'Jobs'),
                           ('age', 56)]))])

```

Notice how the result datastructure has the same structure as the query.

Let's execute one more query to ensure that the service only returns the
requested fields.

```repl
In [10]: query = '{person {firstName} }'

In [11]: result = schema.execute(query)

In [12]: result.data
Out[12]: OrderedDict([('person', OrderedDict([('firstName', 'Steve')]))])

```

We want to expose the service on an endpoint so that browser or any client
can consume the service. Let's expose a Flask endpoint.

```python
# flask_graphql.py

import json
from flask import Flask, request

from hello_graphene import schema

app = Flask(__name__)

@app.route('/graphql')
def graphql():
    query = request.args.get('query')
    result = schema.execute(query)
    d = json.dumps(result.data)
    return '{}'.format(d)

```

Start the flask server.

```bash
$ export FLASK_APP=flask_graphql.py
$ flask run

```

Let's make a request to flask app with a GraphQL query.

![](https://www.agiliq.com/assets/images/graphql/person-first-name.png)

Let's modify the GraphQL service to allow getting details of any person.
This requires using GraphQL arguments.

We need to allow arguments on `person` field. Modify `person` to look like:

```python
person = Field(PersonType, key=Int())

```

We will have to modify resolver for person to accomodate the argument too.

```python
def resolve_person(root, info, key):
    return data[key]

```

Restart the shell and get data for `steve` and `bill`.

```repl
In [1]: from hello_graphene import schema

In [2]: query = '{person(key: 1) {firstName} }'

In [3]: schema.execute(query).data
Out[3]: OrderedDict([('person', OrderedDict([('firstName', 'steve')]))])

In [4]: query = '{person(key: 2) {firstName} }'

In [5]: schema.execute(query).data
Out[5]: OrderedDict([('person', OrderedDict([('firstName', 'bill')]))])

```

We could have named the argument anything. We could have named it `identifier`
instead of `key`.

```python
person = Field(PersonType, identifier=Int())

def resolve_person(root, info, identifier):
    return data[identifier]

```

Let's use the api from browser and get data for `bill`.

![](https://www.agiliq.com/assets/images/graphql/person-detail-details.png)

Ideally the api would be consumed from a mobile client or from a single page
application.

If person is retrieved from a database using SQLAlchemy, then the argument
would probably be named `id` and the resolver would look something like:

```python
person = Field(PersonType, id=Int())

def resolve_person(root, info, id):
    return Person.query.get(id)

```

## Fetching a list of people

We want our service to return details of all people in our system. Let's
add a field called `people` on the root type.

```python
from graphene import List

class Query(ObjectType):
    person = Field(PersonType, key=Int())
    people = List(PersonType)

    def resolve_person(root, info, key):
        return data[key]

    def resolve_people(root, info):
        return data.values()

```

Since `people` field provides a list of people, so we set it's type as `graphene.List`.
Each entry of the list would be a `PersonType`.

```repl
In [1]: from hello_graphene import schema

In [2]: query = '{people {firstName age} }'

In [3]: schema.execute(query).data
Out[3]:
OrderedDict([('people',
              [OrderedDict([('firstName', 'steve'), ('age', 56)]),
               OrderedDict([('firstName', 'bill'), ('age', 63)]),
               OrderedDict([('firstName', 'ken'), ('age', 76)]),
               OrderedDict([('firstName', 'guido'), ('age', 63)])])])

```

We can get all people from a client by calling
`localhost:5000/?query={ people {firstName lastName age} }`.

## Supporting defaults

Currently we cannot query on `person` without `key`. Let's try a query which
would cause an exception.

```repl
In [6]: query = '{person {firstName} }'

In [7]: schema.execute(query).data
TypeError: resolve_person() missing 1 required positional argument: 'key'
Traceback (most recent call last):
  File "/Users/akshar/Envs/gryffindor/lib/python3.6/site-packages/graphql/execution/executor.py", line 450, in resolve_or_error
    return executor.execute(resolve_fn, source, info, **args)
  File "/Users/akshar/Envs/gryffindor/lib/python3.6/site-packages/graphql/execution/executors/sync.py", line 16, in execute
    return fn(*args, **kwargs)
graphql.error.located_error.GraphQLLocatedError: resolve_person() missing 1 required positional argument: 'key'

```

If person's key isn't provided, then we want to respond with `steve`'s details.
We can accomplish this by setting a `default_value` on `person` and this default_value
should contain `steve`'s key.

```python
person = Field(PersonType, key=Int(default_value=1))

```

Restart the shell and try the query once more.

```repl
In [4]: query = '{person {firstName} }'

In [5]: schema.execute(query).data
Out[5]: OrderedDict([('person', OrderedDict([('firstName', 'steve')]))])

```

If we pass a `key` though, then corresponding person's details would be fetched.

```repl
In [6]: query = '{person(key: 2) {firstName} }'

In [7]: schema.execute(query).data
Out[7]: OrderedDict([('person', OrderedDict([('firstName', 'bill')]))])

```
