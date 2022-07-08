---
title: "NLP Full-Text Search With Elasticsearch & spaCy"
link: "https://www.viko.co.uk/blog/nlp-full-text-search/"
publish: "22 November 2021"
author: "Toby Hobson"
---

![](https://www.viko.co.uk/images/blog/nlp-full-text-search/primary.svg)

Natural language processing, in particular natural language understanding,
allows us to fully understand the intent behind search queries. This lets
us offer far more targeted search results along with a much improved user
experience. In this tutorial I'll show you how to compliment Elasticsearch
with Named Entity Recognition (NER). We'll use spaCy, a simple yet powerful
NLP framework.

## Python POC

To make things easy to follow, I've created a [Python POC](https://github.com/viko-ai/nlp-search-poc)
which demonstrates how to use spaCy and Elasticsearch together. There are
4 branches of the code, representing different stages of development. Now's
the time to clone the repo and view the branches:

```bash
$ git clone git@github.com:viko-ai/nlp-search-poc.git
$ cd nlp-search-poc
$ git branch -a
* master
  remotes/origin/1.0-es-basic
  remotes/origin/1.5-es-improved
  remotes/origin/2.0-ner-basic
  remotes/origin/2.5-ner-improved
  remotes/origin/HEAD -> origin/master
  remotes/origin/master

```

We'll start out with a simple Elasticsearch implementation without NLP.
This will allow me to illustrate the problem we're trying to solve. Next
we'll try to improve on this by improving the Elasticsearch query, without
resorting to NLP. We'll then add [spaCy's NER](https://spacy.io/usage/linguistic-features#named-entities)
and see how it solves the issue. Finally, we'll build on this to make it
even more powerful.

## The problem

Firstly let's understand why we need NLU. To understand NLP/NLP, and it's application for search, let's look at a simple search query:

> 'A lightweight jacket'

As humans, we immediately understand that the shopper is looking for a jacket,
ideally something lightweight. 'jacket' is a noun and 'lightweight' is an
adjective that qualifies the word 'jacket'. Elasticsearch doesn't understand
nouns and adjectives. To a full text search engine these are just terms or
tokens. Why is this a problem?

Let me give you a (slightly contrived) scenario. You are developing the catalog
search feature for an outdoor clothing and equipment store. The store sells
a wide range of lightweight jackets, waterproof/windproof jackets. The store
also sells a 'packable mosquito net'. Imagine a shopper searches for a
**'packable jacket'**. It's obvious they are looking for a jacket but what
will Elasticsearch find when searching against the title/description fields?

If there is **'packable jacket'** in the catalog it will find it. But what
if you only have a lightweight jacket? Elasticsearch will have a partial
match against two documents: the 'lightweight **jacket**' and '**packable**
mosquito net'. The underlying **TF-IDF** algorithm favors fields containing
less frequently used terms, which in this case is likely to be packable.
So it will pick the mosquito net.

> **TF-IDF often performs poorly for long tail searches**

## Search filters

Let's look at another query:

> 'lightweight jacket less than $300'

Again it's immediately obvious what the user is looking for. However, Elasticsearch will not work here, at least not without some regex preprocessing. We usually work around these issues by offering search filters. In this case the user would typically:

1. Search for 'lightweight jacket'
2. Filter by jackets only
3. Filter by max price $300

It works, but it's not an ideal experience. This is especially true for users
on mobile devices who need to mess around with dropdowns/accordions and
other paraphernalia. It would be much better if we could fully understand
the initial request and already apply the relevant filters. Fortunately
this is all possible with **natural language understanding**

## Getting started

Hopefully you've already cloned the POC, If not go ahead and do this. Now
switch to the 1.0-es-basic branch:

```bash
$ git checkout 1.0-es-basic 
Branch '1.0-es-basic' set up to track remote branch '1.0-es-basic' from 'origin'.
Switched to a new branch '1.0-es-basic'

```

**Follow the instructions in the README** (make sure you're viewing the `1.0-es-branch`)

**TLDR;**

No time (or inclination) to read the README. Follow these steps:

```bash
$ git checkout 1.0-es-basic
# Setup a virtual env
$ pyenv virtualenv 3.9.7 nlp-search
$ pyenv virtualenv local nlp-search
$ pip install -U pip
$ pip install -r requirements.txt
# Run elasticsearch in Docker
$ docker-compose up -d elasticsearch-7
$ python -m src.tools ping
Elasticsearch alive: True
# Ingest the test data
$ python -m src.tools create
$ python -m src.tools ingest
# Run the uvicorn server
$ bin/server.sh

```

It's a good idea to take a look at the test data `data/products.json` at
this point.

Take a look at `src/product_repository.py`. It's very basic - the full text
search simply searches against the *title* field using a match query:

```astro
es_query = {
    'match': {
        'title': query
    }
}

```

Having imported the test data try querying for `'packable jacket'` using
the provided client:

```astro
$ python -m src.client 'packable jacket'

```

```astro
{
  "results": [
    {
      "title": "packable mosquito net",
      "product_type": "mosquito net",
      "price": 10.0,
      "attrs": [
        "packable"
      ]
    }
  ]
}

```

As expected, it returns the mosquito net not the jacket.

## Improving the query

This setup is absolutely trivial. Let's try making the query a bit more
granular. Instead of just searching against the `title` field, we'll search
against the `product_type` and `attrs` fields:

```astro
es_query = {
    'bool': {
        'must': {
            'match': {
                'product_type': query
            }
        },
        'should': {
            'match': {
                'attrs': query
            }
        }
    }
}

```

Kill the running server and rerun it using the code in the `1.5-es-improved branch`:

```bash
# Kill the server using ctrl-C
$ git checkout 1.5-es-improved
Branch '1.5-es-improved' set up to track remote branch '1.5-es-improved' from 'origin'.
Switched to a new branch '1.5-es-improved'
$ bin/server.sh
uvicorn.error  INFO  Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)  

```

I updated the README in this branch, so feel free to read the more detailed
instructions there.

Query again:

```astro
$ python -m src.client 'packable jacket'

```

```astro
{
  "results": [
    {
      "title": "lightweight black jacket",
      "product_type": "jacket",
      "price": 100.0,
      "attrs": [
        "lightweight",
        "black"
      ]
    }
  ]
}

```

It works!

Don't get too excited though. Replace the existing test data with the new
data in the `1.5-es-improved` branch:

```bash
$ git status
On branch 1.5-es-improved
$ python -m src.tools reset
...
productRepository  INFO  Ingesting packable travel bag

```

Query again ...

```astro
$ python -m src.client 'packable jacket'

```

```astro
{
  "results": [
    {
      "title": "packable travel bag",
      "product_type": "packable bag",
      "price": 20.0,
      "attrs": [
        "packable"
      ]
    }
  ]
}

```

## The underlying problem

This scenario illustrates the fundamental challenge associated with elasticsearch
and other full text search engines. **We use the database structure to infer
intent, instead of fully understanding the query itself**. In this case we
assumed that the `product_type` field is more important than `attrs`, so
attached more weight to it. We could have applied boosting and other tricks
to achieve the same result. It worked until we added a document with 'packable'
in its `product_type` field.

Without NLP/NLU, we really only have two tools at our disposal:

1. Improve the search results by attaching more weight to certain fields
in the document
2. Augment text search with structured search filters

Ideally we want to really understand what the shopper is asking for. That's
where Natural Language Processing comes into its own.

## Introducing NLP using spaCy

Check out the **2.0-ner-basic** branch:

```astro
# Kill server with ctrl-c
$ git checkout 2.0-ner-basic
Branch '2.0-ner-basic' set up to track remote branch '2.0-ner-basic' from 'origin'.
Switched to a new branch '2.0-ner-basic'

```

Again, I've updated the README for this branch so please take a look

**TLDR;**

No time (or inclination) to read the README. Follow these steps:

```astro
# install the new spaCy dependency
$ pip install -r requirements.txt
# start the server again
$ bin/server.sh

```

This branch introduces a new class `NerPredictor` in `src/ner_predictor.py`.
This is a **really basic** implementation of **Named Entity Recognition (NER)**
using a very basic NLP model. Don't be fooled into thinking this model is
suitable for production use, it's not! However, it does illustrate the concept
well. Named Entity Recognition is able to identify products and product
attributes in a search query, turning full text search into structured search:

```astro
# src/product_repository.py
es_query = {
    'bool': {
        'must': {
            'match': {
                'title': product
            }
        },
        'should': {
            'match': {
                'attrs': attrs
            }
        }
    }
}

```

Notice how we can now explicitly query for the desired product along with
the product attributes.

Let's try the same query again:

```astro
$ python -m src.client 'packable jacket'

```

```astro
{
  "ner_prediction": {
    "text": "packable jacket",
    "product": "jacket",
    "attrs": [
      "packable"
    ]
  },
  "results": [
    {
      "title": "lightweight black jacket",
      "product_type": "jacket",
      "price": 100.0,
      "attrs": [
        "lightweight",
        "black"
      ]
    }
  ]
}

```

The search response also includes spaCy's NER prediction. It's useful to
check this to understand how the terms will be fed to the Elasticsearch query.
Now that we understand the user is seeking a jacket we can ensure the search
results are actually jackets.

> **Why not use regex matching?**
> 
> At this point you be thinking this is just hype. Why not simply use regex
> to identify jackets or 'lightweight'? For this trivial example we could
> indeed search for 'jacket' in the query text and assume it's the product.
> We could do the same for commonly used attributes. We may even go a step
> further and just assume that the last word in the phrase is the product
> and the words before it are adjectives.
> 
> This is not a real world solution. We'd likely have tens or hundreds of
> thousands products to include and the list of adjectives is almost infinite.
> We can't even assume the last word is the product - how would we distinguish
> between a 'mosquito net' and 'fishing net'.
> 
> NLP, like all artificial intelligence is able to generalise. We train a
> model with plenty of examples and let it decide what is a product vs an
> attribute. It's statistical in nature, and it's not perfect. However, unlike
> rule based solutions, the code complexity remains constant, no matter how
> many scenarios we need to handle.

## Identifying other entities

The previous example uses NER to identify products and product attributes.
Let's extend this to also identify colors and prices. Check out the `2.5-ner-improved`
branch and import the new data which includes `color` and `price` fields:

```astro
# kill server with ctrl-c
$ git checkout 2.5-ner-improved
Branch '2.5-ner-improved' set up to track remote branch '2.5-ner-improved' from 'origin'.
Switched to a new branch '2.5-ner-improved'
# Ingest the new data with colors and prices
$ python -m src.tools reset
productRepository INFO Ingesting burgundy organic cotton jacket
$ bin/server.sh

```

This is a bit more complex so let's break it down.

The nlp model has been trained with colors and prices. We've also introduced
a new NER entity `PRICE` alongside `PRODUCT` and `ATTRIBUTE`. The Elasticsearch
query has been updated to query across the `title, attrs, color` and price `fields`.

Try querying for a specific color jacket within a price range:

```astro
$ python -m src.client 'black waterproof jacket less than $200'

```

```astro
{
  "results": [
    {
      "title": "waterproof black jacket",
      "product_type": "jacket",
      "price": 150.0,
      "colors": [
        "black"
      ],
      "attrs": [
        "waterproof"
      ]
    }
  ]
}

```

For a production implementation we would use the NER prediction not only
to feed the elasticsearch query but also to pre-select relevant search filters.
In our case we might apply a `category` filter of jacket, `color` of black
and `price_to` of `$200`. Of course, we would extend it to also identify the
gender if specified in the query e.g. 'mens waterproof jacket'.

> **Why no color attribute?**
> 
> You may wonder why we introduced a new `PRICE` attribute but no `COLOR`.
> There are a couple of reasons for this.
> 
> In a real world e-commerce application, a color filter would be restricted
> to a small finite set or colors. Full text search allows the user to type
> anything. Being statistical, the NER model may identify colours that are
> not in the search filter. It may identify colors that didn't even appear
> in the NLP training data.
> 
> NLP is not `100%` accurate. Good models are pretty accurate, but we can't
> guarantee that the model will only identify colors as such. The model
> could decide that 'sustainable' is a color. Searching for a product in
> color 'sustainable' will not result in any matches. However, searching
> for a product with a generic attribute of 'maroon' or 'champagne' would work.
> 
> So, from an NLP/NER perspective, we treat colors like all other generic
> attributes. We then use rules to identify known colors that match our search
> filters. This gives us the best of both worlds. Search filters work as
> expected, but we still support long tail searches for wacky colors.
> 
> As a bonus, this also improves the overall accuracy of the model. Unlike
> most NLP applications, we have a limited amount of *context* available
> to us in the search query. Trying to identify too many attributes that
> are grammatically similar will reduce the overall model performance.

## Learned knowledge

Remember how I said that NLU/NLP is able to generalise. Try searching for
a cotton jacket:

```astro
$ python -m src.client 'cotton jacket'

```

Take a look at the NER prediction:

```astro
{
  "ner_prediction": {
    "text": "cotton jacket",
    "product": "jacket",
    "attrs": [
      "cotton"
    ]
  }
}

```

The NLP model has never seen the word `cotton` before, yet it's able to
correctly identify it as a product attribute. That's the power of NLP at work.

## Summary

Elasticsearch is a great tool, but full text search only gets us so far.
For long tail searches, TF-IDF can actually work against us, selecting results
that aren't relevant. Natural Language Understanding allows us to really
understand what the user is asking for. Given a search phrase, we can identify
specific product types, prices colours and much more. Like all forms of
artificial intelligence, NLP is able to generalise. A good NLP model can
identify new products, colors and other attributes without any code changes.

We just scratched the surface here, but hopefully you have a taste of NLP
and how it compliments full-text search. To learn more about NLP I highly
recommend the spaCy docs. Why not try building your own NER model and pipeline?
For more complex queries you'll want to take things a step further by implementing
Part of Speech tagging and Dependency Parsing. This allows us to understand
the relationship between words and is a nice compliment to named entity recognition.
