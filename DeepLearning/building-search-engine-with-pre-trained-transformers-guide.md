---
title: Building a Search Engine With Pre-Trained Transformers: A Step By Step Guide
link: https://neptune.ai/blog/building-search-engine-with-pre-trained-transformers-guide
author: Aravind CR
---

We all use search engines. We search for information about the best item
to purchase, a nice place to hangout, or to answer our questions about anything
we want to know. 

We also rely heavily on search to check emails, documents and financial
transactions. A lot of these search interactions happen through text or
speech converted to voice input. This means a lot of language processing
happens on search engines, so NLP plays a pretty important role in modern
search engines

Let’s take a quick look into what happens when we search. When you search
using a query, the search engine collects a ranked list of documents that
matches the query. For this to happen, an “**index**” of documents and vocabulary
used in them should be constructed first, and then used to search and rank
results. One of the popular forms of indexing textual data and ranking search
results for search is [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

Recent development in deep learning models for NLP can be used for this.
For example, Google recently started ranking search results and showing
snippets using the [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
model. They claim that this has improved the quality and relevance of search results.

![](https://i0.wp.com/neptune.ai/wp-content/uploads/Fine-tuning-BERT.png)
> Fig:  Fine-tuning BERT on Different Tasks

There are 2 types of search engines:

- **Generic search engines**, such as Google and Bing, that crawl the web
and aim to cover as much as possible by constantly looking for new webpages.
- **Enterprise search engines**, where our search space is restricted to
a smaller set of already existing documents within an organization.

The second form of search is the most common use case you will encounter
at any workplace. It’s clear when you look at the diagram below.

![](https://i0.wp.com/neptune.ai/wp-content/uploads/Search-engine-diagram.png)

You can use state-of-the-art sentence embeddings with transformers, and
use them in downstream tasks for semantic textual similarity.

In this article, we’ll explore how to build a vector-based search engine.

## Why would you need a vector-based search engine?

Keyword-based search engine struggle with:

- Complex queries or words that have dual meaning.
- Long search queries.
- Users not familiar with important keywords to retrieve best results.

Vector-based (also known as semantic search) search solves these problems
by finding a numerical representation of text queries using SOTA language
models. Then it indexes them in high dimensional vector space, and measures
how similar a query vector is to the indexed documents.

Lets see what the pre-trained models have to offer:

- They produce **high quality embeddings**, as they were trained on large
amounts of text data.
- They **don’t force you to create a custom tokenizer**, as transformers
come with their own methods.
- They’re **really simple and handy** to fine tune the model to your downstream task.

These models produce a fixed size vector for each token in the document.

Now, let’s see how we can use a pre-trained BERT model to build a feature
extractor for search engines.

### Step 1: Load the pre-trained model

```python repl
!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
!pip install bert-serving-server --no-deps
```

For this implementation, I’ll be using BERT uncased. There are other variations
of BERT available – bert-as-a-service uses BERT as a sentence encoder and
hosts it as a service via [ZeroMQ](https://zeromq.org/), letting you map
sentences into fixed length representations with just 2 lines of code.
This is useful if you want to avoid additional latency and potential modes
introduced by a client-server architecture.

### Step 2: Optimizing the inference graph

To modify the model graph, we need some low level Tensorflow programming.
Since we’re using bert-as-a-service, we can configure the inference graph
using a simple CLI interface. 

(The version of tensorflow used for this implementation was `tensorflow==1.15.2`)

```python
import os
import tensorflow as tf
import tensorflow.compat.v1 as tfc
 
 
sess = tfc.InteractiveSession()
 
from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser
 
 
# input dir
MODEL_DIR = '/content/uncased_L-12_H-768_A-12' #@param {type:"string"}
# output dir
GRAPH_DIR = '/content/graph/' #@param {type:"string"}
# output filename
GRAPH_OUT = 'extractor.pbtxt' #@param {type:"string"}
 
POOL_STRAT = 'REDUCE_MEAN' #@param ['REDUCE_MEAN', 'REDUCE_MAX', "NONE"]
POOL_LAYER = '-2' #@param {type:"string"}
SEQ_LEN = '256' #@param {type:"string"}
 
 
tf.io.gfile.mkdir(GRAPH_DIR)
 
 
carg = get_args_parser().parse_args(args=['-model_dir', MODEL_DIR,
                              '-graph_tmp_dir', GRAPH_DIR,
                              '-max_seq_len', str(SEQ_LEN),
                              '-pooling_layer', str(POOL_LAYER),
                              '-pooling_strategy', POOL_STRAT])
 
tmp_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)
 
tf.gfile.Rename(
   tmp_name,
   graph_fout,
   overwrite=True
)
print("\nSerialized graph to {}".format(graph_fout))
```

Take a look at a few parameters in the above snippet.

For each text sample, the BERT-base model encoding layer outputs a tensor
of shape [`sequence_len`, `encoder_dim`], with one vector per input token.
To get a fixed representation, we need to apply some sort of pooling.

`POOL_STRAT` parameter defines the pooling strategy applied to the encoder
layer number `POOL_LAYER`. The default value ‘`REDUCE_MEAN`’ averages the
vector for all tokens in the sequence. This particular strategy works best
for most sentence-level tasks, when the model is not fine-tuned. Another
option is `NONE`, in which case no pooling is applied.

`SEQ_LEN` has an impact on the maximum length of sequences processed by the
model. If you want to increase the model inference speed almost linearly,
you can give smaller values.

Running the above code snippet will put the model graph and weights into a
`GraphDef` object, which will be serialized to a `pbtxt` file at `GRAPH_OUT`.
The file will often be smaller than the pre-trained model, because the nodes
and the variables required for training will be removed.

### Step 3: Creating feature extractor

Let’s use the serialized graph to build a feature extractor using [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
API. We need to define 2 things: `input_fn` and `model_fn`.

`input_fn` gets data into the model. This includes executing the whole text
preprocessing pipeline and preparing a `feed_dict` for BERT.

Each text sample is converted into a `tf.Example` instance, with the necessary
features listed in the `INPUT_NAMES`. The `bert_tokenizer` object contains
the `WordPiece` vocabulary and performs text processing. After that, the
examples are regrouped by feature names in `feed_dict`.

```python
import logging
import numpy as np
 
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.keras.utils import Progbar
 
from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features
 
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []
```

```python
GRAPH_PATH = "/content/graph/extractor.pbtxt" #@param {type:"string"}
VOCAB_PATH = "/content/uncased_L-12_H-768_A-12/vocab.txt" #@param {type:"string"}
 
SEQ_LEN = 256 #@param {type:"integer"}
```

```python
INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)
 
def build_feed_dict(texts):
  
   text_features = list(convert_lst_to_features(
       texts, SEQ_LEN, SEQ_LEN,
       bert_tokenizer, log, False, False))
 
   target_shape = (len(texts), -1)
 
   feed_dict = {}
   for iname in INPUT_NAMES:
       features_i = np.array([getattr(f, iname) for f in text_features])
       features_i = features_i.reshape(target_shape).astype("int32")
       feed_dict[iname] = features_i
 
   return feed_dict
```

`tf.Estimators` have a feature which makes them rebuild and reinitialize
the whole computational graph at each call to the predict function. 

So, in order to avoid the overhead, we’ll **pass the generator to the predict
function**, and the generator will yield the features to the model in a
never ending loop.

```python
def build_input_fn(container):
  
   def gen():
       while True:
         try:
           yield build_feed_dict(container.get())
         except:
           yield build_feed_dict(container.get())
 
   def input_fn():
       return tf.data.Dataset.from_generator(
           gen,
           output_types={iname: tf.int32 for iname in INPUT_NAMES},
           output_shapes={iname: (None, None) for iname in INPUT_NAMES})
   return input_fn
 
class DataContainer:
 def __init__(self):
   self._texts = None
  def set(self, texts):
   if type(texts) is str:
     texts = [texts]
   self._texts = texts
  
 def get(self):
   return self._texts
```

The `model_fn` contains the specification of the model. In our case, it’s
loaded from the `pbtxt` file we saved in the previous step. The features
are mapped explicitly to the corresponding input nodes via `input_map`.

```python
def model_fn(features, mode):
   with tf.gfile.GFile(GRAPH_PATH, 'rb') as f:
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())
      
   output = tf.import_graph_def(graph_def,
                                input_map={k + ':0': features[k] for k in INPUT_NAMES},
                                return_elements=['final_encodes:0'])
 
   return EstimatorSpec(mode=mode, predictions={'output': output[0]})
  
estimator = Estimator(model_fn=model_fn)
```

Now that we have things in place, we need to perform inference.

```python
def batch(iterable, n=1):
   l = len(iterable)
   for ndx in range(0, l, n):
       yield iterable[ndx:min(ndx + n, l)]
 
def build_vectorizer(_estimator, _input_fn_builder, batch_size=128):
 container = DataContainer()
 predict_fn = _estimator.predict(_input_fn_builder(container), yield_single_examples=False)
  def vectorize(text, verbose=False):
   x = []
   bar = Progbar(len(text))
   for text_batch in batch(text, batch_size):
     container.set(text_batch)
     x.append(next(predict_fn)['output'])
     if verbose:
       bar.add(len(text_batch))
    
   r = np.vstack(x)
   return r
  return vectorize
bert_vectorizer = build_vectorizer(estimator, build_input_fn)
```

```bash
bert_vectorizer(64*['sample text']).shape
o/p: (64, 768)
```

### Step 4: Exploring vector space with projector

Using the vectorizer, we will generate embeddings for articles from the
[Reuters-221578](https://paperswithcode.com/dataset/reuters-21578)
benchmark corpus.

To explore and visualize the embedding vector space in 3D, we will use a
dimensionality reduction technique called [T-SNE](https://distill.pub/2016/misread-tsne/).

First let’s get the article embeddings.

```python
from nltk.corpus import reuters
 
import nltk
nltk.download("reuters")
nltk.download("punkt")
 
max_samples = 256
categories = ['wheat', 'tea', 'strategic-metal',
             'housing', 'money-supply', 'fuel']
 
S, X, Y = [], [], []
 
for category in categories:
 print(category)
  sents = reuters.sents(categories=category)
 sents = [' '.join(sent) for sent in sents][:max_samples]
 X.append(bert_vectorizer(sents, verbose=True))
 Y += [category] * len(sents)
 S += sents
 X = np.vstack(X)
X.shape
```

After running the above code, if you face any issues in collab that say:
“**Resource reuters not found. Please use the NLTK downloader to obtain
the resource.**”

...then run the following command, where the relative path after -d will
give the location where the file will be unzipped:

```python repl
!unzip /root/nltk_data/corpora/reuters.zip -d /root/nltk_data/corpora
```

The interactive visualizations of the generated embeddings are available
on the [Embedding Projector](https://projector.tensorflow.org/).

From the link, you can run `t-SNE` yourself, or load a checkpoint using the
bookmark in the lower right corner (loading works on Chrome).

To reproduce the input files used for this visualization, run the code
snippet below. Then download the files to your machine and upload to Projector.

```python
with open("embeddings.tsv", "w") as fo:
 for x in X.astype('float'):
   line = "\t".join([str(v) for v in x])
   fo.write(line+'\n')
 
with open('metadata.tsv', 'w') as fo:
 fo.write("Label\tSentence\n")
 for y, s in zip(Y, S):
   fo.write("{}\t{}\n".format(y, s))
```

Here’s what I captured using the Projector.

```python repl
from IPython.display import HTML
 
HTML("""
<video width="900" height="632" controls>
 <source src="https://storage.googleapis.com/bert_resourses/reuters_tsne_hd.mp4" type="video/mp4">
</video>
""")
```

```
[video src="https://neptune.ai/wp-content/uploads/reuters_tsne_hd.mp4"]
```

Building a supervised model with the generated features is straightforward:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
Xtr, Xts, Ytr, Yts = train_test_split(X, Y, random_state=34)
 
mlp = LogisticRegression()
mlp.fit(Xtr, Ytr)
 
print(classification_report(Yts, mlp.predict(Xts)))
```

|| Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Fuel | 0.75 | 0.81 | 0.78 | 26 |
| Housing | 0.73 | 0.75 | 0.74 | 32 |
| Money-supply | 0.84 | 0.88 | 0.86 | 75 |
| Strategic-metal | 0.88 | 0.90 | 0.89 | 48 |
| Tea | 0.85 | 0.80 | 0.82 | 44 |
| Wheat | 0.94 | 0.86 | 0.90 | 59 |
| Accuracy | - | - | 0.85 | 284 |
| Macro avg | 0.83 | 0.83 | 0.83 | 284 |
| Weighted avg | 0.85 | 0.85 | 0.85 | 284 |

### Step 5: Building a search engine

Let’s say we have a knowledge base of 50,000 text samples, and we need to
quickly answer queries based on this data. How can we retrieve the result
most similar to a query from a text database? One of the answers can be
nearest neighbour search.

The search problem we’re solving here can be defined as follows:

Given a set of points **S** in vector space **M** and a query point **Q ∈ M**,
find the closest point **S** to **Q**. There are multiple ways to define
‘**closest**’ in the vector space – we’ll use [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,being%20called%20the%20Pythagorean%20distance.).

To build a search engine for text, we’ll follow these steps:

- Vectorize all samples from the knowledge base – that gives **S**.
- Vectorize the query – that gives **Q**.
- Compute euclidean distance **D** between **Q** and **S**.
- Sort **D** in ascending order- providing indices of most similar samples.
- Retrieve labels for said samples from the knowledge base.

We can create the placeholder for **Q** and **S**:

```python
graph = tf.Graph()
 
sess = tf.InteractiveSession(graph=graph)
 
dim = X.shape[1]
 
Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])
```

Define euclidean distance computation:

```python
squared_distance = tf.reduce_sum(tf.pow(Q - S, 2), reduction_indices=1)
distance = tf.sqrt(squared_distance)
```

Get the most similar indices:

```python
top_k = 10
 
top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)
top_dists = tf.negative(top_neg_dists)
```

```python
from sklearn.metrics.pairwise import euclidean_distances
 
top_indices.eval({Q:X[0], S:X})
 
np.argsort(euclidean_distances(X[:1], X)[0])[:10]
```

### Step 6: Accelerating search with math

In tensorflow this can be done as follows:

```python
Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])
 
Qr = tf.reshape(Q, (1, -1))
 
PP = tf.keras.backend.batch_dot(S, S, axes=1)
QQ = tf.matmul(Qr, tf.transpose(Qr))
PQ = tf.matmul(S, tf.transpose(Qr))
 
distance = PP - 2 * PQ + QQ
distance = tf.sqrt(tf.reshape(distance, (-1,)))
 
top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)
```

In the above formula **PP** and **QQ** are actually squared
[L2 norms](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm)
of the respective vectors. If both vectors are L2 normalized, then:

**PP = QQ = 1**

**Doing L2 normalization discards the information about the vector magnitude,
which in many cases you don’t want to do.**

Instead, we may notice that as long as the knowledge base stays the same
– PP – its squared vector norm also stays the same. So, instead of recomputing
it every time, we can just do it once and then use the precomputed result,
further accelerating the distance computation.

Let’s bring this all together.

```python
class L2Retriever:
 def __init__(self, dim, top_k=3, use_norm=False, use_gpu=True):
   self.dim = dim
   self.top_k = top_k
   self.use_norm = use_norm
   config = tf.ConfigProto(
       device_count = {'GPU': (1 if use_gpu else 0)}
   )
   self.session = tf.Session(config=config)
 
   self.norm = None
   self.query = tf.placeholder("float", [self.dim])
   self.kbase = tf.placeholder("float", [None, self.dim])
 
   self.build_graph()
 
 def build_graph():
   if self.use_norm:
     self.norm = tf.placeholder("float", [None, 1])
 
   distance = dot_l2_distances(self.kbase, self.query, self.norm)
   top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=self.top_k)
   top_dists = tf.negative(top_neg_dists)
 
   self.top_distances = top_dists
   self.top_indices = top_indices
 
 def predict(self, kbase, query, norm=None):
   query = np.squeeze(query)
   feed_dict = {self.query: query, self.kbase: kbase}
   if self.use_norm:
     feed_dict[self.norm] = norm
 
   I, D = self.session.run([self.top_indices, self.top_distances],
                           feed_dict=feed_dict)
   return I, D
 
def dot_l2_distances(kbase, query, norm=None):
 query = tf.reshape(query, (1, 1))
 
 if norm is None:
   XX = tf.keras.backend.batch_dot(kbase, kbase, axes=1)
 else:
   XX = norm
 YY = tf.matmul(query, tf.transpose(query))
 XY = tf.matmul(kbase, tf.transpose(query))
 
 distance = XX - 2 * XY + YY
 distance = tf.sqrt(tf.reshape(distance, (-1, 1)))
 
 return distance
```

We can use this implementation with any vectorizer model, not just BERT.
It’s quite effective at the nearest neighbour retrieval, able to process
dozens of requests per second on a 2-core colab CPU.

There are some extra aspects you need to consider when building machine
learning applications:

- How do you ensure the scalability of your solution?
- Pick the right framework/languages. 
- Use the right processors. 
- Collect and warehouse data. 
- Input pipeline.
- Model training.
- Distributed systems.
- Other optimizations.
- Resource Utilization and monitoring.
- Deploy.
- How do you train, test and deploy your model to production?
- Create a notebook instance that you can use to download and process
your data.
- Prepare the data/preprocess it that you need to train your ML model
and then upload the data (ex: Amazon S3).
- Use your training dataset to train your machine learning model. 
- Deploy the model to an endpoint, reformat and load the csv data, then
run the model to create predictions.
- Evaluate the performance and accuracy of the ML model.

### Side note – make ML easier with experiment tracking

One tool can take care of all your [experiment tracking](https://neptune.ai/experiment-tracking)
and collaboration needs –  [neptune.ai](https://neptune.ai/)

Neptune records your entire experimentation process – exploratory notebooks,
model training runs, code, hyperparameters, metrics, data versions, results,
exploration visualizations, and more. 

It’s the metadata store for MLOps, built for research and production teams
that run a lot of experiments. Focus on ML, and leave metadata management
to Neptune. To get started with Neptune, visit their extensive
[guide](https://docs.neptune.ai/getting-started/hello-world).

**An ML metadata store like Neptune is an essential part of the MLOps stack**.
It takes care of metadata management when you’re building your models.

It logs, stores, displays, organizes, compares and queries all metadata
generated during the ML model lifecycle. 

You can use an ML metastore to track, organize, and compare everything you
care about in your ML experiments. 

![](https://i0.wp.com/neptune.ai/wp-content/uploads/Product_logging-metadata.gif)

Neptune integrates with all of your favorite frameworks and tools – one
of the most popular integrations is [Tensorflow/Keras](https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras),
done directly via
[TensorBoard](https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/tensorboard).

## Conclusion

The main area of exploration for search with BERT is similarity. Similarity
for documents, for recommendations, and similarity between queries and
documents for returning and ranking search results. 

If you can use similarity to solve this problem with highly accurate results,
then you have a pretty great search for your product or application. 

I hope you learned something new here. Thanks for reading. Keep learning.
