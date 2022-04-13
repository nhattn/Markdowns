---
title: Part of speech tagging - Sequence labelling in Python
link: https://dev.to/fferegrino/sequence-labelling-in-python-part-1-4noa
author: Antonio Feregrino
---

I was looking for a cool project to practice sequence labelling with Python
so... there is this Mexican website called **VuelaX**, in it, flight offers
are shown. Most of the offers follow a simple pattern: Destination - Origin
- Price - Extras, while extracting this may seem easy for a regular expression,
it is not as there are many patterns. It would be tough for us to cover them all.

> I know it is not ideal to work in a foreign language, but bear with me,
as the same techniques could be applied in your language of choice.

The idea is to create a tagger that will be able to extract this information.
However, one first tag is to identify the information that we want to extract.
Following the pattern described above:

- **o**: Origin
- **d**: Destination
- **s**: Separator token
- **p**: Price
- **f**: Flag
- **n**: Irrelevant token

<table>
    <thead>
        <tr>
            <th>Text</th>
            <th>d</th>
            <th>o</th>
            <th>p</th>
            <th>n</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>¡CUN a Holanda $8,885! Sin escala EE.UU</td>
            <td>CUN</td>
            <td>Holanda</td>
            <td>8,885</td>
            <td>Sin escala EE.UU</td>
        </tr>
        <tr>
            <td>¡CDMX a Noruega $10,061! (Y agrega 9 noches de hotel por
$7,890!)</td>
            <td>CDMX</td>
            <td>Noruega</td>
            <td>10,061</td>
            <td>Y agrega 9 noches de hotel por $7,890!</td>
        </tr>
        <tr>
            <td>¡Todo México a Pisa, Toscana Italia $12,915! Sin escala
EE.UU (Y por $3,975 agrega 13 noches hotel)</td>
            <td>México</td>
            <td>Pisa, Toscana Italia</td>
            <td>12,915</td>
            <td>Sin escala EE.UU (Y por $3,975 agrega 13 noches hotel)</td>
        </tr>
    </tbody>
<table>

## CRFs in Python

If you are familiar with data science, you know this is known as a sequence
labelling problem. While there are various ways to approach it, in this post,
I will show you one that uses a statistical model known as Conditional Random
Fields. Having said that, I will not delve too much into details, so if you
want to learn more about CRFs you are on your own; I will show you a practical
way to use it with a Python implementation.

## Getting some data

To start, I scraped the offer titles data from the page mentioned above.
I will not detail how I did it since it is pretty straightforward to find
a tutorial on web scraping on the web. If you don't feel like spending some
time scraping a website, I collected some data in a CSV file that you can
[access now here](https://github.com/fferegrino/vuelax-crf/blob/master/data/vuelos.csv).

This tutorial will be divided into other 4 parts:

- Part-Of-Speech tagging (and getting some ground truth)
- Other feature extraction
- Conditional Random Fields with python-crfsuite
- Putting everything together

Our algorithm needs more than the tokens themselves to be more reliable;
We can add part of speech as a feature.

To perform the Part-Of-Speech tagging, we'll be using the [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml);
this tagger (or at least the interface to it) is available to use through
Python's NLTK library; however, we need to download some models from the
[Stanford's download page](https://nlp.stanford.edu/software/tagger.shtml#Download).
In our case, since we are working with spanish, we should download the full
model under the _"2017-06-09 new Spanish and French UD models"_ subtitle.

Once downloaded, it is necessary to unzip it and keep track of where the
files end up being. You could execute:

```bash
make models/stanford
```

To get the necessary files inside a folder called `stanford-models`.
**Be aware that you will need to have Java installed for the tagger to work!**

## Code

Let us start with some imports and loading our dataset:

```python
import json
import pandas as pd

# Load dataset:
vuelos = pd.read_csv('data/vuelos.csv', index_col=0)
with pd.option_context('max_colwidth', 800):
    print(vuelos.loc[:40:5][['label']])
```

Some of the results:

```bash
0                                           ¡CUN a Ámsterdam $8,960! Sin escala en EE.UU
5              ¡GDL a Los Ángeles $3,055! Directos (Agrega 3 noches de hotel por $3,350)
10                      ¡CUN a Puerto Rico $3,296! (Agrega 3 noches de hotel por $2,778)
15    ¡LA a Seúl, regresa desde Tokio 🇯🇵 $8,607! (Por $3,147 agrega 11 noches de hostal)
20                           ¡CDMX a Chile $8,938! (Agrega 9 noches de hotel por $5,933)
25                                               ¡CUN a Holanda $8,885! Sin escala EE.UU
30                              ¡Todo México a París, regresa desde Amsterdam – $11,770!
35  ¡CDMX a Vietnam $10,244! Sin escala en EE.UU (Agrega 15 noches de hostal por $2,082)
40                     ¡CDMX a Europa en Semana Santa $14,984! (París + Ibiza + Venecia)
```

To interface with the Stanford tagger, we could use the `StanforPOSTagger`
inside the `nltk.tag.stanford` module, then we create an object passing in
both our language-specific model as well as the tagger `.jar` we previously
downloaded from Stanford's website.

Then, as a quick test, we tag a spanish sentence to see what is it that we
get back from the tagger.

```python
from nltk.tag.stanford import StanfordPOSTagger

spanish_postagger = StanfordPOSTagger('stanford-models/spanish.tagger', 
                                      'stanford-models/stanford-postagger.jar')

phrase = 'Amo el canto del cenzontle, pájaro de cuatrocientas voces.'
tags = spanish_postagger.tag(phrase.split()) 
print(tags)
```

The results:

```bash
[('Amo', 'vmip000'), ('el', 'da0000'), ('canto', 'nc0s000'), 
('del', 'sp000'), ('cenzontle,', 'dn0000'), ('pájaro', 'nc0s000'), 
('de', 'sp000'), ('cuatrocientas', 'pt000000'), ('voces.', 'np00000')]
```

The first thing to note is the fact that the tagger takes in lists of strings,
not a full sentence, that is why we need to split our sentence before passing
it in. A second thing to note is that we get back of tuples; where the first
element of each tuple is the token and the second is the POS tag assigned
to said token. The POS tags are [explained here](https://nlp.stanford.edu/software/spanish-faq.html),
and I have made a dictionary for easy lookups.

We can inspect the tokens a bit more:

```python
with open("aux/spanish-tags.json", "r") as r:
    spanish_tags = json.load(r)

for token, tag in tags[:10]:
    print(f"{token:15} -> {spanish_tags[tag]['description']}")
```

And the results:

```bash
Amo             -> Verb (main, indicative, present)
el              -> Article (definite)
canto           -> Common noun (singular)
del             -> Preposition
cenzontle,      -> Numeral
pájaro          -> Common noun (singular)
de              -> Preposition
cuatrocientas   -> Interrogative pronoun
voces.          -> Proper noun
```

## Specific tokenisation

As you may imagine, using `split` to tokenise our text is not the best idea;
it is almost certainly better to create our function, taking into consideration
the kind of text that we are going to process. The function above uses the
`TweetTokenizer` and considers flag emojis. As a final touch, it also returns
the position of each one of the returned tokens:

```python
from nltk.tokenize import TweetTokenizer

TWEET_TOKENIZER = TweetTokenizer()

# This function exists in vuelax.tokenisation in this same repository
def index_emoji_tokenize(string, return_flags=False):
    flag = ''
    ix = 0
    tokens, positions = [], []
    for t in TWEET_TOKENIZER.tokenize(string):
        ix = string.find(t, ix)
        if len(t) == 1 and ord(t) >= 127462:  # this is the code for 🇦
            if not return_flags: continue
            if flag:
                tokens.append(flag + t)
                positions.append(ix - 1)
                flag = ''
            else:
                flag = t
        else:
            tokens.append(t)
            positions.append(ix)
        ix = +1
    return tokens, positions




label = vuelos.iloc[75]['label']
print(label)
print()
tokens, positions = index_emoji_tokenize(label, return_flags=True)
print(tokens)
print(positions)
```

And these are the results:

```bash
¡LA a Bangkok 🇹🇭$8,442! (Por $2,170 agrega 6 noches de Hotel)

['¡', 'LA', 'a', 'Bangkok', '🇹🇭', '$', '8,442', '!', '(', 'Por', '$',
'2,170', 'agrega', '6', 'noches', 'de', 'Hotel', ')']
[0, 1, 4, 6, 14, 16, 17, 22, 24, 25, 16, 30, 36, 43, 45, 52, 55, 60]
```

## Obtaining our ground truth for our problem

**We do not need POS Tagging to generate a tagged dataset!.**

Now, since this is a supervised algorithm, we need to get some labels from
"expert" users. These labels will be used to train the algorithm to produce
predictions. The task for the users will be simple: assign one of the following
letters to each token: `{ o, d, s, p, f, n }`. While there are [online tools](https://doccano.herokuapp.com/)
to perform this task, I decided to go more old school with a simple CSV
file with a format more or less like this:

<table>
    <thead>
        <tr>
            <th>Offer</th>
            <th>Id</th>
            <th>Token</th>
            <th>Position</th>
            <th>POS	Label</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
            <td>¡</td>
            <td>0</td>
            <td>faa</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>CUN</td>
            <td>1</td>
            <td>np00000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>a</td>
            <td>5</td>
            <td>sp000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>Ámsterdam</td>
            <td>7</td>
            <td>np00000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>$</td>
            <td>17</td>
            <td>zm</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>8,960</td>
            <td>18</td>
            <td>dn0000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>!</td>
            <td>23</td>
            <td>fat</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>Sin</td>
            <td>25</td>
            <td>sp000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>escala</td>
            <td>29</td>
            <td>nc0s000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>en</td>
            <td>36</td>
            <td>sp000</td>
            <td>[USER LABEL]</td>
        </tr>
        <tr>
            <td>0</td>
            <td>EE.UU</td>
            <td>39</td>
            <td>np00000</td>
            <td>[USER LABEL]</td>
        </tr>
    </tbody>
</table>

Where the values of the column marked with `[USER LABEL]` should be defined
by the expert users who will help us in labelling our data.

```python
from tqdm.notebook import trange, tqdm
import csv

path_for_data_to_label = "data/to_label.csv"

with open(path_for_data_to_label, "w") as w:
    writer = csv.writer(w)
    writer.writerow(['offer_id', 'token', 'position', 'pos_tag', 'label'])

    for offer_id, row in tqdm(vuelos.iterrows(), total=len(vuelos)):
        tokens, positions = index_emoji_tokenize(row["label"], return_flags=True)
        tags = spanish_postagger.tag(tokens)
        for  token, position, (_, pos_tag) in zip(tokens, positions, tags):
            writer.writerow([
                offer_id,
                token,
                position,
                pos_tag,
                None
            ])
```

The file that needs to be labelled is located at `data/to_label.csv`.

**Can we make this easy?** I have gone through the "pains" of labelling some
data myself; the labels are stored in the file `data/to_label-done.csv`.

## Extracting more features - Sequence labelling in Python

While having the POS tags is good and valuable information, it may not be
enough to get valuable predictions for our task. However, we can provide
our algorithm with more information; such as the length of the token, the
length of the sentence, the position within the sentence, whether the token
is a number or all uppercase...

Some imports:

```python
from vuelax.tokenisation import index_emoji_tokenize
import pandas as pd
import csv
```

Starting from our already labelled dataset (remember I have a file called
`data/to_label.csv`). The following are just a few helper functions to read
and augment our dataset:

```python
labelled_data = pd.read_csv("data/to_label-done.csv")
labelled_data.head()
```

We need to create a helper function to read all the labelled offers from
the previously created `labelled_data` dataframe:

```python
def read_whole_offers(dataset):
    current_offer = 0
    rows = []
    for _, row in dataset.iterrows():
        if row['offer_id'] != current_offer:
            yield rows
            current_offer = row['offer_id']
            rows = []
        rows.append(list(row.values))
    yield rows

offers = read_whole_offers(labelled_data)
offer_ids, tokens, positions, pos_tags, labels = zip(*next(offers))
print(offer_ids)
print(tokens)
print(positions)
print(pos_tags)
print(labels)
```

And here is the output from the first flight offer:

```bash
(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
('¡', 'CUN', 'a', 'Ámsterdam', '$', '8,960', '!', 'Sin', 'escala', 'en', 'EE.UU')
(0, 1, 5, 7, 17, 18, 23, 25, 29, 36, 39)
('faa', 'np00000', 'sp000', 'np00000', 'zm', 'dn0000', 'fat', 'sp000',
'nc0s000', 'sp000', 'np00000')
('n', 'o', 's', 'd', 'n', 'p', 'n', 'n', 'n', 'n', 'n')
```

## Building our training set

The features I decided to augment the data with are the following:

- Lengths of each token
- Length of the whole offer (counted in tokens)
- The POS tag of the token to the left
- The POS tag of the token to the right
- Whether the token is uppercase or not

And this is the respective function to generate said features:

```python
def generate_more_features(tokens, pos_tags):
    lengths =  [len(l) for l in tokens]
    n_tokens =  [len(tokens) for l in tokens]
    augmented = ['<p>'] + list(pos_tags) + ['</p>']
    uppercase = [all([l.isupper() for l in token]) for token in tokens]
    return lengths, n_tokens, augmented[:len(tokens)], augmented[2:], uppercase

generate_more_features(tokens, pos_tags)
```

As an example output:

```bash
([1, 3, 1, 9, 1, 5, 1, 3, 6, 2, 5],
 [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
 ['<p>',
  'faa',
  'np00000',
  'sp000',
...
  'sp000',
  'nc0s000',
  'sp000'],
 ['np00000',
  'sp000',
...
  'nc0s000',
  'sp000',
  'np00000',
  '</p>'],
 [False, True, False, False, False, False, False, False, False, False, False])
```

Finally, we need to apply the function to all the offers in our dataset,
and save these to a file, to keep them handy for the next task, you can
read more about it here: Training a CRF in Python.

## Conditional Random Fields in Python - Sequence labelling

Once we have our dataset with all the features we want to include, as well
as all the labels for our sequences; we can move on to the actual training
of our algorithm. For this task, we'll be using the `python-crfsuite` package,
so pip-install it (or use your favourite package manager to get it):

```bash
pip install python-crfsuite
```

Once installed, let's load our dataset:

```python
import pandas as pd
import string

features_labels = pd.read_csv("data/features-labels.csv")
features_labels = features_labels[~features_labels['label'].isna()]
features_labels.head()
```

We also need some helper functions that work on strings; you'll see how
they are used a bit further down:

```python
punctuation = set(string.punctuation)

def is_punctuation(token):
    return token in punctuation

def is_numeric(token):
    try:
        float(token.replace(",", ""))
        return True
    except:
        return False
```

## Input to pycrfsuite

The inputs to the algorithm must follow a particular format, where each
token has its features represented by key-value pairs, each token may also
have different features based on different factors, like its position.
The following function takes in a dataframe and returns the corresponding
features that can be consumed by the training method of our algorithm:

```python
def featurise(sentence_frame, current_idx):
    current_token = sentence_frame.iloc[current_idx]
    token = current_token['token']
    position = current_token['position']
    token_count = current_token['token_count']
    pos = current_token['pos_tag']

    # Shared features across tokens
    features = {
            'bias': True,
            'word.lower': token.lower(),
            'word.istitle': token.istitle(),
            'word.isdigit': is_numeric(token),
            'word.ispunct': is_punctuation(token),
            'word.position':position,
            'word.token_count': token_count,
            'postag': pos, 
    }

    if current_idx > 0: # The word is not the first one...
        prev_token = sentence_frame.iloc[current_idx-1]['token']
        prev_pos = sentence_frame.iloc[current_idx-1]['pos_tag']
        features.update({
            '-1:word.lower': prev_token.lower(),
            '-1:word.istitle':prev_token.istitle(),
            '-1:word.isdigit': is_numeric(prev_token),
            '-1:word.ispunct': is_punctuation(prev_token),
            '-1:postag':prev_pos 
        })
    else:
        features['BOS'] = True

    if current_idx < len(sentence_frame) - 1: # The word is not the last one...
        next_token = sentence_frame.iloc[current_idx+1]['token']
        next_tag = sentence_frame.iloc[current_idx+1]['pos_tag']
        features.update({
            '+1:word.lower': next_token.lower(),
            '+1:word.istitle': next_token.istitle(),
            '+1:word.isdigit': is_numeric(next_token),
            '+1:word.ispunct': is_punctuation(next_token),
            '+1:postag': next_tag 
        })
    else:
        features['EOS'] = True

    return features

featurise(offer_0, 1)
```

By featurising the first token of the first offer we get the following:

```bash
{'bias': True,
 'word.lower': 'cun',
 'word.istitle': False,
 'word.isdigit': False,
 'word.ispunct': False,
 'word.position': 1,
 'word.token_count': 11,
 'postag': 'np00000',
 '-1:word.lower': '¡',
 '-1:word.istitle': False,
 '-1:word.isdigit': False,
 '-1:word.ispunct': False,
 '-1:postag': 'faa',
 '+1:word.lower': 'a',
 '+1:word.istitle': False,
 '+1:word.isdigit': False,
 '+1:word.ispunct': False,
 '+1:postag': 'sp000'}
```

As you can see, the features are represented in a dictionary, where the
keys can be any dictionary key, but I chose to name them like that to make
it easy to find where each particular value comes from.

Again, we need some functions to build the sentences back from the tokens:

```python
def featurize_sentence(sentence_frame):
    labels = list(sentence_frame['label'].values)
    features = [featurize(sentence_frame, i) for i in range(len(sentence_frame))]

    return features, labels

def rollup(dataset):
    sequences = []
    labels = []
    offers = dataset.groupby('offer_id')
    for name, group in offers:
        sqs, lbls = featurize_sentence(group)
        sequences.append(sqs)
        labels.append(lbls)

    return sequences, labels

all_sequences, all_labels = rollup(features_labels)
```

We now have in `all_sequences` and `all_labels` our features and their corresponding
labels ready for training.

## Training

Pretty much like in any other supervised problem, we need to split our training
dataset into two (preferably three) sets of data; we can use `train_test_split`
for this:

```python
from sklearn.model_selection import train_test_split

train_docs, test_docs, train_labels, test_labels = train_test_split(all_sequences, all_labels)

len(train_docs), len(test_docs)
```

## Creating a CRF

Though one can use a `sklearn-like` interface to create, train and infer
with `python-crfsuite`, I've decided to use the original package and do
all "by hand".

The first step is to create an object of the class `Trainer`; then we can
set some parameters for the training phase, feel free to play with these,
as they may improve the quality of the tagger. Finally, we need to pass in
our training data into the algorithm, and we do that with the append method:

```python
import pycrfsuite

trainer = pycrfsuite.Trainer(verbose=False)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 200, 

    'feature.possible_transitions': True
})


# We are feeding our training set to the algorithm here.
for xseq, yseq in zip(train_docs, train_labels):
    trainer.append(xseq, yseq)
```

Finally, we call the method train, that will, at the same time, save the
model to a file that we can then use to perform inferences in new sentences.

```python
trainer.train('model/vuelax-bad.crfsuite')
```

## Labelling "unseen" sequences

To perform sequence labelling on instances that our algorithm did not see
during training it is necessary to use an object of the `Tagger` class,
and then load our saved model into it by using the `open` method.

```python
crf_tagger = pycrfsuite.Tagger()
crf_tagger.open('model/vuelax-bad.crfsuite')
```

Remember that each one of the sentences needs to be processed and put in
the format required for the tagger to work that means, have the same features
we used for training. We already have this in our `test_docs`, and we can
use them directly:

```python
predicted_tags = crf_tagger.tag(test_docs[2])
print("Predicted: ",predicted_tags)
print("Correct  : ",test_labels[2])
```

The result may show that in this specific instance the tagger made no errors:

```bash
Predicted:  ['n', 'o', 's', 'd', 'd', 'n', 'p', 'n']
Correct  :  ['n', 'o', 's', 'd', 'd', 'n', 'p', 'n']
```

## Evaluating the tagger

Seeing our algorithm perform very well in a single example may not inform
us that well, so let's look at the bigger picture and while there may be
better ways to evaluate the performance of the tagger, we'll use the traditional
tools of a classification problem:

```python
from sklearn.metrics import classification_report

all_true, all_pred = [], []

for i in range(len(test_docs)):
    all_true.extend(test_labels[i])
    all_pred.extend(crf_tagger.tag(test_docs[i]))

print(classification_report(all_true, all_pred))
```

Should give you a result similar to...

```bash
              precision    recall  f1-score   support

           d       0.96      0.99      0.97        98
           f       1.00      1.00      1.00        10
           n       1.00      1.00      1.00       831
           o       1.00      1.00      1.00        80
           p       1.00      1.00      1.00        60
           s       1.00      1.00      1.00        60

    accuracy                           1.00      1139
   macro avg       0.99      1.00      1.00      1139
weighted avg       1.00      1.00      1.00      1139
```

Our algorithm performs very, very well.

It may seem like we are done here, but we still need to put everything
together, in an attempt to make it easier for us to tag new offers outside
of our training and testing sets, we'll do that in my next post putting
everything together.

## Putting everything together - Sequence labelling in Python

What good is our system if we can not use it to predict the labels of new
sentences. Before that, though, we need to make sure to set up a complete
pipeline to go from having a new offer as displayed in the VuelaX site to
have a fully labelled offer on our python scripts.

The idea is to borrow functions from all other previous posts; these functions
were replicated somewhere inside the `vuelax` packages and are imported to
make them less messy to work with.

### We've got a new offer!

Imagine getting a new offer that looks like this:

> ¡Sin pasar EE.UU! 🇪🇬¡Todo México a El Cairo, Egipto $13,677!

If your spanish is not on point, I'll translate this for you:

> Without stops in the USA! 🇪🇬 any airport in México to Cairo, Egypt $13,677!

```python
offer_text = "¡Sin pasar EE.UU! 🇪🇬¡Todo México a El Cairo, Egipto $13,677!"
```

### Steps:

**Tokenise**: the first step was to tokenise it, by using our `index_emoji_tokenize`
function

```python
from vuelax.tokenisation import index_emoji_tokenize

tokens, positions = index_emoji_tokenize(offer_text)

print(tokens)
```

**POS Tagging**: the next thing in line is to obtain the POS tags corresponding
to each one of the tokens. We can do this by using the `StanfordPOSTagger`:

```python
from nltk.tag.stanford import StanfordPOSTagger

spanish_postagger = StanfordPOSTagger('stanford-models/spanish.tagger', 
                                      'stanford-models/stanford-postagger.jar')

_, pos_tags = zip(*spanish_postagger.tag(tokens))

print(pos_tags)
```

**Prepare for the CRF**: This step involves adding more features and preparing
the data to be consumed by the CRF package. All the required methods exist
in `vuelax.feature_selection`

```python
from vuelax.feature_selection import featurise_token

features = featurize_sentence(tokens, positions, pos_tags)

print(features[0])
```

**Sequence labelling with pycrfsuite**: And the final step is to load our
trained model and tag our sequence:

```python
import pycrfsuite

crf_tagger = pycrfsuite.Tagger()
crf_tagger.open('model/vuelax-bad.crfsuite')

assigned_tags = crf_tagger.tag(features)

for assigned_tag, token in zip(assigned_tags, tokens):
    print(f"{assigned_tag} - {token}")
```

And the result:

```bash
n - ¡
n - Sin
n - pasar
n - EE.UU
n - !
n - ¡
o - Todo
o - México
s - a
d - El
d - Cairo
d - ,
d - Egipto
n - $
p - 13,677
n - !
```

By visual inspection we can confirm that the tags are correct: "Todo México"
is the origin (o), "El Cairo, Egipto" is the destination and "13,677" is
the price (p).

And that is it. This is the end of this series of posts on how to do sequence
labelling with Python.

## What else is there to do?

There are many ways this project could be improved, a few that come to mind:

- Improve the size/quality of the dataset by labelling more examples
- Improve the way the labelling happens, using a single spreadsheet does
not scale at all
- Integrate everything under a single processing pipeline
- "Productionify" the code, go beyond an experiment.
