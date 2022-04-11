---
title: 3 subword algorithms help to improve your NLP model performance
linke: https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
author: Edward Ma
---

![](https://miro.medium.com/max/1400/0*OzrGnF_f8bj3nqAf)
> Photo by Edward Ma on Unsplash

[Classic word representation](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a)
cannot handle unseen word or rare word well. [Character embeddings](https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10)
is one of the solution to overcome out-of-vocabulary (OOV). However, it may
too fine-grained any missing some important information. Subword is in between
word and character. It is not too fine-grained while able to handle unseen
word and rare word.

For example, we can split “subword” to “sub” and “word”. In other word we
use two vector (i.e. “sub” and “word”) to represent “subword”. You may argue
that it uses more resource to compute it but the reality is that we can
use less footprint by comparing to word representation.

This story will discuss about [SentencePiece: A simple and language independent
subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf)
(Kudo et al., 2018) and further discussing about different subword algorithms.
The following are will be covered:

- Byte Pair Encoding (BPE)
- WordPiece
- Unigram Language Model
- SentencePiece

## Byte Pair Encoding (BPE)

Sennrich et al. (2016) proposed to use Byte Pair Encoding (BPE) to build
subword dictionary. Radfor et al adopt BPE to construct subword vector to
build [GPT-2](https://towardsdatascience.com/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655)
in 2019.

## Algorithm

- Prepare a large enough training data (i.e. corpus)
- Define a desired subword vocabulary size
- Split word to sequence of characters and appending suffix “</w>” to end
of word with word frequency. So the basic unit is character in this stage.
For example, the frequency of “low” is 5, then we rephrase it to “l o w </w>”: 5
- Generating a new subword according to the high frequency occurrence.
- Repeating step 4 until reaching subword vocabulary size which is defined
in step 2 or the next highest frequency pair is 1.
![](https://miro.medium.com/max/970/1*_bpIUb6YZr6DOMLAeSU2WA.png)
> Algorithm of BPE (Sennrich et al., 2015)

## Example

Taking “low: 5”, “lower: 2”, “newest: 6” and “widest: 3” as an example,
the highest frequency subword pair is `e` and `s`. It is because we get 6
count from `newest` and 3 count from `widest`. Then new subword (`es`) is
formed and it will become a candidate in next iteration.

In the second iteration, the next high frequency subword pair is `es` (generated
from previous iteration )and `t`. It is because we get 6count from `newest`
and 3 count from `widest`.

Keep iterate until built a desire size of vocabulary size or the next highest
frequency pair is 1.

## WordPiece

WordPiece is another word segmentation algorithm and it is similar with BPE.
Schuster and Nakajima introduced WordPiece by solving Japanese and Korea
voice problem in 2012. Basically, WordPiece is similar with BPE and the
difference part is forming a new subword by likelihood but not the next
highest frequency pair.

## Algorithm

1. Prepare a large enough training data (i.e. corpus)
2. Define a desired subword vocabulary size
3. Split word to sequence of characters
4. Build a languages model based on step 3 data
5. Choose the new word unit out of all the possible ones that increases the
likelihood on the training data the most when added to the model.
6. Repeating step 5until reaching subword vocabulary size which is defined
in step 2 or the likelihood increase falls below a certain threshold.

## Unigram Language Model

Kudo. introduced unigram language model as another algorithm for subword
segmentation. One of the assumption is all subword occurrence are independently
and subword sequence is produced by the product of subword occurrence probabilities.
Both WordPiece and Unigram Language Model leverages languages model to build
subword vocabulary.

## Algorithm

1. Prepare a large enough training data (i.e. corpus)
2. Define a desired subword vocabulary size
3. Optimize the probability of word occurrence by giving a word sequence.
4. Compute the loss of each subword
5. Sort the symbol by loss and keep top X % of word (e.g. X can be 80).
To avoid out-of-vocabulary, character level is recommend to be included as
subset of subword.
6. Repeating step 3–5until reaching subword vocabulary size which is defined
in step 2 or no change in step 5.

## SentencePiece

So, any existing library which we can leverage it for our text processing?
Kudo and Richardson implemented [SentencePiece](https://github.com/google/sentencepiece)
library. You have to train your tokenizer based on your data such that you
can encode and decoding your data for downstream tasks.

First of all, preparing a plain text including your data and then triggering
the following API to train the model

```python
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=test/botchan.txt --model_prefix=m' \
    '--vocab_size=1000')
```

It is super fast and you can load the model by

```python
sp = spm.SentencePieceProcessor()
sp.Load("m.model")
```

To encode your text, you just need to

```python
sp.EncodeAsIds("This is a test")
```

For more examples and usages, you can access this [repo](https://github.com/google/sentencepiece/blob/master/python/README.md).

## Take Away

- Subword balances vocabulary size and footprint. Extreme case is we can
only use 26 token (i.e. character) to present all English word. 16k or 32k
subwords are recommended vocabulary size to have a good result.
- Many Asian language word cannot be separated by space. Therefore, the initial
vocabulary is larger than English a lot. You may need to prepare over 10k
initial word to kick start the word segmentation. From Schuster and Nakajima
research, they propose to use 22k word and 11k word for Japanese and Korean
respectively.

## Extension Reading

- [Classic word representation](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a)
- [Character embeddings](https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10)
- [Too powerful NLP model (GPT-2)](https://towardsdatascience.com/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655)
- [SentencePiece GIT repo](https://github.com/google/sentencepiece)

## Reference

- _T. Kudo and J. Richardson. [SentencePiece: A simple and language independent
subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf). 2018_
- _R. Sennrich, B. Haddow and A. Birch. [Neural Machine Translation of Rare
Words with Subword Units](http://aclweb.org/anthology/P16-1162). 2015_
- _M. Schuster and K. Nakajima. [Japanese and Korea Voice Search](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf). 2012
- _Taku Kudo. [Subword Regularization: Improving Neural Network Translation_
Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf). 2018_
