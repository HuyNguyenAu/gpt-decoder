# GPT Decoder
A simple GPT decoder for educational purposes following Andrej Karpathy's 'Let's build GPT: from scratch, in code, spelled out'.

## Requirements
- Python 3.11
  - Pytorch 2.0.1

## Getting Started
Install dependencies:
```
Go to https://pytorch.org and install Pytorch.=
```

Run the decoder training:
```
$ python gpt-decoder.py 
```

## Optimisations

### Improving Transformer Optimization Through Better Initialization Paper
- Added Xavier Uniform Init.
- Added T-FixUp Init.

### Primer: Searching for Efficient Transformers for Language Modeling Paper
- Added SquaredReLU.
- Added Depthwise Convolution.
