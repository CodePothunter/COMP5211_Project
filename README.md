# COMP5211 Final Project Tutorials

In the final project, you will need to run inference of BERT for your deisgned
task. These tutorials will help you get quick start to how to use BERT, as well
as how to install the environment, if you want to run on your own computer.

[Installation Introduction](Bert-as-Service_Install.md)

[General Using Guide of Bert-as-Service](https://github.com/hanxiao/bert-as-service#book-tutorial)

[A 2-minute Example](application.md)

You are highly recommanded to use your own computer as there are too many constraints
when using CSD machines. Eventhough you can also utilize the machines provided by the 
CSE department. For instruction, please refer to 
[How to Run on CSE Department Machine](cse_lab_machine.md)

<!-- **Don't forget to register your group at [this link](http://bit.ly/comp5211group)** -->

If you encounter any problem, please feel free to send me email, or **create an 
issue (recommended, so that everyone can learn from it)**. The response time will be normally within two hours during 8:00-22:00.

This page will keep updated for more tutorials and FAQ.
----

**2019-11-27: [A 2-minute Example](application.md) and [Installation Introduction](Bert-as-Service_Install.md) updated**

## Q1: How to do `[MASK]` in the query?

It can be achieved if you go through the [General Using Guide of Bert-as-Service](https://github.com/hanxiao/bert-as-service#book-tutorial) carefully. 
Here I will present a simple method to achieve it. At the begining, please ensure that your server has turn on the following flags

```bash
    -pooling_strategy=NONE -show_tokens_to_client
```

which enables each-token representation, and allows you to check our parsed 
tokens in the client. After doing this, the `encode()` method should be called as

```python
vec = bc.encode(['hey you', 'whats up?'])[:,0,:]
# this provide you a [CLS] represntation for each sentence, instead of mean pooling.
# while you can also implement any pooling method as you like.
```

If you want to check the parsed token, you can type

```python
print(bc.encode(['hey you', 'whats up?'], show_tokens=True))

(array([[[ 0.1649077 ,  0.77392733,  0.20736952, ..., -0.39933944,
         -0.3007825 ,  0.29385632],
        [ 0.35753685,  0.06508213,  0.76516646, ..., -0.33485305,
         -0.07497814, -0.5159149 ],
        [-0.76333636, -0.6616222 ,  0.12231285, ..., -0.11380126,
          0.4786739 , -0.48920187],
        ...,
        [ 0.        ,  0.        ,  0.        , ..., -0.        ,
          0.        , -0.        ],
        [ 0.        ,  0.        ,  0.        , ..., -0.        ,
          0.        , -0.        ],
        [-0.        ,  0.        ,  0.        , ..., -0.        ,
         -0.        ,  0.        ]],

       [[ 0.318993  , -0.41422266,  0.02958095, ..., -0.6185721 ,
          0.03289787, -0.14265244],
        [ 0.21804485, -0.2876658 ,  0.5752587 , ...,  0.27102396,
         -0.08176719, -1.1708245 ],
        [-0.70380414, -0.84018505,  1.080997  , ..., -0.65443337,
          0.77894425, -0.59493685],
        ...,
        [ 0.        ,  0.        ,  0.        , ...,  0.        ,
          0.        , -0.        ],
        [-0.        , -0.        ,  0.        , ..., -0.        ,
         -0.        , -0.        ],
        [ 0.        , -0.        ,  0.        , ...,  0.        ,
         -0.        , -0.        ]]], dtype=float32), 
[['[CLS]', 'hey', 'you', '[SEP]'], ['[CLS]', 'what', '##s', 'up', '?', '[SEP]']])
```

**If you want to mask some words in the query, you WILL NEED to use customized tokenizer (e.g. NLTK tokenizer).**
**Please be noted that our pretrained model is _uncased_, so you will need to lower**
**all tokens**. For example, if you only split the sentence by space:

```python
print(bc.encode(['hey you'.lower().split(), 'whats up?'.lower().split()], 
    show_tokens=True, is_tokenized=True))

(array([[[ 0.1649077 ,  0.77392733,  0.20736952, ..., -0.39933944,
         -0.3007825 ,  0.29385632],
        [ 0.35753685,  0.06508213,  0.76516646, ..., -0.33485305,
         -0.07497814, -0.5159149 ],
        [-0.76333636, -0.6616222 ,  0.12231285, ..., -0.11380126,
          0.4786739 , -0.48920187],
        ...,
        [ 0.        ,  0.        ,  0.        , ..., -0.        ,
          0.        , -0.        ],
        [ 0.        ,  0.        ,  0.        , ..., -0.        ,
          0.        , -0.        ],
        [-0.        ,  0.        ,  0.        , ..., -0.        ,
         -0.        ,  0.        ]],

       [[-0.4868275 ,  0.37581897, -0.78558445, ..., -0.50906205,
         -0.12649894,  0.22267678],
        [-0.9576893 ,  0.84955466,  0.0573747 , ..., -0.65695876,
         -0.03366615, -0.5995212 ],
        [-1.0952191 ,  0.31994814, -0.7178457 , ..., -0.14848474,
         -0.34432426, -0.942693  ],
        ...,
        [ 0.        , -0.        , -0.        , ..., -0.        ,
          0.        , -0.        ],
        [ 0.        , -0.        , -0.        , ..., -0.        ,
          0.        , -0.        ],
        [ 0.        , -0.        , -0.        , ...,  0.        ,
          0.        , -0.        ]]], dtype=float32), 
[['[CLS]', 'hey', 'you', '[SEP]'], ['[CLS]', '[UNK]', '[UNK]', '[SEP]']])
```

You may discover, the word `whats` is replaced with `[UNK]`, while in the upper
case it is correctly tokenized as `what` and `##s`. That's why we need a tokenier. 
Now, let's turn to masking, although it is almost there. To be more clear, we 
first define a toy tokenier, and call it when encoding:

```python
def myTKNZ(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('whats', 'what ##s')
    sentence = sentence.replace('up?', 'up ?')
    sentence = sentence.replace('[mask]', '[MASK]')
    return sentence.split(' ')

print(bc.encode([myTKNZ('hey [MASK]'), myTKNZ('whats [MASK] up?')], 
    show_tokens=True, is_tokenized=True))

(array([[[ 0.22764891,  0.5705535 ,  0.21054205, ..., -0.37580425,
         -0.8027175 , -0.26455823],
        [-0.22807643, -0.2704285 ,  0.5898228 , ..., -0.00814681,
         -0.34706724, -0.33271754],
        [-1.4227939 , -0.577669  ,  0.00882812, ..., -0.0088397 ,
         -0.36877272, -0.02079812],
        ...,
        [-0.        ,  0.        ,  0.        , ..., -0.        ,
         -0.        ,  0.        ],
        [-0.        ,  0.        ,  0.        , ..., -0.        ,
         -0.        ,  0.        ],
        [-0.        ,  0.        ,  0.        , ..., -0.        ,
         -0.        ,  0.        ]],

       [[ 0.5089544 ,  0.3035191 ,  0.10707118, ..., -0.5996052 ,
         -0.45790258, -0.03997044],
        [ 0.16432127, -0.11981091,  0.28169835, ..., -0.18201783,
         -0.26462194, -0.77857745],
        [-0.59251356, -0.62193507,  0.8270298 , ..., -0.42277074,
          0.60381263, -0.20892641],
        ...,
        [ 0.        , -0.        ,  0.        , ...,  0.        ,
         -0.        , -0.        ],
        [ 0.        ,  0.        ,  0.        , ...,  0.        ,
          0.        , -0.        ],
        [-0.        ,  0.        , -0.        , ..., -0.        ,
         -0.        ,  0.        ]]], dtype=float32), 
[['[CLS]', 'hey', '[MASK]', '[SEP]'], ['[CLS]', 'what', '##s', '[MASK]', 'up', '?', '[SEP]']])
```

You, the clever students, are able to find a BERT tokenizer by yourselves!