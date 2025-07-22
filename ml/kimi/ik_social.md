# remove a post

promote 
95 confident that the post is harmful

precision threshold

latency

business constraints

1 billion posts

detect them as soon as created

wait until you review

no. of views

doesn't need 
throughput latency

business 

# data/features

do we have a labeled data set

text from dataset

has this post

comments

report buttons

brainstorm different

reports/view

normalize some data

tokenize

what tokenizer

BPE

image embedding for the imaging how to tokenize

# modeling architecture

multimodal transformer model text and image data

text image in posts

transformer

contextual features

late fusion

output of the model

harmful or not

reports per view

super expensive to train

1 billion posts per days

too expensive to evaluate on
every single posts

lightweight model

logistic regression

GBT - super fast to train
and super fast to evalute

multistage problem

user posts -> light weight model

threshold to optimze for recall

5% gets to the transformer

50 million to the transformer

monitor the posts views and comments and reports

once a post reaches 1000 views

contextual features have changed

can cache 

# inference

# evaluation

online and offline evaluation

A/B test make sure
when you evalute make

multi-task learning

1/0

number of reports/views

final number

input features 

predict the final number

0.8 high threshold

0.1 is harmful - wanna catch as much as you can

Tune that one for precision.


netflix recommendation
home feed

doordash
logists company

advertising
recommend right ads




