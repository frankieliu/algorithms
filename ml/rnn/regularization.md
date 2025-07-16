Regularizers that'll work best will depend on your specific architecture, data, and problem; as usual, there isn't a single cut to rule all, but there _are_ do's and (especially) don't's, as well as _systematic means_ of determining what'll work best - via careful introspection and evaluation.

---

**How does RNN regularization work?**

Perhaps the best approach to understanding it is _information_\-based. First, see "How does 'learning' work?" and "RNN: Depth vs. Width". To understand RNN regularization, one must understand how RNN handles information and learns, which the referred sections describe (though not exhaustively). Now to answer the question:

RNN regularization's goal is any regularization's goal: maximizing information utility and traversal of the test loss function. The specific _methods_, however, tend to differ substantially for RNNs per their recurrent nature - and some work better than others; see below.

---

**RNN regularization methods**:

**WEIGHT DECAY**

1. **General**: shrinks the norm ('average') of the weight matrix
    
    - _Linearization_, depending on activation; e.g. `sigmoid`, `tanh`, but less so `relu`
    - _Gradient boost_, depending on activation; e.g. `sigmoid`, `tanh` grads flatten out for large activations - linearizing enables neurons to keep learning
2. **Recurrent weights**: default `activation='sigmoid'`
    
    - **Pros**: linearizing can help BPTT (remedy vanishing gradient), hence also _learning long-term dependencies_, as _recurrent information utility_ is increased
    - **Cons**: linearizing can harm representational power - however, this can be offset by stacking RNNs
3. **Kernel weights**: for many-to-one (`return_sequences=False`), they work similar to weight decay on a typical layer (e.g. `Dense`). For many-to-many (`=True`), however, kernel weights operate on every timestep, so pros & cons similar to above will apply.
    

**Dropout**:

- **Activations** (kernel): can benefit, but only if limited; values are usually kept less than `0.2` in practice. Problem: tends to introduce too much noise, and erase important context information, especially in problems w/ limited timesteps.
- **Recurrent activations** (`recurrent_dropout`): the [recommended dropout](https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout)

**Batch Normalization**:

- **Activations** (kernel): worth trying. Can benefit substantially, or not.
- **Recurrent activations**: should work better; see [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025). No Keras implementations yet as far as I know, but I may implement it in the future.

**Weight Constraints**: set hard upper-bound on weights l2-norm; possible alternative to weight decay.

**Activity Constraints**: don't bother; for most purposes, if you have to manually constrain your outputs, the layer itself is probably learning poorly, and the solution is elsewhere.

---

**What should I do?** Lots of info - so here's some concrete advice:

1. **Weight decay**: try `1e-3`, `1e-4`, see which works better. Do _not_ expect the same value of decay to work for `kernel` and `recurrent_kernel`, especially depending on architecture. Check weight shapes - if one is much smaller than the other, apply smaller decay to former
    
2. **Dropout**: try `0.1`. If you see improvement, try `0.2` - else, scrap it
    
3. **Recurrent Dropout**: start with `0.2`. Improvement --> `0.4`. Improvement --> `0.5`, else `0.3`.
    
4. **Batch Normalization**: try. Improvement --> keep it - else, scrap it.
5. **Recurrent Batchnorm**: same as 4.
6. **Weight constraints**: advisable w/ higher learning rates to prevent exploding gradients - else use higher weight decay
7. **Activity constraints**: probably not (see above)
8. **Residual RNNs**: introduce significant changes, along a regularizing effect. See application in [IndRNNs](https://arxiv.org/abs/1803.04831)
9. **Biases**: weight decay and constraints become important upon attaining good backpropagation properties; without them on bias weights but _with_ them on kernel (K) & recurrent kernel (RK) weights, bias weights may grow much faster than the latter two, and dominate the transformation - also leading to exploding gradients. I recommend weight decay / constraint less than or equal to that used on K & RK. Also, with `BatchNormalization`, you ~can~ _cannot_ set `use_bias=False` as an "equivalent"; BN applies to _outputs_, not _hidden-to-hidden transforms_.
10. **Zoneout**: don't know, never tried, might work - see [paper](https://arxiv.org/abs/1606.01305).
11. **Layer Normalization**: some report it working better than BN for RNNs - but my application found it otherwise; [paper](https://arxiv.org/abs/1607.06450)
12. **Data shuffling**: is a strong regularizer. Also shuffle _batch samples_ (samples in batch). See relevant info on [stateful RNNs](https://stackoverflow.com/questions/58276337/proper-way-to-feed-time-series-data-to-stateful-lstm/58277760#58277760)
13. **Optimizer**: can be an inherent regularizer. Don't have a full explanation, but in my application, Nadam (& NadamW) has stomped every other optimizer - worth trying.

**Introspection**: bottom section on 'learning' isn't worth much without this; don't just look at validation performance and call it a day - _inspect_ the effect that adjusting a regularizer has on _weights_ and _activations_. Evaluate using info toward bottom & relevant theory.

**BONUS**: weight decay can be powerful - even more powerful when done right; turns out, _adaptive optimizers_ like Adam can harm its effectiveness, as described in [this paper](https://arxiv.org/abs/1711.05101). _Solution_: use AdamW. My Keras/TensorFlow implementation [here](https://github.com/OverLordGoldDragon/keras-adamw).

---

**This is too much!** Agreed - welcome to Deep Learning. Two tips here:

1. [_Bayesian Optimization_](https://philipperemy.github.io/visualization/); will save you time especially on prohibitively expensive training.
2. `Conv1D(strides > 1)`, for many timesteps (`>1000`); slashes dimensionality, shouldn't harm performance (may in fact improve it).

---

**Introspection Code**:

**Gradients**: see [this answer](https://stackoverflow.com/questions/59017288/how-to-visualize-rnn-lstm-gradients-in-keras-tensorflow/59017289#59017289)

**Weights**: see [this answer](https://stackoverflow.com/questions/59275959/how-to-visualize-rnn-lstm-weights-in-keras-tensorflow/59275960#59275960)

**Weight norm tracking**: see [this Q & A](https://stackoverflow.com/questions/61481921/how-to-set-and-track-weight-decays/61481922#61481922)

**Activations**: see [this answer](https://stackoverflow.com/questions/58356868/how-visualize-attention-lstm-using-keras-self-attention-package/58357581#58357581)

**Weights**: [`see_rnn.rnn_histogram`](https://github.com/OverLordGoldDragon/see-rnn/blob/master/see_rnn/visuals_rnn.py#L10) or [`see_rnn.rnn_heatmap`](https://github.com/OverLordGoldDragon/see-rnn/blob/master/see_rnn/visuals_rnn.py#L264) (examples in README)

---

**How does 'learning' work?**

The 'ultimate truth' of machine learning that is seldom discussed or emphasized is, **we don't have access to the function we're trying to optimize** - the _test loss function_. _All_ of our work is with what are _approximations_ of the true loss surface - both the train set and the validation set. This has some critical implications:

1. Train set global optimum can lie _very far_ from test set global optimum
2. Local optima are unimportant, and irrelevant:
    - Train set local optimum is almost always a better test set optimum
    - Actual local optima are almost impossible for high-dimensional problems; for the case of the "saddle", you'd need the gradients w.r.t. _all of the millions of parameters_ to equal zero at once
    - [Local attractors](https://www.wikiwand.com/en/Attractor) are lot more relevant; the analogy then shifts from "falling into a pit" to "gravitating into a strong field"; once in that field, your loss surface topology is bound to that set up by the field, which defines its own local optima; high LR can help exit a field, much like "escape velocity"

Further, loss functions are way too complex to analyze directly; a better approach is to _localize_ analysis to individual layers, their weight matrices, and roles relative to the entire NN. Two key considerations are:

3. **Feature extraction capability**. _Ex_: the driving mechanism of deep classifiers is, given input data, to _increase class separability_ with each layer's transformation. Higher quality features will filter out irrelevant information, and deliver what's essential for the output layer (e.g. softmax) to learn a separating hyperplane.
    
4. **Information utility**. _Dead neurons_, and _extreme activations_ are major culprits of poor information utility; no single neuron should dominate information transfer, and too many neurons shouldn't lie purposeless. Stable activations and weight distributions enable gradient propagation and continued learning.
    

---

**How does regularization work?** read above first

In a nutshell, via maximizing NN's information utility, and improving estimates of the test loss function. Each regularization method is unique, and no two exactly alike - see "RNN regularizers".

---

**RNN: Depth vs. Width**: not as simple as "one is more nonlinear, other works in higher dimensions".

- **RNN width** is defined by (1) # of input channels; (2) # of cell's filters (output channels). As with CNN, each RNN filter is an _independent feature extractor_: _more_ is suited for higher-complexity information, including but not limited to: dimensionality, modality, noise, frequency.
- **RNN depth** is defined by (1) # of stacked layers; (2) # of timesteps. Specifics will vary by architecture, but from information standpoint, unlike CNNs, RNNs are _dense_: every timestep influences the ultimate output of a layer, hence the ultimate output of the next layer - so it again isn't as simple as "more nonlinearity"; stacked RNNs exploit both spatial and temporal information.

---

**Update**:

Here is an example of a near-ideal RNN gradient propagation for 170+ timesteps:

![](https://i.sstatic.net/71seM.png)

This is rare, and was achieved via careful regularization, normalization, and hyperparameter tuning. Usually we see a large gradient for the last few timesteps, which drops off sharply toward left - as [here](https://stackoverflow.com/questions/59017288/how-to-visualize-rnn-lstm-gradients-in-keras-tensorflow/59017289#59017289). Also, since the model is stateful and fits 7 equivalent windows, gradient effectively spans **1200 timesteps**.

**Update 2**: see 9 w/ new info & correction

**Update 3**: add weight norms & weights introspection code