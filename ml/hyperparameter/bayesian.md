![](https://miro.medium.com/v2/resize:fit:1875/1*CtJD4zJr6PNxbUZMwZxPKA.jpeg)

# A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning

## The concepts behind efficient hyperparameter tuning using Bayesian optimization

[ ![Will Koehrsen](https://miro.medium.com/v2/resize:fill:64:64/1*SckxdIFfjlR-cWXkL5ya-g.jpeg) ](https://medium.com/@williamkoehrsen?source=post_page---byline--b8172278050f---------------------------------------)

[Will Koehrsen](https://medium.com/@williamkoehrsen?source=post_page---byline--b8172278050f---------------------------------------)

Follow

14 min read

·

Jun 24, 2018

6.3K

32

[

Listen









](https://medium.com/plans?dimension=post_audio_button&postId=b8172278050f&source=upgrade_membership---post_audio_button-----------------------------------------)

Share

More

Following are four common methods of hyperparameter optimization for machine learning in order of increasing efficiency:

1. **Manual**
2. **Grid search**
3. **Random search**
4. **Bayesian model-based optimization**

(There are also other methods such as [evolutionary](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Evolutionary_optimization) and [gradient-based](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Gradient-based_optimization).)

I was pretty proud that I’d recently moved up the ladder from manual to random search until I found this image deep in a [paper by Bergstra et al.](http://proceedings.mlr.press/v28/bergstra13.pdf):

Zoom image will be displayed

![](https://miro.medium.com/v2/resize:fit:1875/1*E0_THdPH2NfKB37JUQB8Eg.png)

Validation Errors comparing random search and a model based approach on LFW (left) and PubFig83 (right)

These figures compare validation error for hyperparameter optimization of an image classification neural network with random search in grey and Bayesian Optimization (using the Tree Parzen Estimator or TPE) in green. Lower is better: a smaller validation set error generally means better test set performance, and a smaller number of trials means less time invested. Clearly, there are significant advantages to Bayesian methods, and these graphs, along with [other impressive results,](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) convinced me it was time to take the next step and learn model-based hyperparameter optimization.

The one-sentence summary of [Bayesian hyperparameter optimization](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) is: build a probability model of the objective function and use it to select the most promising hyperparameters to evaluate in the true objective function.

If you like to operate at a very high level, then this sentence may be all you need. However, if you want to understand the details, this article is my attempt to outline the concepts behind Bayesian optimization, in particular Sequential Model-Based Optimization (SMBO) with the Tree Parzen Estimator (TPE). With the mindset that you don’t know a concept until you can explain it to others, I went through [several academic](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) [papers](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) and will try to communicate the results in a (relatively) easy to understand format.

Although we can often implement machine learning methods without understanding how they work, I like to try and get an idea of what is going on so I can use the technique as effectively as possible. In later articles I’ll walk through using these methods in Python using libraries such as [Hyperopt](https://github.com/hyperopt/hyperopt), so this article will lay the conceptual groundwork for implementations to come!

Update: [Here is a brief Jupyter Notebook](https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Introduction%20to%20Bayesian%20Optimization%20with%20Hyperopt.ipynb) showing the basics of using Bayesian Model-Based Optimization in the Hyperopt Python library.

# Hyperparameter Optimization

The aim of hyperparameter optimization in machine learning is to find the hyperparameters of a given machine learning algorithm that return the best performance as measured on a validation set. (Hyperparameters, in contrast to model parameters, are set by the machine learning engineer before training. The number of trees in a random forest is a hyperparameter while the weights in a neural network are model parameters learned during training. I like to think of hyperparameters as the model settings to be tuned.)

Hyperparameter optimization is represented in equation form as:

Zoom image will be displayed

![](https://miro.medium.com/v2/resize:fit:731/1*QR4_VOfAAWLVe2I0nqwtTg.png)

Here f(x) represents an objective score to minimize— such as RMSE or error rate— evaluated on the validation set; x\* is the set of hyperparameters that yields the lowest value of the score, and x can take on any value in the domain X. In simple terms, we want to **find the model hyperparameters that yield the best score on the validation set metric**.

The problem with hyperparameter optimization is that evaluating the objective function to find the score is extremely expensive. Each time we try different hyperparameters, we have to train a model on the training data, make predictions on the validation data, and then calculate the validation metric. With a large number of hyperparameters and complex models such as ensembles or deep neural networks that can take days to train, this process quickly becomes intractable to do by hand!

Grid search and [random search are slightly better](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) than manual tuning because we set up a grid of model hyperparameters and run the train-predict -evaluate cycle automatically in a loop while we do more productive things (like [feature engineering](https://www.featuretools.com/)). However, even these methods are relatively inefficient because they do not choose the next hyperparameters to evaluate based on previous results. **Grid and random search are completely _uninformed_ by past evaluations,** and as a result, often spend a significant amount of time evaluating “bad” hyperparameters.

For example, if we have the following graph with a lower score being better, where does it make sense to concentrate our search? If you said below 200 estimators, then you already have the idea of Bayesian optimization! We want to focus on the most promising hyperparameters, and if we have a record of evaluations, then it makes sense to use this information for our next choice.

![](https://miro.medium.com/v2/resize:fit:1050/1*MiNXGrkk5BbjfkNAXZQSNA.png)

Random and grid search pay no attention to past results at all and would keep searching across the entire range of the number of estimators even though it’s clear the optimal answer (probably) lies in a small region!

# Bayesian Optimization

[Bayesian approaches](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf), in contrast to random or grid search, keep track of past evaluation results which they use to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function:

![](https://miro.medium.com/v2/resize:fit:576/1*u00KlxHhd1fz6-Jaeou6PA.png)

[In the literature](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf), this model is called a “surrogate” for the objective function and is represented as p(y | x). The surrogate is much easier to optimize than the objective function and Bayesian methods work by finding the next set of hyperparameters to evaluate on the actual objective function by selecting hyperparameters that perform best on the surrogate function. In other words:

1. **Build a surrogate probability model of the objective function**
2. **Find the hyperparameters that perform best on the surrogate**
3. **Apply these hyperparameters to the true objective function**
4. **Update the surrogate model incorporating the new results**
5. **Repeat steps 2–4 until max iterations or time is reached**

The [aim of Bayesian reasoning is to become “less wrong” with more data](https://towardsdatascience.com/bayes-rule-applied-75965e4482ff) which these approaches do by continually updating the surrogate probability model after each evaluation of the objective function.

At a high-level, Bayesian optimization methods are efficient because they choose the next hyperparameters in an _informed manner_**.** The basic idea is: **spend a little more time selecting the next hyperparameters in order to make fewer calls to the objective function.** In practice, the time spent selecting the next hyperparameters is inconsequential compared to the time spent in the objective function. By evaluating hyperparameters that appear more promising from past results, Bayesian methods can find better model settings than random search in fewer iterations.

If there’s one thing to take away from this article it’s that [Bayesian model-based methods](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Bayesian_optimization) can find better hyperparameters in less time because they reason about the best set of hyperparameters to evaluate based on past trials.

As a good visual description of what is occurring in Bayesian Optimization take a look at the images below ([source](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf)). The first shows an initial estimate of the surrogate model — in black with associated uncertainty in gray — after two evaluations. Clearly, the surrogate model is a poor approximation of the actual objective function in red:

Zoom image will be displayed

![](https://miro.medium.com/v2/resize:fit:1313/1*RQ-pAwQ88yC904QppChGPQ.png)

The next image shows the surrogate function after 8 evaluations. Now the surrogate almost exactly matches the true function. Therefore, if the algorithm selects the hyperparameters that maximize the surrogate, they will likely yield very good results on the true evaluation function.

Zoom image will be displayed

![](https://miro.medium.com/v2/resize:fit:1313/1*bSLAe1LCj3mMKfaZsQWCrw.png)

Bayesian methods have always made sense to me because they operate in much the same way we do: we form an initial view of the world (called a prior) and then we update our model based on new experiences (the updated model is called a posterior). Bayesian hyperparameter optimization takes that framework and applies it to finding the best value of model settings!

## Sequential Model-Based Optimization

[Sequential model-based optimization (SMBO) methods (SMBO)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) are a formalization of Bayesian optimization. The sequential refers to running trials one after another, each time trying better hyperparameters by applying Bayesian reasoning and updating a probability model (surrogate).

There are five aspects of model-based hyperparameter optimization:

1. **A domain of hyperparameters over which to search**
2. **An objective function which takes in hyperparameters and outputs a score that we want to minimize (or maximize)**
3. **The surrogate model of the objective function**
4. **A criteria, called a selection function, for evaluating which hyperparameters to choose next from the surrogate model**
5. **A history consisting of (score, hyperparameter) pairs used by the algorithm to update the surrogate model**

There are several variants of [SMBO methods that differ](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) in steps 3–4, namely, how they build a surrogate of the objective function and the criteria used to select the next hyperparameters. Several common choices for the surrogate model are [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process), [Random Forest Regressions](http://aad.informatik.uni-freiburg.de/papers/13-GECCO-BBOB_SMAC.pdf), and Tree Parzen Estimators (TPE) while the most common choice for step 4 is Expected Improvement. In this post, we will focus on TPE and Expected Improvement.

## Domain

In the case of random search and grid search, the domain of hyperparameters we search is a grid. An example for a random forest is shown below:

For a model-based approach, the domain consists of _probability distributions_. As with a grid, this lets us encode domain knowledge into the search process by placing greater probability in regions where we think the true best hyperparameters lie. If we wanted to express the above grid as a probability distribution, it may look something like this:

Zoom image will be displayed![](https://miro.medium.com/v2/resize:fit:788/1*luY6Ahh7uttR4quIcgOCBw.png)

Zoom image will be displayed![](https://miro.medium.com/v2/resize:fit:746/1*YfoPLKK8_WXIsRaQ7zcSjg.png)

Zoom image will be displayed![](https://miro.medium.com/v2/resize:fit:741/1*e6cIETdFd1rzD9ivofNJqw.png)

Here we have a uniform, log-normal, and normal distribution. These are informed by prior practice/knowledge (for example the [learning rate domain is usually a log-normal distribution over several orders of magnitude](https://www.kdnuggets.com/2017/11/estimating-optimal-learning-rate-deep-neural-network.html)).

## Objective Function

The objective function takes in hyperparameters and outputs a single real-valued score that we want to minimize (or maximize). As an example, let’s consider the case of building a random forest for a regression problem. The hyperparameters we want to optimize are shown in the hyperparameter grid above and the score to minimize is the Root Mean Squared Error. Our objective function would then look like (in Python):

While the objective function looks simple, it is very expensive to compute! If the objective function could be quickly calculated, then we could try every single possible hyperparameter combination (like in grid search). If we are using a simple model, a small hyperparameter grid, and a small dataset, then this might be the best way to go. However, in cases where the objective function may take hours or even days to evaluate, we want to limit calls to it.

The entire concept of Bayesian model-based optimization is to reduce the number of times the objective function needs to be run by choosing only the most promising set of hyperparameters to evaluate based on previous calls to the evaluation function. The next set of hyperparameters are selected based on a model of the objective function called a surrogate.

## Surrogate Function (Probability Model)

The surrogate function, also called the response surface, is the probability representation of the objective function built using previous evaluations. This is called sometimes called a response surface because it is a high-dimensional mapping of hyperparameters to the probability of a score on the objective function. Below is a simple example with only two hyperparameters:

![](https://miro.medium.com/v2/resize:fit:900/0*aBsprZzniYMB0KWc.png)

Response surface for AdaBoost algorithm ([Source](http://www.hylap.org/meta_data/adaboost/))

There are several different forms of the surrogate function including Gaussian Processes and Random Forest regression. However, in this post we will focus on the Tree-structured Parzen Estimator as [put forward by Bergstra et al](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) in the paper “Algorithms for Hyper-Parameter Optimization”. These methods differ in how they construct the surrogate function which we’ll explain in just a bit. First we need to talk about the selection function.

## Selection Function

The selection function is the criteria by which the next set of hyperparameters are chosen from the surrogate function. The most common choice of criteria is Expected Improvement:

![](https://miro.medium.com/v2/resize:fit:671/1*ebsqjhOTSGKBbIR_RLkjSQ.png)

Here y\* is a threshold value of the objective function, x is the proposed set of hyperparameters, y is the actual value of the objective function using hyperparameters x, and p(y | x) is the surrogate probability model expressing the probability of y given x. If that’s all a little much, in simpler terms, **the aim is to maximize the Expected Improvement with respect to x.** This means finding the best hyperparameters under the surrogate function p (y | x).

If p (y | x) is zero everywhere that y < y\*, then the hyperparameters x are not expected to yield any improvement. If the integral is positive, then it means that the hyperparameters x are expected to yield a better result than the threshold value.

**Tree-structured Parzen Estimator (TPE)**

Now let’s get back to the surrogate function. The methods of SMBO differ in how they construct the surrogate model p(y | x). The Tree-structured Parzen Estimator builds a model by applying Bayes rule. Instead of directly representing p( y | x), it instead uses:

![](https://miro.medium.com/v2/resize:fit:480/1*4D1QpDZzWpBOl7ANBhsSJA.png)

Bayes Rule in Action!

p (x | y), which is the probability of the hyperparameters given the score on the objective function, in turn is expressed:

![](https://miro.medium.com/v2/resize:fit:546/1*idWxsGylqq2ZaMGpHmbxDg.png)

where y < y\* represents a lower value of the objective function than the threshold. The explanation of this equation is that we make _two different distributions for the hyperparameters_: one where the value of the objective function is less than the threshold, _l(x),_ and one where the value of the objective function is greater than the threshold, _g(x)_.

Let’s update our Random Forest graph to include a threshold:

![](https://miro.medium.com/v2/resize:fit:1050/1*H5pyf3G115WGJwPpg65yaQ.png)

Now we construct two probability distributions for the number of estimators, one using the estimators that yielded values under the threshold and one using the estimators that yielded values above the threshold.

![](https://miro.medium.com/v2/resize:fit:1088/1*6SH5O_ail54karro8j0NGg.png)

Intuitively, it seems that we want to draw values of x from _l(x)_ and not from _g(x)_ because this distribution is based only on values of x that yielded lower scores than the threshold. It turns out this is exactly what the math says as well! With Bayes Rule, and a few substitutions, the expected improvement equation (which we are trying to maximize) becomes:

![](https://miro.medium.com/v2/resize:fit:1168/1*ybiePL_8lKNouHlSb5OSgQ.png)

The term on the far right is the most important part. What this says is that the [Expected Improvement](https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf) is proportional to the ratio _l(x) / g(x)_ and therefore, to maximize the Expected Improvement, we should maximize this ratio. Our intuition was correct: we should draw values of the hyperparameters which are more likely under _l(x)_ than under _g(x)_!

The Tree-structured Parzen Estimator works by drawing sample hyperparameters from _l(x)_, evaluating them in terms of _l(x) / g(x)_, and returning the set that yields the highest value under _l(x) / g(x)_ corresponding to the greatest expected improvement_._ These hyperparameters are then evaluated on the objective function. If the surrogate function is correct, then these hyperparameters should yield a better value when evaluated!

The expected improvement criteria allows the model to balance [exploration versus exploitation](https://en.wikipedia.org/wiki/Multi-armed_bandit). _l(x)_ is a distribution and not a single value which means that the hyperparameters drawn are likely close but not exactly at the maximum of the expected improvement. Moreover, because the surrogate is just an estimate of the objective function, the selected hyperparameters may not actually yield an improvement when evaluated and the surrogate model will have to be updated. This updating is done based on the current surrogate model and the history of objective function evaluations.

## History

Each time the algorithm proposes a new set of candidate hyperparameters, it evaluates them with the actual objective function and records the result in a pair (score, hyperparameters). These records form the **history**. The algorithm builds _l(x)_ and _g(x)_ using the history to come up with a probability model of the objective function that improves with each iteration.

[This is Bayes’ Rule at work](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7): we have an initial estimate for the surrogate of the objective function that we update as we gather more evidence. Eventually, with enough evaluations of the objective function, we hope that our model accurately reflects the objective function and the hyperparameters that yield the greatest Expected Improvement correspond to the hyperparameters that maximize the objective function.

# **Putting it All Together**

How do Sequential Model-Based Methods help us more efficiently search the hyperparameter space? Because the algorithm is proposing better candidate hyperparameters for evaluation, the score on the objective function improves much more rapidly than with random or grid search leading to fewer overall evaluations of the objective function.

Even though the algorithm spends more time selecting the next hyperparameters by maximizing the Expected Improvement, this is much cheaper in terms of computational cost than evaluating the objective function. [In a paper about using SMBO with TPE](http://proceedings.mlr.press/v28/bergstra13.pdf), the authors reported that finding the next proposed set of candidate hyperparameters took several seconds, while evaluating the actual objective function took hours.

If we are using better-informed methods to choose the next hyperparameters, that means we can spend less time evaluating poor hyperparameter choices. Furthermore, sequential model-based optimization using tree-structured Parzen estimators is able to find better hyperparameters than random search in the same number of trials. In other words, we get

- Reduced running time of hyperparameter tuning
- Better scores on the testing set

Hopefully, this has convinced you Bayesian model-based optimization is a technique worth trying!

## Implementation

Fortunately for us, there are now a number of libraries that can do SMBO in Python. [Spearmint](https://github.com/JasperSnoek/spearmint) and [MOE](https://github.com/Yelp/MOE) use a Gaussian Process for the surrogate, [Hyperopt](https://github.com/hyperopt/hyperopt) uses the Tree-structured Parzen Estimator, and [SMAC](https://github.com/automl/SMAC3) uses a Random Forest regression. These libraries all use the Expected Improvement criterion to select the next hyperparameters from the surrogate model. In later articles we will take a look at using Hyperopt in Python and there are already [several](http://fastml.com/optimizing-hyperparams-with-hyperopt/) [good](https://github.com/jaberg/hyperopt/wiki/FMin) articles and [code examples](https://www.programcreek.com/python/example/98788/hyperopt.Trials) for learning.

# Conclusions

**Bayesian model-based optimization methods build a probability model of the objective function to propose smarter choices for the next set of hyperparameters to evaluate. SMBO is a formalization of Bayesian optimization which is more efficient at finding the best hyperparameters for a machine learning model than random or grid search.**

Sequential model-based optimization methods differ in they build the surrogate, but they all rely on information from previous trials to propose better hyperparameters for the next evaluation. The Tree Parzen Estimator is one algorithm that uses Bayesian reasoning to construct the surrogate model and can select the next hyperparameters using Expected Improvement.

There are a number of libraries to implement SMBO in Python which we will explore in further articles. The concepts are a little tough at first, but understanding them will allow us to use the tools built on them more effectively. I’d like to mention I’m still trying to work my way through the details and if I’ve made a mistake, please let me know (civilly)!

For more details, the following articles are extremely helpful:

1. Algorithms for Hyper-Parameter Optimization \[[Link](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)\]
2. Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures \[[Link](http://proceedings.mlr.press/v28/bergstra13.pdf)\]
3. Bayesian Optimization Primer \[[Link](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)\]
4. Taking the Human Out of the Loop: A Review of Bayesian Optimization \[[Link](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)\]

As always, I welcome feedback and constructive criticism. I can be reached on Twitter [@koehrsen\_will](http://twitter.com/@koehrsen_will)

6.3K

32

[

Machine Learning

](https://medium.com/tag/machine-learning?source=post_page-----b8172278050f---------------------------------------)

[

Education

](https://medium.com/tag/education?source=post_page-----b8172278050f---------------------------------------)

[

Bayesian Machine Learning

](https://medium.com/tag/bayesian-machine-learning?source=post_page-----b8172278050f---------------------------------------)

[

Computer Science

](https://medium.com/tag/computer-science?source=post_page-----b8172278050f---------------------------------------)

[

Towards Data Science

](https://medium.com/tag/towards-data-science?source=post_page-----b8172278050f---------------------------------------)

6.3K

6.3K

32

[

![TDS Archive](https://miro.medium.com/v2/resize:fill:96:96/1*JEuS4KBdakUcjg9sC7Wo4A.png)



](https://medium.com/data-science?source=post_page---post_publication_info--b8172278050f---------------------------------------)

[

![TDS Archive](https://miro.medium.com/v2/resize:fill:128:128/1*JEuS4KBdakUcjg9sC7Wo4A.png)



](https://medium.com/data-science?source=post_page---post_publication_info--b8172278050f---------------------------------------)

Following

[

## Published in TDS Archive

](https://medium.com/data-science?source=post_page---post_publication_info--b8172278050f---------------------------------------)

[828K followers](https://medium.com/data-science/followers?source=post_page---post_publication_info--b8172278050f---------------------------------------)

·[Last published Feb 3, 2025](https://medium.com/data-science/diy-ai-how-to-build-a-linear-regression-model-from-scratch-7b4cc0efd235?source=post_page---post_publication_info--b8172278050f---------------------------------------)

An archive of data science, data analytics, data engineering, machine learning, and artificial intelligence writing from the former Towards Data Science Medium publication.

Following

[

![Will Koehrsen](https://miro.medium.com/v2/resize:fill:96:96/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)



](https://medium.com/@williamkoehrsen?source=post_page---post_author_info--b8172278050f---------------------------------------)

[

![Will Koehrsen](https://miro.medium.com/v2/resize:fill:128:128/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)



](https://medium.com/@williamkoehrsen?source=post_page---post_author_info--b8172278050f---------------------------------------)

Follow

[

## Written by Will Koehrsen

](https://medium.com/@williamkoehrsen?source=post_page---post_author_info--b8172278050f---------------------------------------)

[39K followers](https://medium.com/@williamkoehrsen/followers?source=post_page---post_author_info--b8172278050f---------------------------------------)

·[20 following](https://medium.com/@williamkoehrsen/following?source=post_page---post_author_info--b8172278050f---------------------------------------)

Senior Machine Learning Engineer at Cortex Sustainability Intelligence

Follow

## Responses (32)

[](https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page---post_responses--b8172278050f---------------------------------------)

![Frankie Liu](https://miro.medium.com/v2/resize:fill:64:64/0*EtpEh9TMeb86cDvu.)

Frankie Liu

What are your thoughts?﻿

Cancel

Respond

[

![Michael Klear](https://miro.medium.com/v2/resize:fill:64:64/1*IX7DnYLLQrpEInHrTOwVDw.jpeg)



](https://medium.com/@michael.r.klear?source=post_page---post_responses--b8172278050f----0-----------------------------------)

[

Michael Klear



](https://medium.com/@michael.r.klear?source=post_page---post_responses--b8172278050f----0-----------------------------------)

[

May 9, 2019

](https://medium.com/@michael.r.klear/this-is-a-great-blog-post-but-i-was-still-confused-after-reading-it-21013c74add5?source=post_page---post_responses--b8172278050f----0-----------------------------------)

This is a great blog post, but I was still confused after reading it. [This talk by a Hyperopt developer](https://www.youtube.com/watch?v=tdwgR1AqQ8Y) really helped clarify the TPE algorithm, and after that understanding I came back to this article and found it much more helpful.

5

Reply

[

![Shaina Race](https://miro.medium.com/v2/resize:fill:64:64/0*M0uPykgpMfz5g_UL.jpg)





](https://medium.com/@shainarace?source=post_page---post_responses--b8172278050f----1-----------------------------------)

[

Shaina Race



](https://medium.com/@shainarace?source=post_page---post_responses--b8172278050f----1-----------------------------------)

[

Jan 9, 2019

](https://medium.com/@shainarace/in-your-jupyter-notebook-you-have-the-following-output-which-appears-to-be-correctly-specified-fcb80d4d36a?source=post_page---post_responses--b8172278050f----1-----------------------------------)

In your Jupyter notebook, you have the following output (which appears to be correctly specified from your given output):  
Number of trials needed to attain minimum with TPE: 655  
Number of trials needed to attain minimum with random: 235

Then, you…more

5

Reply

[

![Zecheng Zhang](https://miro.medium.com/v2/resize:fill:64:64/1*sBD5O73oBVdgBxcpOwWsRg.png)





](https://medium.com/@tom_z_z?source=post_page---post_responses--b8172278050f----2-----------------------------------)

[

Zecheng Zhang



](https://medium.com/@tom_z_z?source=post_page---post_responses--b8172278050f----2-----------------------------------)

[

Apr 7, 2019

](https://medium.com/@tom_z_z/think-about-what-is-the-score-this-should-be-the-object-function-or-the-one-we-need-to-minimize-b21db0139620?source=post_page---post_responses--b8172278050f----2-----------------------------------)

this model is called a “surrogate” for the objective function and is represented as p(y | x).

Think about what is the score? This should be the object function or the one we need to minimize.

4

Reply

See all responses

## More from Will Koehrsen and TDS Archive

![Hyperparameter Tuning the Random Forest in Python](https://miro.medium.com/v2/resize:fit:1273/1*mTBEiGR_W-cYMw8cIWQh0w.jpeg)

[

![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)



](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----0---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

In

[

TDS Archive

](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----0---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

by

[

Will Koehrsen

](https://medium.com/@williamkoehrsen?source=post_page---author_recirc--b8172278050f----0---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

## Hyperparameter Tuning the Random Forest in Python

### Improving the Random Forest Part Two



](https://medium.com/data-science/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74?source=post_page---author_recirc--b8172278050f----0---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

Jan 9, 2018

[

A clap icon8.2K

A response icon45







](https://medium.com/data-science/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74?source=post_page---author_recirc--b8172278050f----0---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

![Building Knowledge Graphs with LLM Graph Transformer](https://miro.medium.com/v2/resize:fit:1273/0*KHns0-DJoCjfzxyr)

[

![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)



](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----1---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

In

[

TDS Archive

](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----1---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

by

[

Tomaz Bratanic

](https://medium.com/@bratanic-tomaz?source=post_page---author_recirc--b8172278050f----1---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

## Building Knowledge Graphs with LLM Graph Transformer

### A deep dive into LangChain’s implementation of graph construction with LLMs



](https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59?source=post_page---author_recirc--b8172278050f----1---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

Nov 5, 2024

[

A clap icon1.2K

A response icon16







](https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59?source=post_page---author_recirc--b8172278050f----1---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

![10 Common Software Architectural Patterns in a nutshell](https://miro.medium.com/v2/resize:fit:1273/1*M22DR3WPqbWXWidYIq2GwA.png)

[

![TDS Archive](https://miro.medium.com/v2/resize:fill:38:38/1*JEuS4KBdakUcjg9sC7Wo4A.png)



](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----2---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

In

[

TDS Archive

](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f----2---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

by

[

Vijini Mallawaarachchi

](https://medium.com/@vijini?source=post_page---author_recirc--b8172278050f----2---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

## 10 Common Software Architectural Patterns in a nutshell

### Ever wondered how large enterprise scale systems are designed? Before major software development starts, we have to choose a suitable…



](https://medium.com/data-science/10-common-software-architectural-patterns-in-a-nutshell-a0b47a1e9013?source=post_page---author_recirc--b8172278050f----2---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

Sep 4, 2017

[

A clap icon41K

A response icon143







](https://medium.com/data-science/10-common-software-architectural-patterns-in-a-nutshell-a0b47a1e9013?source=post_page---author_recirc--b8172278050f----2---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

![Random Forest Simple Explanation](https://miro.medium.com/v2/resize:fit:1273/1*i0o8mjFfCn-uD79-F1Cqkw.png)

[

![Will Koehrsen](https://miro.medium.com/v2/resize:fill:38:38/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)



](https://medium.com/@williamkoehrsen?source=post_page---author_recirc--b8172278050f----3---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

Will Koehrsen

](https://medium.com/@williamkoehrsen?source=post_page---author_recirc--b8172278050f----3---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

## Random Forest Simple Explanation

### Understanding the Random Forest with an intuitive example



](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d?source=post_page---author_recirc--b8172278050f----3---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

Dec 27, 2017

[

A clap icon5.9K

A response icon23







](https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d?source=post_page---author_recirc--b8172278050f----3---------------------4697a709_f127_4ad5_b7d1_838de325ba07--------------)

[

See all from Will Koehrsen

](https://medium.com/@williamkoehrsen?source=post_page---author_recirc--b8172278050f---------------------------------------)

[

See all from TDS Archive

](https://medium.com/data-science?source=post_page---author_recirc--b8172278050f---------------------------------------)

## Recommended from Medium

![Retrieval Augmented Generation (RAG) 06 :- BM25 Retriever: When and Why to Use It (With Code Demo)?](https://miro.medium.com/v2/resize:fit:1273/0*gJkDBw418lGjdwkW)

[

![Yashwanth S](https://miro.medium.com/v2/resize:fill:38:38/0*M8MEhkHvbOQn8O2W)



](https://medium.com/@yashwanths_29644?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

Yashwanth S

](https://medium.com/@yashwanths_29644?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## Retrieval Augmented Generation (RAG) 06 :- BM25 Retriever: When and Why to Use It (With Code Demo)?

### Retrieval is the first and most crucial step in many Natural Language Processing (NLP) applications like search engines, chatbots, and…



](https://medium.com/@yashwanths_29644/retrieval-augmented-generation-rag-06-bm25-retriever-when-and-why-to-use-it-with-code-demo-132ed70c6bfd?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

May 28

[

A clap icon5







](https://medium.com/@yashwanths_29644/retrieval-augmented-generation-rag-06-bm25-retriever-when-and-why-to-use-it-with-code-demo-132ed70c6bfd?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

![10 ML Algorithms Every Data Scientist Should Know — Part 1](https://miro.medium.com/v2/resize:fit:1273/1*lPuRae1B1MhOHoFeszuZvw.jpeg)

[

![Learning Data](https://miro.medium.com/v2/resize:fill:38:38/1*2h_G6zLH23eg9t6lkN_sZg.jpeg)



](https://medium.com/learning-data?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

In

[

Learning Data

](https://medium.com/learning-data?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

by

[

Rita Angelou

](https://medium.com/@ritaaggelou?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## 10 ML Algorithms Every Data Scientist Should Know — Part 1

### I understand well that machine learning might sound intimidating. But once you break down the common algorithms, you’ll see they’re not.



](https://medium.com/learning-data/10-ml-algorithms-every-data-scientist-should-know-part-1-2deced7f325f?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

Jun 10

[

A clap icon27







](https://medium.com/learning-data/10-ml-algorithms-every-data-scientist-should-know-part-1-2deced7f325f?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

![9 Things You Discover Only After Building Your First Real ML Project](https://miro.medium.com/v2/resize:fit:1273/0*zmnYCBHivu0sVAQA)

[

![AI-ML Interview Playbook](https://miro.medium.com/v2/resize:fill:38:38/1*iDF-dnJJBq0kFigJ-jkdUA.png)



](https://medium.com/ai-ml-interview-playbook?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

In

[

AI-ML Interview Playbook

](https://medium.com/ai-ml-interview-playbook?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

by

[

Sajid Khan

](https://medium.com/@sajidkhan.sjic?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## 9 Things You Discover Only After Building Your First Real ML Project

### It’s not in the course. It’s not in the docs. But it hits you fast



](https://medium.com/ai-ml-interview-playbook/9-things-you-discover-only-after-building-your-first-real-ml-project-3a4ed73d062f?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

Jul 7

[

A clap icon15







](https://medium.com/ai-ml-interview-playbook/9-things-you-discover-only-after-building-your-first-real-ml-project-3a4ed73d062f?source=post_page---read_next_recirc--b8172278050f----0---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

![The 47-Line Code That Made One Developer $2 Million from AI](https://miro.medium.com/v2/resize:fit:1273/1*F1HE6GGa40dhfv1ZZkwToA.png)

[

![Yash Batra](https://miro.medium.com/v2/resize:fill:38:38/1*qzg4k1dbmci1Z47Qm6BTsQ@2x.jpeg)



](https://medium.com/@yashbatra11111?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

Yash Batra

](https://medium.com/@yashbatra11111?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## The 47-Line Code That Made One Developer $2 Million from AI

### In late 2024, a solo indie developer pushed 47 lines of Python code to GitHub.



](https://medium.com/@yashbatra11111/the-47-line-code-that-made-one-developer-2-million-from-ai-7269383d65db?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

Jul 8

[

A clap icon1.8K

A response icon57







](https://medium.com/@yashbatra11111/the-47-line-code-that-made-one-developer-2-million-from-ai-7269383d65db?source=post_page---read_next_recirc--b8172278050f----1---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

![The Python Feature That’s 10x Faster Than Loops (But Only 3% of Developers Know It)](https://miro.medium.com/v2/resize:fit:1273/0*IGaaMDW1shirhQVD.png)

[

![Sohail Saifi](https://miro.medium.com/v2/resize:fill:38:38/1*3SigoC_uqb4zv03teeR7OQ.jpeg)



](https://medium.com/@sohail_saifi?source=post_page---read_next_recirc--b8172278050f----2---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

Sohail Saifi

](https://medium.com/@sohail_saifi?source=post_page---read_next_recirc--b8172278050f----2---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## The Python Feature That’s 10x Faster Than Loops (But Only 3% of Developers Know It)

### For years, you’ve been writing Python. You know what list comprehensions are, you’ve heard of NumPy, and you probably think you’re pretty…



](https://medium.com/@sohail_saifi/the-python-feature-thats-10x-faster-than-loops-but-only-3-of-developers-know-it-f8580aece8be?source=post_page---read_next_recirc--b8172278050f----2---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

Jul 27

[

A clap icon705

A response icon12







](https://medium.com/@sohail_saifi/the-python-feature-thats-10x-faster-than-loops-but-only-3-of-developers-know-it-f8580aece8be?source=post_page---read_next_recirc--b8172278050f----2---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

![Master Hyperparameter Optimization with Optuna: A Complete Guide [Part 1]](https://miro.medium.com/v2/resize:fit:1273/0*oSBuccIP06H40Wxq)

[

![Mihir Shah](https://miro.medium.com/v2/resize:fill:38:38/1*5PIOPZ1FDWSsVB5r1rKXjA.png)



](https://medium.com/@mdshah930?source=post_page---read_next_recirc--b8172278050f----3---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

Mihir Shah

](https://medium.com/@mdshah930?source=post_page---read_next_recirc--b8172278050f----3---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

## Master Hyperparameter Optimization with Optuna: A Complete Guide \[Part 1\]

### In the world of machine learning, one of the most challenging tasks is finding the optimal set of hyperparameters for your model. Unlike…



](https://medium.com/@mdshah930/master-hyperparameter-optimization-with-optuna-a-complete-guide-89971b799b0a?source=post_page---read_next_recirc--b8172278050f----3---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

Feb 22

[

A clap icon20

A response icon1







](https://medium.com/@mdshah930/master-hyperparameter-optimization-with-optuna-a-complete-guide-89971b799b0a?source=post_page---read_next_recirc--b8172278050f----3---------------------34befc49_c519_470d_9fff_15f940ac77b5--------------)

[

See more recommendations

](https://medium.com/?source=post_page---read_next_recirc--b8172278050f---------------------------------------)