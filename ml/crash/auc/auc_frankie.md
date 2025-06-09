# auc curve

For perfect auc curve: get area of 1
1. vertical axis is tp/np 
2. horizontal axis is tn/nn

Eventually the curve has to read tp/np, tn/nn = 1,1

The way we plot the curve is:
1. rank all scores (predictions) 
1. then as we step through ranks
   1. accumulate number of true labels and false labels
   1. that is for each data point:
      1. corresponding to a particular threshold
      1. we can find out tp/np and tn/nn so far

If all the positive true labels come before negative true labels, then as we step through the points, all the true
positives will have to be exhausted, thus we reach (0,1).

Note:

1. any horizontal line segments on the AUC curve means we are accumulating negative samples
1. any vertical line segments on the AUC curve means we are accumulating positive samples

Calculating the AUC score:
1. rank by the prediciton scores (higher number means higher rank)
1. AUC = U1/(total # pairs of pos and neg samples)
1. total # pairs: np * nn
1. U1 score number of pairs where positive > negative
1. if pos are always ranked higher neg then U1 = np*nn
1. otherwise calculate U1 = \sum Rpos - n(n-1)/2
   1. Each rank says how many elements are below me
      1. or how many pairs are below me
   1. Since we only care about negative that are below
      positives, we subtract by the number of ways a
      positive can be below a positive n(n-1)/2

1.           \sum Rpos - n(n-1)/2
      AUC =  --------------------
                   np * nn
