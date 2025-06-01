# reverse accumulation

1. begin from the end and propagate the derivative backwards

1. $dy/dw_k = dy/dw_{k+1} \cdot dw_{k+1}/dw_k$

1. More generally
$\bar{w}_k = \sum_{j \in successors} \bar{w}_j \cdot dw_j/dw_k$

1. The flow of derivative goes from right to the left.

In the backward mode, you have to do an additional backward pass after the forward evaluation pass to calculate the derivative.  The advantage is that you can compute the derivate for each of the independent variables simultaneously.

This is not the case if you have multiple outputs and you want to figure out the derivative with respect to each of the output *separately*.  You would need to have a separate backward pass for each of the outputs.

# forward mode
In forward mode differentiation, the flow of derivatives is forward, i.e.

$dw_k/dx = \sum_{j \in predec} dw_k/dw_j \cdot dw_j/dw_x$

This flow forwards allows computation of the derivative in the forward direction, and get the derivative in the same forward pass.

The issue is that you need a separate forward pass for each independent variable.