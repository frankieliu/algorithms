Toolformer (Meta, 2023)
Gorilla (Berkeley, 2023)
Voyager (NVIDIA, 2023)
Self-Discover (Stanford, 2024)
Devin (Cognition Labs, 2024) – First AI software enginee1

This answer is not useful

Save this answer.

[](https://math.stackexchange.com/posts/4821055/timeline)

Show activity on this post.

 The formula which you derived is correct. Here is another way to calculate it...

Assume that the gradient of a scalar cost function L$\mathcal{L}$ with respect to the matrix Y$Y$ is known.  
Expand the _differential_ of L$\mathcal{L}$ and change the independent variable from Y→W$Y \rightarrow W$

YGdL∂L∂W\=XWT+B⟹\=∂L∂Y\=G:dY\=G:(XdWT)\=GT:(dWXT)\=(GTX):dW\=GTXdY\=XdWT{known gradient}{differential}{new gradient}$$Y & = X W^{T} + B \Longrightarrow & d Y = X d W^{T} \\ G & = \frac{\partial \mathcal{L}}{\partial Y} & \left{\right. k n o w n \&\text{nbsp}; g r a d i e n t \left.\right} \\ d \mathcal{L} & = G : d Y & \left{\right. d i f f e r e n t i a l \left.\right} \\ & = G : \left(\right. X d W^{T} \left.\right) \\ & = G^{T} : \left(\right. d W X^{T} \left.\right) \\ & = \left(\right. G^{T} X \left.\right) : d W \\ \frac{\partial \mathcal{L}}{\partial W} & = G^{T} X & \left{\right. n e w \&\text{nbsp}; g r a d i e n t \left.\right}$$

 where (:)$\left(\right. : \left.\right)$ denotes the Frobenius product, which is a concise notation for the trace

A:BA:AA:B(AB):C\=∑i\=1m∑j\=1nAijBij\=Tr(ATB)\=∥A∥2F{Frobeniusnorm}\=B:A\=BT:AT\=A:(CBT)\=B:(ATC)$$A : B & = \sum_{i = 1}^{m} \sum_{j = 1}^{n} A_{i j} B_{i j} = Tr ⁡ \left(\right. A^{T} B \left.\right) \\ A : A & = \left(\parallel A \parallel\right)_{F}^{2} \left{\right. F r o b e n i u s n o r m \left.\right} \\ A : B & = B : A = B^{T} : A^{T} \\ \left(\right. A B \left.\right) : C & = A : \left(\right. C B^{T} \left.\right) = B : \left(\right. A^{T} C \left.\right)$$

 The advantage of using differentials is to avoid the need for any awkward fourth-order tensors.

## Update

I misread the question. Here is the derivation for the tensor-valued gradient.

Define the following tensors

Fijkl\=δilδjkHijklmn\={1ifi\=k\=mandj\=l\=n0otherwise$$& \mathcal{F}_{i j k l} = \delta_{i l} \delta_{j k} \\ & \mathcal{H}_{i j k l m n} = \begin{cases} 1 i f i = k = m a n d j = l = n \\ 0 o t h e r w i s e \end{cases}$$

 and the matrix variables

L\=h(Y),L′\=h′(Y)$$L = h \left(\right. Y \left.\right) , L^{'} = h^{'} \left(\right. Y \left.\right)$$

 where h′$h^{'}$ is the ordinary (scalar) derivative of the h$h$ function and is applied elementwise.

Expand the differential of the matrix-valued function and change Y→W$Y \rightarrow W$ once again

dL∂L∂W∂Lkl∂Wpq\=L′⊙dY\=L′:H:dY⟹∂L∂Y\=L′:H\=L′:H:(XdWT)\=L′:H:(X⋅F):dW\=L′:H:(X⋅F)\=L′ijHijklmnXmsFsnpq\=h′(Yij)HijklmnXmqδnp$$d L & = L^{'} \bigodot d Y \\ & = L^{'} : \mathcal{H} : d Y \Longrightarrow \frac{\partial L}{\partial Y} = L^{'} : \mathcal{H} \\ & = L^{'} : \mathcal{H} : \left(\right. X d W^{T} \left.\right) \\ & = L^{'} : \mathcal{H} : \left(\right. X \cdot \mathcal{F} \left.\right) : d W \\ \frac{\partial L}{\partial W} & = L^{'} : \mathcal{H} : \left(\right. X \cdot \mathcal{F} \left.\right) \\ \\ \frac{\partial L_{k l}}{\partial W_{p q}} & = L_{i j}^{'} \mathcal{H}_{i j k l m n} X_{m s} \mathcal{F}_{s n p q} \\ & = h^{'} \left(\right. Y_{i j} \left.\right) \mathcal{H}_{i j k l m n} X_{m q} \delta_{n p}$$

 In the above, (⋅)$\left(\right. \cdot \left.\right)$ is the single-contraction product, (:)$\left(\right. : \left.\right)$ is the double-contraction product, (⊙)$\left(\right. \bigodot \left.\right)$ is the elementwise/Hadamard product, and the index expression employs the [Einstein summation](https://mathworld.wolfram.com/EinsteinSummation.html) convention.

NB:$\mathsf{N} \mathsf{B} :$ The Hadamard tensor H$\mathcal{H}$ is a sixth-order tensor defined such that

A⊙B\=A:H:B$$A \bigodot B = A : \mathcal{H} : B$$

  for any two matrices {A,B}$\left{\right. A , B \left.\right}$ which have identical dimensions.r