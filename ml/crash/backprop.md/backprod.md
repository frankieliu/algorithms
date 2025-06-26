[stack](https://math.stackexchange.com/posts/4821055/timeline)

$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
$

The formula which you derived is correct.
Here is another way to calculate it...

Assume that the gradient of a scalar cost function $
{\cal L}$ with respect to the matrix $Y$ is known.

<br>Expand the *differential* of ${\cal L},$ and change the independent variable from $\,Y\to W$ 
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\eqalign{
Y &= XW^T + B \qiq &dY = X\:dW^T \\
G &= \grad{\L}{Y} &\{{\rm known\ gradient}\} \\
d\L &= G:dY  &\{{\rm differential}\} \\
 &= G:\LR{X\,dW^T} \\
 &= G^T:\LR{dW\,X^T} \\
 &= \LR{G^TX}:dW \\
\grad{\L}{W}
 &= {G^TX} &\{{\rm new\ gradient}\}  \\
}$$
where $(:)$ denotes the Frobenius product, which is a concise
notation for the trace 
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\eqalign{
A:B &= \sum_{i=1}^m\sum_{j=1}^n A_{ij}B_{ij} \;=\; \trace{A^TB} \\
A:A &= \frob{A}^2 \qquad \{ {\rm Frobenius\;norm} \}\\
A:B &= B:A \;=\; B^T:A^T \\
\LR{AB}:C &= A:\LR{CB^T} \;=\; B:\LR{A^TC} \\
}$$
The advantage of using differentials is to avoid the need for any awkward fourth-order tensors.

Update
---
I misread the question. Here is the derivation for the tensor-valued gradient.

Define the following tensors
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\def\d{\delta}  \def\o{{\tt1}}
\def\H{{\large\cal H}} \def\F{{\cal F}}
\eqalign{
&\F_{ijkl} = \d_{il}\,\d_{jk} \\
&\H_{ij\,kl\,mn} = \begin{cases}
\o \qquad {\rm if}\;i=k=m\;\;{\rm and}\;\;j=l=n \\
0 \qquad {\rm otherwise}
\end{cases}
}$$
and the matrix variables
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\eqalign{
L = h(Y), \qquad L' = h'(Y) \\
}$$
where $h'$ is the ordinary (scalar) derivative of the $h$ function and is applied elementwise.

Expand the differential of the matrix-valued function
and change $Y\to W$ once again
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\def\d{\delta}  \def\o{{\tt1}}
\def\H{{\large\cal H}} \def\F{{\cal F}}
\eqalign{
dL
 &= L'\odot dY \\
 &= L':\H:dY \qquad\qiq \grad LY = L':\H \\
 &= L':\H:\LR{X\,dW^T} \\
 &= L':\H:\LR{X\cdot\F}:dW \\
\grad LW &= L':\H:\LR{X\cdot\F} \\
\\
\grad{L_{kl}}{W_{pq}}
 &= L'_{ij}\:\H_{ijklmn}\:{X_{ms}\F_{snpq}} \\
 &= h'(Y_{ij})\;\H_{ijklmn}\:{X_{mq}\,\d_{np}} \\
}$$
In the above,
$(\cdot)$ is the single-contraction product,
$(:)$ is the double-contraction product,
$(\odot)$ is the elementwise/Hadamard product,
and the index expression employs the [Einstein summation][1] convention. 

$\sf NB\!:\:$ The Hadamard tensor $
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\def\d{\delta}  \def\o{{\tt1}}
\def\H{{\large\cal H}} \def\F{{\cal F}}
\H$ is a sixth-order tensor defined such that
$$
\gdef\qiq{\quad\implies\quad}
\gdef\L{{\cal L}}
\gdef\LR#1{\left(#1\right)}
\xdef\op#1{\operatorname{#1}}
\xdef\trace#1{\op{Tr}\LR{#1}}
\gdef\frob#1{\left\| #1 \right\|_F}
\gdef\p{\partial}
\xdef\grad#1#2{\frac{\p #1}{\p #2}}
\gdef\eqalign#1{\begin{align}#1\end{align}}
\def\d{\delta}  \def\o{{\tt1}}
\def\H{{\large\cal H}} \def\F{{\cal F}}
 A\odot B = A:\H:B \qquad \qquad \qquad \qquad \quad $$
$\qquad$ for any two matrices $\{A,B\}$ which have identical dimensions.

1: https://mathworld.wolfram.com/EinsteinSummation.html