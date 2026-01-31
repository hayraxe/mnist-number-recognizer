# Model definition

We define a multilayer perceptron (MLP) with architecture $784 \to 16 \to 16 \to 10$ as a composition of affine maps and elementwise nonlinearities.

---

## Input, preprocessing, and labels

Each input is a grayscale image of size $28 \times 28$ with raw pixel intensities
$$
\tilde{I}^{(k)} \in \{0,1,\dots,255\}^{28 \times 28}.
$$

We normalize to $[0,1]$:
$$
I^{(k)} = \frac{\tilde{I}^{(k)}}{255} \in [0,1]^{28 \times 28}.
$$

We flatten the image into a column vector using a fixed ordering:
$$
x^{(k)} = \mathrm{vec}\!\left(I^{(k)}\right) \in [0,1]^{784}.
$$

A mini-batch is stored column-wise:
$$
X = \begin{bmatrix} x^{(1)} & x^{(2)} & \cdots & x^{(m)} \end{bmatrix} \in \mathbb{R}^{784 \times m},
$$
where $m$ is the batch size and **columns are samples**. Define $A^{[0]} := X$.

Let the class index for sample $k$ be $c^{(k)} \in \{0,\dots,9\}$. The one-hot label vector $y^{(k)} \in \{0,1\}^{10}$ is defined by
$$
y^{(k)}_i =
\begin{cases}
1, & i = c^{(k)} \\
0, & \text{otherwise}
\end{cases}
\qquad i=0,\dots,9.
$$

Stacking labels column-wise gives
$$
Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \in \{0,1\}^{10 \times m}.
$$

---

## Parameters and functions

For each layer $\ell \in \{1,2,3\}$,
$$
W^{[\ell]} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}, \qquad b^{[\ell]} \in \mathbb{R}^{n_\ell \times 1}.
$$

For this network:
- $W^{[1]} \in \mathbb{R}^{16 \times 784}$, $b^{[1]} \in \mathbb{R}^{16 \times 1}$
- $W^{[2]} \in \mathbb{R}^{16 \times 16}$,  $b^{[2]} \in \mathbb{R}^{16 \times 1}$
- $W^{[3]} \in \mathbb{R}^{10 \times 16}$,  $b^{[3]} \in \mathbb{R}^{10 \times 1}$

Bias vectors are added to every column.

Hidden-layer activation function (ReLU):
$$
\phi(z) = \max(0,z), \qquad
\phi'(z) =
\begin{cases}
1, & z>0 \\
0, & z \le 0
\end{cases}.
$$

Softmax is applied **per column**. For a vector $z \in \mathbb{R}^{10}$,
$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}, \qquad i=0,\dots,9.
$$

---

## Forward propagation

Vectorized (mini-batch) forward pass:
$$
\begin{aligned}
Z^{[1]} &= W^{[1]}A^{[0]} + b^{[1]}, & A^{[1]} &= \phi\!\left(Z^{[1]}\right) \\
Z^{[2]} &= W^{[2]}A^{[1]} + b^{[2]}, & A^{[2]} &= \phi\!\left(Z^{[2]}\right) \\
Z^{[3]} &= W^{[3]}A^{[2]} + b^{[3]}, & A^{[3]} &= \sigma\!\left(Z^{[3]}\right)
\end{aligned}
$$

Elementwise softmax (column $k$):
$$
A^{[3]}_{i,k} = \frac{e^{Z^{[3]}_{i,k}}}{\sum_{j=0}^{9} e^{Z^{[3]}_{j,k}}}.
$$

Composition form:
$$
A^{[3]}
=
\sigma\!\left(
W^{[3]}\,\phi\!\left(
W^{[2]}\,\phi\!\left(W^{[1]}X + b^{[1]}\right) + b^{[2]}
\right) + b^{[3]}
\right).
$$

Single-sample forward pass (one column $x$ of $X$):
$$
\begin{aligned}
a^{[0]} &= x \\
z^{[1]} &= W^{[1]}a^{[0]} + b^{[1]}, & a^{[1]} &= \phi\!\left(z^{[1]}\right) \\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]}, & a^{[2]} &= \phi\!\left(z^{[2]}\right) \\
z^{[3]} &= W^{[3]}a^{[2]} + b^{[3]}, & a^{[3]} &= \sigma\!\left(z^{[3]}\right)
\end{aligned}
$$

---

## Loss function (cross-entropy)

Batch loss:
$$
\mathcal{L}(A^{[3]}, Y)
=
-\frac{1}{m}\sum_{k=1}^{m}\sum_{i=0}^{9}
Y_{i,k}\,\log\!\left(A^{[3]}_{i,k}\right).
$$

Single-sample loss:
$$
\mathcal{C}(y,a^{[3]})
=
-\sum_{i=0}^{9} y_i \log\!\left(a^{[3]}_i\right).
$$

---

## Backpropagation

Define the layerwise error as
$$
\delta^{[\ell]} := \frac{\partial \mathcal{L}}{\partial Z^{[\ell]}} \in \mathbb{R}^{n_\ell \times m}.
$$

### Delta recursion
Output layer (softmax + cross-entropy):
$$
\delta^{[3]} = A^{[3]} - Y.
$$

Let $\odot$ denote the Hadamard (elementwise) product.

Hidden layers:
$$
\delta^{[2]} = \left(\left(W^{[3]}\right)^T \delta^{[3]}\right) \odot \phi'\!\left(Z^{[2]}\right),
$$
$$
\delta^{[1]} = \left(\left(W^{[2]}\right)^T \delta^{[2]}\right) \odot \phi'\!\left(Z^{[1]}\right).
$$

### Gradients

$$
\frac{\partial \mathcal{L}}{\partial W^{[3]}} = \frac{1}{m}\,\delta^{[3]}\left(A^{[2]}\right)^T,
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[3]}} = \frac{1}{m}\sum_{k=1}^{m}\delta^{[3]}_{:,k},
$$
$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m}\,\delta^{[2]}\left(A^{[1]}\right)^T,
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m}\sum_{k=1}^{m}\delta^{[2]}_{:,k},
$$
$$
\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m}\,\delta^{[1]}\left(A^{[0]}\right)^T,
\qquad
\frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m}\sum_{k=1}^{m}\delta^{[1]}_{:,k}.
$$

---

## Backpropagation with the chain rule (single-sample)

By expanding the single forward pass $(x,y)$, the dependency chain opens up clearly:

$$
z^{[1]} = W^{[1]}x + b^{[1]}, \quad a^{[1]}=\phi(z^{[1]}),
$$
$$
z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}, \quad a^{[2]}=\phi(z^{[2]}),
$$
$$
z^{[3]} = W^{[3]}a^{[2]} + b^{[3]}, \quad a^{[3]}=\sigma(z^{[3]}),
$$
and the loss is $L=\mathcal{C}(y,a^{[3]})$.

To get the gradient w.r.t. $W^{[1]}$, apply the chain rule from the outside back to the first layer:

$$
\frac{\partial L}{\partial W^{[1]}}
=
\frac{\partial L}{\partial z^{[3]}}
\cdot
\frac{\partial z^{[3]}}{\partial a^{[2]}}
\cdot
\frac{\partial a^{[2]}}{\partial z^{[2]}}
\cdot
\frac{\partial z^{[2]}}{\partial a^{[1]}}
\cdot
\frac{\partial a^{[1]}}{\partial z^{[1]}}
\cdot
\frac{\partial z^{[1]}}{\partial W^{[1]}}.
$$

By using the shorthand $\delta^{[\ell]} := \partial L/\partial z^{[\ell]}$, this turns into the same delta recursion as before:

$$
\delta^{[3]} = a^{[3]} - y,
\qquad
\delta^{[2]} = \left(\left(W^{[3]}\right)^T \delta^{[3]}\right)\odot \phi'(z^{[2]}),
\qquad
\delta^{[1]} = \left(\left(W^{[2]}\right)^T \delta^{[2]}\right)\odot \phi'(z^{[1]}).
$$

And once you have $\delta^{[1]}$, the gradient for the first weight matrix is just an outer product with the input:
$$
\frac{\partial L}{\partial W^{[1]}} = \delta^{[1]} x^T.
$$
