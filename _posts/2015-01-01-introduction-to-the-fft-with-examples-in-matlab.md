---
layout: post
title: Introduction to the FFT
date: 2015-01-01 12:00:00-0800
description: A tutorial on the theory of the discrete and fast Fourier transforms with examples in MATLAB.
---

## Introduction

Fast Fourier transform (FFT) refers to any one of a family of algorithms that can compute the discrete Fourier transform (DFT) of a signal with *n* elements in *O*(*n* lg *n*) [FLOPs](https://en.wikipedia.org/wiki/FLOPS). The DFT is a transformation that converts a signal from its original domain (e.g., time or space) into the [frequeny domain](https://en.wikipedia.org/wiki/Frequency_domain).

This article introduces the theory of the DFT and FFT and gives some examples in MATLAB.

## Discrete Fourier transform

### Roots of unity

**Definition** (Roots of unity)**.** Let *n* be a positive integer. A complex number *z* is said to be an *n*-th root of unity if and only if *z<sup>n</sup>* = 1.

**Corollary.** If *z* is an *n*-th root of unity, so too is *z<sup>w</sup>* for any complex number *w*.

*Proof*. (*z<sup>w</sup>*)<sup>*n*</sup> = (*z<sup>n</sup>*)<sup>*w*</sup> = 1<sup>*w*</sup> = 1. ∎

**Corollary.** Let *n* be a positive integer and 𝜔<sub>*n*</sub> = *e*<sup>-2𝜋*i*/*n*</sup>. Then, 𝜔<sub>*0*</sub>, ..., 𝜔<sub>*n*-1</sub> are the only *n*-th roots of unity.

*Proof*. It's easily verified that 𝜔<sub>*0*</sub>, ..., 𝜔<sub>*n*-1</sub> define *n* distinct *n*-th roots of unity. Uniqueness follows from the fundamental theorem of algebra applied to the polynomial *z<sup>n</sup>* - 1. ∎

**Definition** (Kronecker delta)**.** The Kronecker delta 𝛿<sub>*jj'*</sub> is defined to be 1 if and only if *j* = *j'* and 0 otherwise.

**Lemma.** Let *n* be a positive integer and *z* ≠ 1 be an *n*-th root of unity. Then, *z*<sup>0</sup> + ... + *z*<sup>*n*-1</sup>  = 0.

*Proof*. The sum in question is a geometric series, with closed form (1 - *z<sup>n</sup>*) / (1 - *z*). Since *z* is an *n*-th root of unity, this is identically zero. ∎

**Corollary.** Let *n* be a positive integer and *j* and *j'* be nonnegative integers  strictly smaller than *n*. Then,

$$\sum_{k=0}^{n-1} \omega_n^{(j-j^\prime) k} = n \delta_{jj^\prime}.$$

*Proof*. If *j* = *j'*, each summand is 1. Otherwise, $\omega_n^{(j-j^\prime)}$ is an *n*-th root of unity and the desired result follows from the previous lemma. ∎

### Forward and inverse transforms

**Definition.** The conjugate transpose of a complex number *z* = *a* + *i* *b* is *z<sup>\*</sup>* = a - i b. The conjugate transpose ***z***<sup>\*</sup> of a complex vector (or matrix) ***z*** is obtained by taking the transpose of the vector (or matrix) and conjugating each entry.

**Theorem.** Let

$$\boldsymbol{u}^{(j)} = \frac{1}{\sqrt{n}} (\omega_n^0, \omega_n^j, \omega_n^{2j}, \ldots, \omega_n^{(n-1)j}).$$

Then, the vectors ***u***<sup>(0)</sup>, ..., ***u***<sup>(*n*-1)</sup> form an orthonormal basis for ℂ<sup>*n*</sup>.

*Proof*. Using the previous corollary,

$$n \left \langle u^{(j)}, u^{(j^\prime)} \right \rangle = \sum_{k=0}^{n-1} \omega_n^{-j^\prime k} \omega_n^{j k} = \sum_{k=0}^{n-1} \omega_n^{(j-j^\prime) k} = n \delta_{jj^\prime}.$$ ∎

The above establishes that the matrix *F*

![](/assets/img/introduction-to-the-fft-with-examples-in-matlab/forward.gif)

(whose rows are the basis elements in the theorem) is [unitary](https://en.wikipedia.org/wiki/Unitary_matrix). The matrix *F* is called the *forward transform*. Because this matrix is unitary, its conjugate transpose *F*<sup>\*</sup> is its inverse (i.e., *F<sup>\*</sup>F* = *I*). As such, we call the conjugate transpose the *inverse transform*.

*Remark*. In the above, we used the scaling factor 1/√*n* to ensure that the matrix *F* was unitary, simplifying mathematical discussion. MATLAB's definitions of forward and inverse transforms `fft` and `ifft` do not use the same scaling factor (they are *F√n* and *F<sup>\*</sup>/√n*, respectively) so as to avoid the cost of an extra vector-scalar multiply in the forward transform. Software libraries have their own conventions when it comes to the scaling factors for forward and inverse transforms, so it's best to proceed carefully!

### Some immediate results

**Theorem** (Plancherel theorem)**.** Let ***x*** and ***y*** be vectors in ℂ<sup>*n*</sup> and denote by ***X*** = *F* ***x*** and ***Y*** = *F* ***y*** their forward transforms. Then, ⟨***x***, ***y***⟩ = ⟨***X***, ***Y***⟩.

*Proof*. ⟨***X***, ***Y***⟩ = ⟨*F* ***x***, *F* ***y***⟩ = ***y***<sup>\*</sup> *F*<sup>\*</sup> *F* ***x*** = ***y***<sup>\*</sup> ***x*** = ⟨***x***, ***y***⟩. ∎

**Corollary** (Parseval's theorem)**.** Let ***x*** be a vector in ℂ<sup>*n*</sup> and denote by ***X*** = *F* ***x*** its forward transform. Then, ⟨***x***, ***x***⟩ = ⟨***X***, ***X***⟩ (i.e., the original vector and its transform have the same Euclidean norm).

### Convolution

For integers *a* and *b* with *b* nonnegative, denote by *a % b* the least nonnegative residue of *a* modulo *b* (i.e., the usual definition of the `%` operator provided by most programming languages). For a vector ***x*** with *n* entries, we introduce the indexing convention *x<sub>j</sub>* = *x*<sub>*j* % *n*</sub>  whenever *j* is negative or at least as large as *n*. Subject to this convention, we have the following results:

**Definition** (Circular convolution)**.** The circular convolution of vectors ***x*** and ***y*** in ℂ<sup>*n*</sup> is a vector ***x*** ✳︎ ***y*** with entries [***x*** ✳︎ ***y***]<sub>*j*</sub> = *x*<sub>0</sub>y<sub>*j*-0</sub> + ... + *x*<sub>*n*-1</sub>y<sub>*j*-(*n*-1)</sub>.

**Theorem** (Convolution theorem)**.** Let ⨂ denote the [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)), ***x*** and ***y*** be vectors in ℂ<sup>*n*</sup>. Then, *F*(***x*** ✳︎ ***y***) = *F* ***x*** ⨂ *F* ***y*** and *F* (***x*** ⨂ ***y***) = *F* ***x*** ✳︎ *F* ***y***.

The above theorem tells us that, for example, the convolution of vectors can be computed by

1. taking their discrete Fourier transforms ***X*** and ***Y***
2. multiplying these element-wise to get ***Z*** = ***X*** ⨂ ***Y***, and
3. taking the inverse Fourier transform of ***Z***.

In a subsequent section, we will prove that the discrete Fourier transform can be computed using only *O*(*n* lg *n*) FLOPs, suggesting that the above procedure is superior to naively computing the circular convolution from the formula, which requires *O*(*n*<sup>2</sup>) FLOPs.

Similar results can be established for the [circular cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation).

### Real signals

In practice, the input to the forward transform is often a real signal (i.e., a vector in ℝ<sup>*n*</sup>). It is useful to derive some facts about such signals.

**Theorem** (Conjugate symmetry)**.** Let ***x*** be a vector in ℝ<sup>*n*</sup> and ***X*** = *F* ***x***. Then, (*X<sub>k</sub>*)<sup>\*</sup> = *X*<sub>*n*-*k*</sub>.

*Proof*. First, note that ***X***<sup>\*</sup> = (*F* ***x***)<sup>\*</sup> = ***x***<sup>⊺</sup> *F*<sup>\*</sup> = ((*F*<sup>\*</sup>)<sup>⊺</sup> ***x***)<sup>⊺</sup> where ⊺ denotes the ordinary transpose operation. Now, using the definition of *F* and the conjugate transpose operation, it is straightforward to establish that

![](/assets/img/introduction-to-the-fft-with-examples-in-matlab/inverse.gif)

from which the desired result follows.

**Theorem.** Let ***x*** and ***y*** be vectors in ℝ<sup>*n*</sup>  and ***z*** = ***x*** + *i* ***y***. Further let ***X***, ***Y***, and ***Z*** be their corresponding forward transforms. Then, 2*X<sub>k</sub>* = *Z<sub>k</sub>* + (*Z*<sub>*n*-*k*</sub>)<sup>\*</sup> and 2*Y<sub>k</sub>* = *Z<sub>k</sub>* - (*Z*<sub>*n*-*k*</sub>)<sup>\*</sup>.

In other words, the DFT of two real signals ***x*** and ***y*** can be computed by

1. creating a complex signal ***z*** whose real part is ***x*** and whose imaginary part is ***y***,
2. computing the DFT of ***z***, and
3. retrieving the DFTs of ***x*** and ***y*** using the above formulas.

The proof is left as an exercise. MATLAB code for this procedure is provided below:

```matlab
function [X, Y] = fft_real_signals (x, y)
% FFT_REAL_SIGNALS Computes the DFT of two real signals using one FFT.

z = x + 1.i * y;
Z = fft (z);

% Create Zr_conj, whose (j+1)-th component (MATLAB indices start at 1)
% is the conjugate of the (n-j+1)-th component of Z.
Zr_conj = conj ([Z(1) fliplr (Z(2:end))]);

X = (Z + Zr_conj) / 2.;
Y = (Z - Zr_conj) / 2.;

end
```



## Fast Fourier transform

The [Cooley-Tukey algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm) is the most common FFT algorithm. It re-expresses the DFT of an arbitrary composite size *n* = *ab* in terms of smaller DFTs of sizes *a* and *b*.

### Radix-2 algorithm

Assuming *n* is even, the radix-2 algorithm corresponds to taking *a* = *b* = *n* / 2. The motivation comes from noting that

$$X_k = \sum_{j=0}^{n-1} x_j \omega_n^{jk} = \sum_{j = 0}^{n/2 - 1} x_{2j} \omega_n^{2jk} + \omega_n^k \sum_{j = 0}^{n/2-1} x_{2j+1} \omega_n^{2jk}$$

and hence

$$X_k = E_k + \omega_n^k O_k$$

where ***E*** and ***O*** are the DFTs of the even and odd parts of the original signal ***x***, respectively.

By the periodicity of the DFT and

$$\omega_n^{k+n/2} = e^{-i \pi} \omega_n^k = -\omega_n^k$$

the above can be equivalently expressed as the two equations

$$X_k = E_k + \omega_n^k O_k \qquad \text{and} \qquad X_{k+n/2} = E_k - \omega_n^k O_k$$

for nonnegative integers *k* strictly less than *n*/2.

## Image compression (with MATLAB code)

Image compression is one possible application of the FFT. It can be achieved by

1. transforming the image into the frequency domain,
2. dropping high-frequency components, and
3. saving the result.

To view the image, one must inverse transform back to the original domain.

```matlab
function [dft_R, dft_G, dft_B] = compress_image (image, ratio)
% COMPRESS_IMAGE Compresses an image.
%
% Inputs
% ------
% image    The original image.
% ratio    The compression ratio in (0, 1].
%
% Outputs
% -------
% dft_R    The DFT of the red channel.
% dft_G    The DFT of the green channel.
% dft_B    The DFT of the blue channel.

% Normalize.
image_d = double (image) / double (max (max (max (image))));

% FFT
fft_image_d = fft2 (image_d);
dft_R = sparse (fft_image_d(:, :, 1));
dft_G = sparse (fft_image_d(:, :, 2));
dft_B = sparse (fft_image_d(:, :, 3));

% Size of image.
[m, n, ~] = size (image);
p = ceil (m / 2);
q = ceil (n / 2);

% Number of frequencies to remove first and second dimensions.
f  = (1. - sqrt (ratio)) / 2;
dm = round (f * m);
dn = round (f * n);

% The new image will only have r * m * n frequencies.
i = p - dm + 1 : p + dm;
j = q - dn + 1 : q + dn;
dft_R(i, :) = 0;
dft_R(:, j) = 0;
dft_G(i, :) = 0;
dft_G(:, j) = 0;
dft_B(i, :) = 0;
dft_B(:, j) = 0;

end
```

```matlab
function [image] = decompress_image (dft_R, dft_G, dft_B)
% DECOMPRESS_IMAGE Returns a full representation of the image.
%
% Inputs
% ------
% dft_R    The DFT of the red channel.
% dft_G    The DFT of the green channel.
% dft_B    The DFT of the blue channel.
%
% Outputs
% -------
% image    The image.

[m, n] = size (dft_R);
image = zeros (m, n, 3);
image(:, :, 1) = real (ifft2 (full (dft_R)));
image(:, :, 2) = real (ifft2 (full (dft_G)));
image(:, :, 3) = real (ifft2 (full (dft_B)));

end
```

```matlab
% Code used to create the example.
earth = imread ('earth.jpg')
ratio = 0.15;
[dft_R, dft_G, dft_B] = compress_image (earth, ratio);
imshow (decompress_image (dft_R, dft_G, dft_B));
```

#### Original image

![Uncompressed image](/assets/img/introduction-to-the-fft-with-examples-in-matlab/earth.png)

#### Compressed image (15% data retention)

![Compressed image](/assets/img/introduction-to-the-fft-with-examples-in-matlab/earth15.png)
