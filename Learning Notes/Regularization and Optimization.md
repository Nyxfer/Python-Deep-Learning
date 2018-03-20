# Regularization and Optimization

## The solution for bias and varance

**If bias:**

- Bigger network (deeper or bigger layer)
- Train longer
- Find neuro network architecture

**If variance:**

- More training data
- Regularization 
- Find neuro network architecture

## Regularization

1. **L2 regularization**

   Cost function:

   <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;J(w,&space;b)&space;=&space;\cfrac{1}{m}&space;\sum_{i&space;=&space;1}^{m}L(\widehat{y}^{(i)},&space;y^{(i)})&plus;\frac{\lambda}{2m}\left&space;\|&space;w&space;\right&space;\|&space;^{_{F}^{2}}&plus;&space;\frac{\lambda}{2m}b^{2}" title="J(w, b) = \cfrac{1}{m} \sum_{i = 1}^{m}L(\widehat{y}^{(i)}, y^{(i)})+\frac{\lambda}{2m}\left \| w \right \| ^{_{F}^{2}}+ \frac{\lambda}{2m}b^{2}" />

   ​		<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\frac{\lambda}{2m}b^{2}" title="\frac{\lambda}{2m}b^{2}" /> can use or omit

   Frobenius norm:

   <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\left&space;\|&space;w&space;\right&space;\|_{F}^{2}&space;=&space;\sum_{j&space;=&space;1}^{n_{x}}w_{j}^{2}&space;=&space;w^{T}w" title="\left \| w \right \|_{F}^{2} = \sum_{j = 1}^{n_{x}}w_{j}^{2} = w^{T}w" />

   Update:

   ​	<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\partial&space;w^{[l]}&space;=&space;(from&space;backprop)&space;&plus;&space;\frac{\lambda}{m}w^{[l]}" title="\partial w^{[l]} = (from backprop) + \frac{\lambda}{m}w^{[l]}" />

   ​	<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;w^{[l]}&space;-=&space;\alpha&space;\partial&space;w^{[l]}" title="w^{[l]} -= \alpha \partial w^{[l]}" />

   Weight decay:

   ​	<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;w^{[l]}&space;-&space;\frac{\alpha&space;\lambda}{m}w^{[l]}" title="w^{[l]} - \frac{\alpha \lambda}{m}w^{[l]}" />

   ​		--> <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;(1&space;-&space;\frac{\alpha&space;\lambda}{m})&space;w^{[l]}" title="(1 - \frac{\alpha \lambda}{m}) w^{[l]}" />

   *L1 regularization*

   ​	<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\frac{\lambda}{m}\sum_{k=1}^{n_x}\left&space;|&space;w&space;\right&space;|&space;=&space;\frac{\lambda}{2m}\left&space;\|&space;w&space;\right&space;\|_1" title="\frac{\lambda}{m}\sum_{k=1}^{n_x}\left | w \right | = \frac{\lambda}{2m}\left \| w \right \|_1" />

   ​	*w will end up being sparse (lots of 0 in it).*

   Why:

   when <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\lambda" title="\lambda" /> increase, then <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;w^{[l]}\approx&space;0" title="w^{[l]}\approx 0" />, so roughly linear.

2. **Dropout regularization**

   Implementing:

   - set keep.prop

   - get <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\partial&space;[l]" title="\partial [l]" />	

     ​	<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\partial&space;[l]&space;=&space;np.random.rand(al.shape[0],&space;al.shape[1])&space;<&space;keep.prob" title="\partial [l] = np.random.rand(al.shape[0], al.shape[1]) < keep.prob" /> (boolean value)

   - <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;al&space;*=&space;\partial&space;l" title="al *= \partial l" />

   - After dropout `a` will reduce by `(1 - keep.prob)`, to avoid `a` and `z` reducing too much, do <img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;al&space;/=&space;keep.prob" title="al /= keep.prob" />.

     1) When `keep.prob` increases, regularization effect and training set error reduce.

     2) When testing, no drop out, no `/ keep.prob`.

   Why:

   - Shrinking the squared norm of the weights.
   - The output cannot rely on any one feature.

   Tips:

   - Give smaller `keep.prop` to layer which easy to over fit.
   - Just set `keep.prop = 1` when checking the model, then set suitable values to train data.

3. **Data augmentation** --> A way to get more train data

   - flipping
   - random crops of image
   - rotation
   - distortion

4. **Early stopping**

   ​

