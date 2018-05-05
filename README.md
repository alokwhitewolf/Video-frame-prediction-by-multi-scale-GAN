# Video frame prediction by multi scale GAN
![Image](media/best.gif)
<br>This is a <a href="https://chainer.org/">Chainer</a> implementation of <a href="https://arxiv.org/pdf/1511.05440.pdf">"Deep Multi Scale Video Prediction Beyond Mean Square Error"</a>.
The neural network is trained to predict predict upcoming future frame of the video given past few frames. This project has been influenced by the Tensorflow <a href="https://github.com/dyelax/Adversarial_Video_Generation">implementation</a>
of the same paper by dyelax. 

## Why Video frame prediction is important?
The ability of a model to predict future frames requires to build accurate, non trivial internal representations. Learning to
predict future images from a video sequence involves the construction of an internal representation that models the
image evolution accurately, and therefore,  to  some  degree,  its  content  and  dynamics. 

Also, a robust model that can build good internal representations of it's environment can be

## Why use GANs ?
Using an l2 loss, and to a lesser extent l1, produces blurry predicitons, increasingly worsen when predicting further in the future
. If the probability distribution for an output pixel has equally likely modes v1 and v, then the v_avg = (v1 + v2)/2 minimizes the l2
loss over the data even id v_avg has very low probability. In case of l1 norm, the effect diminishes but do not disappear.
<br>
![Image](media/bimodal-distribution-2.jpg)

GANs come into rescue because of the inherent nature of the way it's trained. 
<incomplete!! I am writing the README if you are reading this>
## How to run
<incomplete!!I am writing the README if you are reading this>
