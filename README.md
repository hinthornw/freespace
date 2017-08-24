## Recurrent Research

**Author:** William Hinthorn  
**permanent email:** whinthorn at gmail dot com


So far, our results have been minimal on the Resnet-50 base model. Recurrent information seems to add little (it in fact detracts from the accuracy in general) to the accuracy of the model if naively added to the end. This makes sense from a statistical standpoint as well. If you have a model which is trained to make a prediction given a particular input and add a second model which is designed to guess the output of the next frame (or later frames) from a previous input, without a proper post-processing system, the inference in the temporal dimension (from inputs t-1, i > 0) would almost always be of lowquality/ higher noise than that made at time. Improvements claimed in the semi-supervised setting in the Davis dataset have been primarily fueled by a low-noise, gt input at time t=0. Errors accumulate over time, which manifests itself after training in the extremely low values of the weights within the recurrent units (every weight is very small). This is the case for both vanilla RNN (resrec[Dual]\*Elm50) and the GRU structures.

These observations lead me to pose the question in a slightly different light. It at least makes one wonder ways in which this information CAN be used. Proposals include:

* **Dual stack:** Initiate the sequence with a proposal form a large (Res50, Res101, etc), accurate network, then feed this to a smaller (res12, res18) network. The pseudo accurate temporal information could boost thsoemwhat accurate, very fast rnn.  
* **Differential/ Kalman networks:** Instantiate the sequence with a CNN proposal then train a network to operate on the DIFFERENCES between images (or differences between features extracted at layer l). This is in the vein of the Predictive-corrective blocks in Russakovsky, et al. Perhaps focusing on changes between frames could lead to a much smaller network (if initial prediction is precise)  
* **Sequential Attention:** A number of methods have been proposed for object counting, etc. We could develop a learnable pyramid of image precision where the output of network at time t decides which area of the input at t+1 should be viewed in higher definition. Could use 3 channel inputs all of same dimensions but of different precision. 
    

    The last proposal to me seems interesting but could be difficult ot implmement and train in a limited time.


Comments on "Novel" CNN architectures:

Leapnets:  Perhaps have some potential use, but are VERY slow... They will be even slower if I do the necessary modifications and use cat instead of add to allow for more clear flow of information. (a la DenseNet) That being said, we could come to a compromise and 

DenseShuffle:

I compare two styles - One with a straight up method, one with a fusion from stem at midway and a single smoother.

A) Round 0 : 0.676

B) Round 0: 0.653

PERHAPS too early to call, but it seems that this raw data hurts the predicitons significantly because it hasn't been sufficiently smoothed out.





ll's July-August Summary

Thank you to 梁继and 夏老师 for your patience, mentorship, and optimism this summer. I am truly fortunate to have had the chance to work with the freespace team. I look forward to the next time we cross paths!


### CNN Structures

Before launching into temporal research, I experimented with a number of alterations to our original CNN model.  Experiments were focused on increasing IoU without greatly changing the Parameter space.

#### Dilations and DUC
Maintaining resolution in deeper layers is an easy way to increase the localization accuracy when performing semantic segmentation. The algorithm à trous, or dilated convolution, is a way to maintain resolution while increasing the receptive field in later layers. Recent successes of Fisher Yu and Wang et. al in [Dilated Convolutional Networks](https://www.arxiv.org/abs/1705.09914) and [Understanding Convolution for Semantic Segmentation](https://www.arxiv.org/abs/1702.08502v1) respectively. 

DUC (dense upsampling convolution) is an operation proposed by Shi et. al in [Is the Is the deconvolution layer the same as a convolutional layer?](https://www.arxiv.org/abs/1609.07009) which seeks to strengthen upsampling operations by first expanding the width at a lower resolution then deterministically reshaping the filter to a higher resolution. In this way, the convolution can *always* (apart from edge cases, of course) operate on data rather than inserted holes.

Tests were performed on Cityscapes dataset using 26 different labels. I focused on road IoU. The base architecture was ResNet18 (though this is a misnomer since there were 4 additional transposed convolutional layers to upsample at the end). The model is defined by its down- stride of (i.e. for ds 16, 448x448 input becomes 28x28 in last block, requiring the model to up-sample $ds=\log{16}=4$ times before outputting the results ). The model was structured as a standard FCN without additive connections form lower levels to up sampled higher ones. This architecture is structured in 4 residual blocks of 2, 2, 6, 6, layers each. Each model was trained from scratch on the Cityscapes dataset with random cropping, flipping, and resizing. 

| Downstride | Downsample Op    | Upsample Op     | Dilation Block 3 | Dilations Block 4 | Road IOU | Comments                                                           |
|------------|------------------|-----------------|------------------|-------------------|----------|--------------------------------------------------------------------|
| 32         | Maxpool          | Transposed Conv | [1,1,1,1,1,1]    | [1,1,1,1,1,1]     | 0.94143  | Control                                                            |
| 32         | Conv w/ stride 2 | Transposed Conv | [1,1,1,1,1,1]    | [1,1,1,1,1,1]     | 0.94222  |                                                                    |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,1,1,1,1]    | [2,2,2,2,2,2]     | 0.94512  | Replace last downsample with dil=2                                 |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,2,2,2]    | [4,4,4,4,4,4]     | 0.94531  |                                                                    |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,2,2,2]    | [4,4,4,4,1,1]     | 0.9451   |                                                                    |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,2,2,2]    | [4,4,4,4,1,1]     | 0.947    | Removed residual connections from last two layers (see Yu, et. al) |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,2,3,3]    | [2,2,3,3,5,5]     | 0.938    | HDC from Wang et. al                                               |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,3,5,9]    | [7,5,3,2,1,1]     | 0.9501   | First Attempt to balance the two                                   |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,1,2,3,5,5]    | [2,3,5,5,1,1]     | 0.9531   | First Attempt to balance the two                                   |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,2,3,5,3,2]    | [2,3,5,3,2,1]     | 0.949    |                                                                    |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,2,2,3,3,5, 5] | [9,9,5,1,1]       | 0.959    | Shifted a layer over for continuity                                |
| 16         | Conv w/ stride 2 | Transposed Conv | [1,2,2,3,3,5]    | [9,17,3,2, 1,1]   | 0.95173  | dilating > 9 result in too sparse an operation                     |
| 16         | Conv w/ stride 2 | DUC             | [1,2,2,3,3,5, 5] | [9,9,5,2,1,1]     | 0.963    | Dense Upsampling Convolution (using Pixel Shuffle)                 |
| 8          | Conv w/ stride 2 | DUC             | [2,2,3,3,5, 7]   | [9,9,5,2, 1,1]    | 0.968    | increasing resolution again made computations much too slow        |


Dilations do in fact help with the predictions, and having large dilations can help greatly. One could try researching replacing the dilated convolutions on the greater end of the spectrum with (i.e. dil>=8) with dilations of a greater kernel size and smaller dilations for greater density in predictions. For example, it may be worthwhile to try replacing a 3x3 convolutional kernel with a dilation of 9 with a 5x5 kernel with a dilation of 4. I began experiments with this but had to stop in order to move on to RNN work.


### Strange Architectures to enable greater connectivity

Outside of work, I began to wonder whether allowing greater flexibility in the feedforward structure of the net would allow greater flexibility in hierarchical connections and the *eventual* incorporation of more temporal information throughout the web. This hypothesis stems from the recognition by a number of papers in action recognition (e.g.  [Every Moment Counts](http://arxiv.org/abs/1507.05738), [Predictive-Corrective Networks for Action Recognition](http://arxiv.org/abs/1704.03615), and [Convolutional Gated Recurrent Networks for Video Segmentation](http://arxiv.org/abs/1611.05435)) that adding temporal information can benefit the predictions if inserted in the last few layers (i.e. fc in VGG) AND/OR in the very first few layers (after the stem). Networks are nonlinear, but it still is interesting that the benefits can be realized by bookending operations but not through dense connections. I hoped to develop networks that a) matched the performance of our existing model and b) rapidly changed the resolution to pave the way for greater hierarchical/temporal connectivity. All tests were performed on CityScapes.

**LeapNet**
*Idea*: Extend the idea of feature cascades, residual learning, etc. in a manner that allows full flexibility of hierarchical connections. ( Catch a trend here? Perhaps I'm crazy, but I think that *eventually* networks will rely on greater interconnectivity between a high-level semantic state and low-level processing (call it attention, context, or your own favorite term) in order to make an accurate conjectures about the scene presented. LeapNets learn residuals that are in the same space as the original extracted features (i.e. output of the initial stem). Each residual block operates increasingly low  resolution filters but is bookended by dense reshuffling down and up to sum with the residual in a high resolution (DDC -> BLOCK -> DUC -> SUM).

*Tl;dr* LeapNets were too slow as originally proposed (with a smoothing convolution at high resolution before each downsize operation), but after modifications, their performance approached that of the original res18 (still falling short by 0.08 points for road IoU on CityScapes).  It's a strange formulation and I haven't taken adequate time to really get the design to work. 

[INSERT TABLE HERE]


**ShuffleNet** (not to confused with the model of the same name which came out after I began work on this)
*Idea*: Max unpooling boosted accuracy for pooling-based models; enforce symmetry on nets with DUC. If this succeeds, it may be possible to have resolution changes  at higher orders of magnitude with graceful degradation in accuracy. Much experimentation was done to compare downshuffles with 1x1 convolutions, 3x3 convolutions, and grouped convolutions. 
*Tl;dr*: Initial tests show that there is no graceful degradation, but I still am encouraged by results of models where strided convolutions are replaced by  Dense Downsampling Convolutions (DDC) and believe there is some chance that a multicontext pipeline could be developed using this type of layer. The best Road IoU that was validated was 0.927.

**TiramisuNet** ([The Hundred Layers Tiramisu...](https://www.arxiv.org/abs/1611.09326v2))
*Idea:* DenseNet was successful for image recognition, so why not modify it for semantic segmentation?
*Icing on the cake*: The structure is similar to UNet in its inclusion of more processing in the up-sampling path. Perhaps the propagation of higher-level semantic information at the bottom of the U (but still far before the end of the net) would allow the network to utilize temporal information for context but focus on lower levels for localization.
*Tl;dr*: None of the relationships were able to significantly beat the baseline CNN. This may be related to the position of the added recurrent unit. The road IoU for the CNN version on the Kitti dataset was 0.942247.






### RNN Tests

###  Elman connection comparison

My Vanilla rnn comparison was first based on the Elman Network, defined as:
\begin{equation}
h_t = \sigma_h(W_h x_t + U_h h_{t-1} + b_h)
\end{equation}\begin{equation}
y_t = \sigma_y (W_y h_t + b_y)
\end{equation}

Where $h_t$, $y_t$ are the hidden and output layer values respectively at time $t$, and $W$ and $U$ are weight matrices of dimensionality s.t. the input $x_t$ is mapped to to the hidden layer space, and the previous hidden layer state is linearly transformed within the hidden layer space.

I modified the Elman update to compare the following types of connections:

$$y_t = x_t +  W_{x,h}f(W_{h,x} \ast x_t + W_h \ast h_{t-1})$$ $$
y_t = x_t - W_{x,h}f(W_{h,x} \ast x_t + W_h \ast h_{t-1})$$ $$
y_t = x_t \circ  W_{x,h}\sigma(W_{h,x} \ast x_t \circ W_h \ast h_{t-1})$$ $$
y_t = x_t \circ W_{x,h}\sigma(W_{h,x} \ast x_t \circ \sigma(W_h \ast h_{t-1}))$$ 

Note that thes Elman updates were modified s.t. inner products are replaced by 3x3 convolutions, and the biases are eliminated. The hidden layer was set to have 2x the channels of the input. Recognizing that this is still not the simplest sort of temporal connection, I also compared with the following updates:
\begin{equation} y_t = y_t + y_{t-1} \end{equation}\begin{equation} 
y_t = x_t + (W_h\ast(x_t - y_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t + (W_h\ast(x_t - x_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t + \sigma_e(W_h\ast(x_t - y_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t+ W_{x,h}\ast \sigma_e(W_x\ast x_t + U_h\ast h_t) \end{equation}\begin{equation}
y_t = x_t- W_{x,h}\ast\sigma_e(W_x\ast x_t + U_h\ast h_t) \end{equation}\begin{equation}
y_t = x_t \circ W_{x,h}\ast\sigma_e(W_x\ast x_t + U_h\ast h_t) \end{equation}\begin{equation}
y_t = x_t \circ W_{x,h}\ast\sigma_s(W_h\sigma_e(W_x\ast x_t + U_h\ast h_t) \end{equation}


$\sigma_e$ was tested using ReLU and ELU, and $\sigma_s$ was tested as ReLU and Sigmoid. In the end, the results using ELU and Sigmoid for $\sigma_e$ and $\sigma_s$ respectively were found to be the best (or least destructive in most cases since the recurrent information failed to significantly help the models.

Beyond simple RNN units, I experimented with convolutional GRU, as is carelessly proposed in [Convolutional Gated Recurrent Units for Video Segmentation](http://arxiv.org/abs/1611.05435). The updates tested for this are as follows:



Additionally, I experimented with replacing the convolution layers with two-layer mini-networks and in separate experiments tested the use of dilated convolutions, but neither of these ideas improved the results. The two-layer updates improved or hurt the analogous update by a mere factor of +/-0.003, and the algorithm a trous update hurt the network on average by 0.005 points. I have all the results in my journal, but they aren't that worth writing up. The operations also led to an additional delay in time, and the two-layer update significantly altered the parameter space without improving the predictions.

Without further delay, the slightly unfortunate results on the kitti dataset ( showing the best updates)

| Equation | Description         | Road IOU Training | PR Road IOU Validation |
|----------|---------------------|-------------------|------------------------|
| 0        | None (CNN)          | 0.9213            | 0.942247               |
| 1        | x + w(f(wx + wh)    | 0.8922            | 0.9388                 |
| 2        | x-wf(wx+wy)         | 0.9199            | 0.943247               |
| 3        | x*wS(wx*wh)         | 0.8101            | 0.84902                |
| 4        | x*wf(s(wx)*wh)      | 0.8012            | 0.84792                |
| 5        | x+y_{t-1}           | 0.9021            | 0.91013                |
| 6        | x + W(x-y_{t-1})    | 0.9143            | 0.93891                |
| 7        | x+W(x-x_{t-1})      | 0.9132            | 0.93122                |
| 8        | x+S(W(x_t-y_{t-1})  | 0.87082           | 0.9134                 |
| 9        | x +W*f(wx + wh)     | 0.8665            | 0.91363                |
| 10       | x - W*f(wx + wh)    | 0.9212            | 0.944132               |
| 11       | x*W*S(Wx + Uh)      | 0.8211            | 0.8792                 |
| 12       | X*W*f(W(f(Wx + Uh)) | 0.8327            | 0.8374                 |



### Linear is better?

Though my results above didn't seem to support this, the successes which Tianyu had had using a additive linear transformation prompted me to experiment with this. I switched from the slow Tiramisu model of previous experiments and again test the following recurrent layers on our newly labeled dataset. The recurrent unit is placed on the end of the 

\begin{equation} y_t = x_t + (W_1\ast1(x_t - y_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t + (W_1\ast(x_t - x_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t + W_2\ast(W_1\ast(x_t - y_{t-1}) )\end{equation}\begin{equation} 
y_t = x_t + W_1\ast(y_{t-1}) \end{equation}

In order to directly compare models with the same number of parameters, I set the control as $y_t = x_t + W_1\ast x_t$ rather than simply $y_t = x_t$. 

| Equation | Description                   | Road IOU Training | PR Road IOU Validation |
|----------|-------------------------------|-------------------|------------------------|
| 0        | CNN                           | 0.91983           | 0.9316                 |
| 1        | CNN + one conv                | 0.92341           | 0.93213                |
| 2        | Linear conv                   | 0.91623           | 0.93207                 |
| 3        | Linear conv plus acceleration | 0.89183           | 0.89134                |

The validation accuracy increased slightly both with the rnn added and with the extra convolution; at the very least it didn't hurt the model.

### Recurrent Momentum

My experiments for this were too limited to definitively say that the following formulation is worthless, but at the same time, the results aren't promising.

$$\Delta_t = W_\Delta \ast (y_t - y_{t-1})$$$$
a_t = a_{t-1} + W_a \ast (\Delta_t + \Delta_{t-1})$$$$
y_t = W_x \ast x_t + W_h \ast \Delta_t + W_a \ast a_t$$

The reason I believe it was too early for us to test this type of model is that we still haven't sufficiently *proven* that the temporal data can be effectively incorporated in short sequences. Including momentum (which I had hoped could assist in letting the model logically separate short and long-term relationships and more efficiently process the results), would only really help for sequences > 3 frames.

The model above lagged behind simpler models by roughly 1 point (for sequences of length 8).


### Entropic Gating

FCN's can be strengthened by directly adding outputs from  high resolution, low semantic value layers to the semantic, low-resolution layers when up-sampling to a final prediction. That said, I noticed from some of the predictions that *at times* the network would predict holes in an otherwise contiguous road where shadows, cracks, some road markings, and other debris is. This was not the case with the original CNN which predicted large amorphous blobs of road with inaccurate edges. Recognizing this, I sought to screen out distracting information from lower levels which could confuse the final predictions. I hypothesized that edge regions of prediction maps would have high entropy, and likewise regions which are clearly not road would have low entropy. This then could be used as a surrogate for regions of high expected error, since error and misclassification rate were found to be highly correlated. Because of this, I experimented with creating masks using the exponentiated entropy of a single pixel along the channels of a prediction map. I first used a single prediction map by a single DUC step from the end of block 4 in ResNet18 as a mask for every layer then also experimented with having 4 separate prediction steps to get error directed at that specific point in the network.

If this gating is successful, it can lead to research in other areas, including but not limited to:
    1)  Entropy propagation: regions of high entropy in consecutive frames likely belong to similar objects or object edges. One can then use the trajectory of these masks to guide recurrent updates.
    2) Gated updates: Rather than simply adding or multiplying, one may mask the regions that needn't be passed on to fine tune results at every step. This can also be a sort of self-regulation function, allowing the network to focus on the present information in cases when past predictions are too noisy. It may limit the increase of error in later steps as has been documented by all the members of our team.
    3) Bilateral/ general filtering: Rather than directly measuring the entropy at a certain location, treat the softmax values as an embedding and create object masks based on vector distances from pixels within a set distance. (I.e. instead of im2col->elementwise multiply do im2col-> subtract from index 0), then measuring the distance of a pixel from its neighbors, one may approximate its abjectness. Once the network may divide the picture into semantic regions, it *may* be able to better understand how semantic information is propagated.

etc.

I defined exponentiated entropy $\mathcal{E}$ as :
$$\mathcal{E} = e^{-H} $$
where $H = -\sum p_c log(p_c)$ is calculated using the softmax and log softmax on a prediction map.

I tested using both this and its inverse (i.e. 1-$\mathcal{E}$) as multiplicative masks, and my hypothesis that its inverse variant would outperform the former was supported. Allowing targeted information from early stages around areas of uncertainty creates better predictions than silencing the information from those regions and allowing more data from low stages flow into ages of low entropy (potentially confusing the network on previously OK predictions)

The actual results on simple hierarchical CNN tests have been somewhat discouraging. 

The original model 


### Recurrent Attention

There has been little research in this over the past two years, so it may be that it is not a fruitful path to investigate (which is also the reason why I stopped after the initial tests). I believe that this is too gimmicky at the moment and requires more specific objective functions if the system could work. It is an attention variant of the The goal of the attention is to allow the network to intelligently increase the resolution of areas in the image which are far, changing rapidly, or complex. In essence, at each timestep the network first selects a coordinate $(x,y)$ on which to focus and is fed 3 channels. One is the entire image scaled to the input size (i.e. 448x448). One is a cropped image  in full resolution centered on the area of interest, and one is a frame somewhere in between. The location is vectorized and used to map the three frames onto a hidden space, and the predictions proceed as normal. The network outputs predictions and the next location to which to attend. The structure could be extended from that proposed by Graves et. al. in [Recurrent Models of Visual Attention](http://arxiv.org/abs/1406.6247%5Cnhttp://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention%5Cnhttp://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
Care must be taken to make the network aware of exactly where the focus is occurring.

### Grid LSTM

I haven't finished coding [this](http://arxiv.org/abs/1507.01526) up, but it generalizes the 2d LSTM to the temporal (or other) dimension, inputting and outputting a concatenated vector of N hidden layer activities. The context is propagated to neighboring blocks and corresponding blocks at the next time step to allow the entire image to be fully analyzed. Weights are shared along analogous axes.

