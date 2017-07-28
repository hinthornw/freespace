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


Terrible Tiramisu: in which I compare RNN's added at the BEGINNING of block 5 (of 9) for a Tiramisu net.

1) CNN (W2TL): R0 = 
2) ADD (W2BL): R0 = 
3) ADDMULT (W1BR): R0 = 
4) MULTADD (W2BR): R0 = 
5) MULTMULT(W1BL): R0 = 





