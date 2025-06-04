https://arxiv.org/pdf/2304.06795
https://arxiv.org/pdf/1211.3711
https://proceedings.mlr.press/v32/graves14.pdf
https://arxiv.org/pdf/2305.05084
https://arxiv.org/pdf/2005.08100

Example - (https://distill.pub/2017/ctc/ )

If we take a few seconds of speech sample ,  we have about 100sampes/sec input, in a streaming case we go through a while loop with a few seconds of speech sample

What we are talking about is one such iteration of 100’s of input tokens

That algo is called Dynamic Time Warping.

Conformer code: https://github.com/HarunoriKawano/Conformer/ 
I think this should match - it was from another conformer paper.

Very interesting analogy

The paper is a tradeoff in performance

They focus on long transcription

They try to fit hours of audio in one forward pass
But ___ question is a very valid one
They also mention some customization of custom vocabulary
With respect to say product names
This model essentially requires that out of distribution optimization later

The OOD customization info  is in the nvidia GitHub docs

Once again, the authors where Frankie copied this graphic didn't label the axes as query position and key position. 😜

So Kernel size decrease doesn’t decrease GMAC?
Thank you Frankie!

Thank you!

Awesome job

Thank you, Frankie! Great job.

A Gentle Intro to Dynamic Time Warping
https://www.blasbenito.com/post/dynamic-time-warping/ 

MultiDecode