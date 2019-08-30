# DR(eye)VE Project: code repository

*A deep neural network trained to reproduce the human driver focus of attention.*

<p align="center">
<a href="https://arxiv.org/pdf/1705.03854.pdf" target="_blank"><img src="img/overview.jpg" height="300px"/></a>
</p>

## Results (video)
<p align="center">
  <a href="https://www.youtube.com/watch?v=GKjzOcwoc68"><img src="http://img.youtube.com/vi/GKjzOcwoc68/0.jpg" alt="video_results" width="50%" height="50%"></a>
</p>

## How-To

This repository was used throughout the whole work presented in the [paper](https://arxiv.org/pdf/1705.03854.pdf) so it contains quite a large amount of code. Nonetheless, it should be quite easy to navigate into. In particular:

* `docs`: project supplementary website, holding some additional information concerning the paper.
* `dreyeve-tobii`: cpp code to acquire gaze over dreyeve sequences with Tobii EyeX.
* `semseg`: python project to calculate semantic segmentation over all frames of a dreyeve sequence
* [**`experiments`**](experiments)**: python project that holds stuff for experimental section**
* `matlab`: some matlab code to compute optical flow, blends or to create the new fixation groundtruth.

The [**`experiments`**](experiments) section is the one that probably interest the reader, in that is the one that contains the code used for developing and training both our model and baselines and competitors. More detailed documentation is available [**`there`**](experiments).

**All python code has been developed and tested with Keras 1 and using Theano as backend.**

### Pre-trained weights:

Pre-trained weights of the `multi-branch model` can be downloaded from [this](https://drive.google.com/open?id=1cS7VaguQCoqkZ5mHWO3zDj5mNlOY61bj) link.

---

### The code accompanies the following paper: 
<p align="center">
 <table>
  <tr>
  <td align="center"><a href="https://arxiv.org/abs/1705.03854" target="_blank"><img src="img/paper_thumb.png" width="200px"/></a></td>
  </tr>
  <tr>
  <td><pre>  @article{palazzi2018predicting,
  title={Predicting the Driver's Focus of Attention: the DR (eye) VE Project},
  author={Palazzi, Andrea and Abati, Davide and Solera, Francesco and Cucchiara, Rita},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={41},
  number={7},
  pages={1720--1733},
  year={2018},
  publisher={IEEE}
}</pre></td>
  </tr>
</table> 
</p>
<!--
<a href="" target="_blank"><img src="img/paper_thumb.png" height="250px"/></a>
<div align="right">
<pre>
@inproceedings{...}
</pre>
</div>
<div style="clear:both;"></div><br />
-->
