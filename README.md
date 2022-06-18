# PRIMFIT: Learning to Fit Primitives Improves Few Shot Point Cloud Segmentation
Published at Symposium on Geometry Processing 2022.

[Gopal Sharma](https://hippogriff.github.io/), [Bidya Dash](https://www.linkedin.com/in/bidyadash/), [Aruni RoyChowdhury](https://arunirc.github.io/), [Matheus Gadelha](http://mgadelha.me/), [Marios Loizou](https://marios2019.github.io/), [Liangliang Cao](http://llcao.net/), [Rui Wang](https://people.cs.umass.edu/~ruiwang/), [Erik G. Learned-Miller](https://people.cs.umass.edu/~elm/), [Subhransu Maji](https://people.cs.umass.edu/~smaji/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/)
***
<p align="center">
  <img src="snip.PNG" alt="drawing" width="300"/>
</p>
**TL;DR** PrimFit uses primitive fitting within a semisupervised
learning framework to learn 3D shape representations.
Top row: 3D shapes represented as point clouds, where the
color indicates the parts such as wings and engines. The induced
partitions and shape reconstruction obtained by fitting ellipsoids to
each shape using our approach are shown in the middle row and
bottom row respectively. The induced partitions often have a significant
overlap with semantic parts.

### Abstract
_We present PrimFit, a  semi-supervised approach for label-efficient learning of 3D point cloud segmentation networks. 
PrimFit combines geometric primitive fitting with point-based representation learning. Its key idea is to learn point representations whose clustering reveals shape regions that can be approximated well by
basic geometric primitives, such as cuboids and ellipsoids. The learned point representations can then be re-used in existing network architectures for 3D point cloud segmentation,
and improves their performance in the few-shot setting. According to our experiments on the widely used ShapeNet and PartNet benchmarks.
PrimFit outperforms several state-of-the-art methods in this setting, suggesting that decomposability into primitives is a useful prior for learning representations predictive of semantic parts.
We present a number of ablative experiments varying the choice of geometric primitives and downstream tasks to demonstrate the effectiveness of the method._
