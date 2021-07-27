# HierMUD: a Hierarchical Multi-task Unsupervised Domain adaptation framework

This is the repository for the paper:

>* HierMUD: Hierarchical Multi-task Unsupervised Domain Adaptation between Bridges for Drive-by Damage Diagnosis,
Jingxiao Liu, Susu Xu, Mario Bergés, Hae Young Noh

[[slides]](docs/slides.pdf)[[paper]](https://arxiv.org/abs/2107.11435)[[video]](docs/video.mp4) 

### Description
We introduce HierMUD, a novel approach for multi-task unsupervised domain adaptation. This approach is developed for bridge health monitoring using drive-by vehicle vibrations, but it can be applied to other problems, such as digit recognition, image classification, etc.

![The architecture of our hierarchical multi-task and domain-adversarial learning algorithm. The red and black arrows between blocks represent source and target domain data stream, respectively. Orange blocks are feature extractors, blue blocks are task predictors, and red blocks are domain classifiers.](imgs/arch.png)

In this repository, we demonstrate our approach through two examples:

- A drive-by bridge health monitoring example, which transfers model learned using vehicle vibration data collected from one bridge to detect, localize and quantify damage on another bridge.
- A digit recognition example, which transfers model learned using MNIST data to MNIST-M data and conducts two tasks: odd-even classification and digits comparison.

Note: the drive-by bridge health monitoring experiment involves data that is not publicly available. We will work towards making the experiment replicable without violating data usage policy.

### Code Usage
```
git clone https://github.com/jingxiaoliu/HierMUD.git
cd multi-task-UDA
```

- Run the drive-by bridge health monitoring example with 
```
jupyter notebook demo_dbbhm.ipynb
```
- Run the digit recognition example with
```
jupyter notebook demo_mnist.ipynb
```
### Contact
Feel free to send any questions to:
- [Jingxiao Liu](mailto:liujx@stanford.edu), Ph.D. Candidate at Stanford University, Department of Civil and Environmental Engineering.

### Citation
If you use this implementation, please cite our paper as follows:

```
@misc{liu2021hiermud,
      title={HierMUD: Hierarchical Multi-task Unsupervised Domain Adaptation between Bridges for Drive-by Damage Diagnosis}, 
      author={Jingxiao Liu and Susu Xu and Mario Bergés and Hae Young Noh},
      year={2021},
      eprint={2107.11435},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
