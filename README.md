# multi-task-UDA

This is the repository for the paper:

>* Jingxiao Liu

[[slides]](docs/slides.pdf)[[paper]]()[[video]](docs/video.mp4) 

### Description
We introduce a novel approach for multi-task unsupervised domain adaptation. This approach is developed for bridge health monitoring using drive-by vehicle vibrations, but it can be applied to other problems, such as digit recognition, image classification, etc.

![The architecture of our hierarchical multi-task and domain-adversarial learning algorithm. The red and black arrows between blocks represent source and target domain data stream, respectively. Orange blocks are feature extractors, blue blocks are task predictors, and red blocks are domain classifiers.](imgs/arch.png)

In this repository, we demonstrate our approach through a digit recognition example, which transfers model learned using MNIST data to MNIST-M data and conducts two tasks: odd-even classification and digits comparison.

For the drive-by bridge health monitoring application, the implementation is similar.

Note: the drive-by bridge health monitoring experiment involves data that are not publicly available. 
To get this dataset, please send an email to [Jingxiao Liu](mailto:liujx@stanford.edu).

### Code Usage
```
git clone https://github.com/jingxiaoliu/multi-task-UDA.git
cd multi-task-UDA
```

Run the digit recognition example with 'demo_mnist.ipynb'.

### Contact
Feel free to send any questions to:
- [Jingxiao Liu](mailto:liujx@stanford.edu), Ph.D. Candidate at Stanford University, Department of Civil and Environmental Engineering.

### Citation
If you use this implementation, please cite our paper as follows: