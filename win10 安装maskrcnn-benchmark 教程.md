# win10 安装maskrcnn-benchmark 教程

[![img](https://cdn2.jianshu.io/assets/default_avatar/13-394c31a9cb492fcb39c27422ca7d2815.jpg)](https://www.jianshu.com/u/b5c451615cb6)

[sien_xx](https://www.jianshu.com/u/b5c451615cb6)关注

2019.03.05 16:28:39字数 478阅读 3,645

基本环境：WIN10+python 3.7+CUDA 9.0+cudnn 7.1+visualcppbuildtools_full

有了最后那个就不需要装Visual Stdio

[visualcppbuildtools_full下载](https://pan.baidu.com/s/1J0CAz_d9semPyiEWu8nIbg) 提取码：k490

基于官方安装教程的踩坑记录

[官方教程](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md)

# 1.装anaconda3

注意点是这一步的两个框都勾上

![img](https://upload-images.jianshu.io/upload_images/3844415-1bc302b97d65ee70.png?imageMogr2/auto-orient/strip|imageView2/2/w/503/format/webp)

# 2.用anaconda创建环境

conda creat --name test （test 环境名字，自取）

conda activate test 激活环境

在当前文件夹下创建一个安装目录 mkdir test（名字自取）

![img](https://upload-images.jianshu.io/upload_images/3844415-8bd65a06bc29a938.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/247/format/webp)

激活成功

cd test 进入目录



![img](https://upload-images.jianshu.io/upload_images/3844415-b30cff9ef72a3ec0.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/306/format/webp)

# 3.装 ipython

conda install ipython

# 4.markrcnn_benchmark and coco api dependencies

pip freeze>requirements.txt

pip install -r requirements.txt

# 5.装pytorch（不要装nightly）

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

(这里网速慢会装很久甚至失败，失败可以参考下面文章，把下载源改到清华)

[如何改安装源](https://www.jianshu.com/p/e3efaaf7e655)

# 6.装torchvision

（这个我感觉第五步是有的，不过保险起见，我还是再装了一遍）

pip install --no-deps torchvision

# 7.装pycocotools

pip install Cython

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

# 8.下载maskrcnn-benchmark

git clone https://github.com/facebookresearch/maskrcnn-benchmark.git

cd maskrcnn-benchmark



![img](https://upload-images.jianshu.io/upload_images/3844415-e7de489b5c69f6b0.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/482/format/webp)

# 9.修改SigmodiFocalLoss_cuda.cu文件

这个文件在test1->maskrcnn-benchmark->maskrcnn_benchmark->csrc->cuda

![img](https://upload-images.jianshu.io/upload_images/3844415-e2996b1269bacb4d.png?imageMogr2/auto-orient/strip|imageView2/2/w/667/format/webp)

#### 第一处：120行

dim3 grid(std::min(THCCeilDiv(losses_size, 512L), 4096L));

![img](https://upload-images.jianshu.io/upload_images/3844415-4778594cf812bfd9.png?imageMogr2/auto-orient/strip|imageView2/2/w/608/format/webp)

改成：

dim3 grid(std::min((long)(losses_size+511)/512L, 4096L));

#### 第二处：164行

dim3 grid(std::min(THCCeilDiv(d_logits_size, 512L), 4096L));

![img](https://upload-images.jianshu.io/upload_images/3844415-8d95595cfaff1368.png?imageMogr2/auto-orient/strip|imageView2/2/w/616/format/webp)

改成：

dim3 grid(std::min((long)(d_logits_size+511)/512L, 4096L));

保存文件

# 10.安装maskrcnn-benchmark

回到第8步的文件路径

![img](https://upload-images.jianshu.io/upload_images/3844415-b9f4cf86ce4fcd67.png?imageMogr2/auto-orient/strip|imageView2/2/w/474/format/webp)

python setup.py build develop



![img](https://upload-images.jianshu.io/upload_images/3844415-0dc18e8f77f12d5f.png?imageMogr2/auto-orient/strip|imageView2/2/w/511/format/webp)

安装成功

#  11.运行demo测试

cd demo

pip install opencv-python

pip install yacs

pip install matplotlib

python webcam.py --min-image-size 800

（运行这个需要有摄像头）

![img](https://upload-images.jianshu.io/upload_images/3844415-da57466659d26f65.png?imageMogr2/auto-orient/strip|imageView2/2/w/921/format/webp)

运行成功

### 一些参考： 

[python setup.py build develop fails on Windows 10](https://github.com/facebookresearch/maskrcnn-benchmark/issues/254)

[Maskrcnn-benchmark在windows系统的安装](https://www.jianshu.com/p/e3efaaf7e655)

[build maskrcnn-benchmark for win10+vs2017](https://github.com/danpe1327/remember_doc/blob/master/build maskrcnn-benchmark for win10%2Bvs2017.md)

[edited CUDA files to avoid Windows 10 build errors](https://github.com/facebookresearch/maskrcnn-benchmark/pull/271)

如果出错可以来这里找找有没有同病相怜的：

[官方bug讨论](https://github.com/facebookresearch/maskrcnn-benchmark/issues)





















# [Solved] dcn_v2_cuda.obj : error LNK2001: unresolved external symbol state caused by extern THCState *state; #14

 Closed

[ausk](https://github.com/ausk)  

## Comments

[![@ausk](https://avatars1.githubusercontent.com/u/4545060?s=88&v=4)](https://github.com/ausk)

 

### **[ausk](https://github.com/ausk)** commented [on 18 Apr](https://github.com/CharlesShang/DCNv2/issues/14#issue-434688949) • edited 



[![@ausk](https://avatars1.githubusercontent.com/u/4545060?s=88&v=4)](https://github.com/ausk)

 

Author

### **[ausk](https://github.com/ausk)** commented [on 18 Apr](https://github.com/CharlesShang/DCNv2/issues/14#issuecomment-484459899) • edited 





 https://github.com/CharlesShang/DCNv2/issues/14 