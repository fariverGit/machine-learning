# machine-learning
@(cs229)

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

##linear regression
###LMS(最小均方差算法)
#### BGD vs SGD
$$min  J(\theta )=\frac{1}{2}\sum_{i=1}^{m}\left ( h_{\theta}(x^{(i)})-y^{(i)} \right )^{2}$$
 当仅有一个样本时：
$$\frac{\partial }{\partial \theta_j}J(\theta)=(h_\theta(x)-y)x_{j}$$
多样本时的更新算法：
#####BGD
Repeat until convergence:{
for every j:
$$\theta_{j}=\theta_{j}-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_{j}^{(i)}	$$
}
#####SGD
Repeat until convergence:{
for i=1 to m {
for every j:
$$\theta_{j}=\theta_{j}-\alpha\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_{j}^{(i)}	$$
&emsp;&emsp;}
}
#####比较
| method |   原理| 性能 |
| :----: | :--------:| :--: |
| BGD  | 用所有样本依次更新每一个参数 |  慢、占内存   |
| SGD  | 每个样本都更新所有参数 |  快、常用   |
####Newton's method
 对于凸函数的代价函数最小化，除了SGD与BGD还有一个常用的算法：Newton's method
 该方法的主要思想是每次学习的步长为$\Delta$(根据梯度得出)，而非固定学习率$\alpha$
 以LMS凸函数示例算法过程
 ![newthon method|center](./newtonmethod.jpg)
当样本特征为多维的时候，$\theta$也是一个向量，这时的更新方式为：
$$\theta = \theta - H^{-1}\bigtriangledown_{\theta}l(\theta)$$
$$H_{ij}=\frac{\partial^2 l(\theta)}{{\partial \theta_i}{\partial \theta_j}}$$
Newton's method方法的缺点就在于：
- 海森矩阵的逆不一定存在，就算存在计算量也比较大
- 当n比特别大的时候，该算法不一定比SGD快
 
###the normal equation
####$\bigtriangledown _{A}f(A)$含义
1. f 本身代表一个关于矩阵的函数
	- 表示f关于矩阵A的导数
	- 其自变量为矩阵A
	- 应变量为一个实数
2. $\bigtriangledown _{A}f(A)$是一个矩阵，矩阵的第i行j列的元素为f(A)关于$A_{ij}$的偏导数

####normal equation的推导
1. tr operator:
 $$trA=\sum_{i=1}^{n}A_{ii}$$
 即矩阵A的迹为其对角线元素之和，为一个实数
2. 预备公式：
$$trA = tr A^{T}$$
$$traA=atrA$$
$$\bigtriangledown_{A^{T}}trABA^{T}C=B^{T}A^{T}C^{T}+BA^{T}C$$
3. 推导
$$\bigtriangledown_{\theta}J(\theta)=0\rightarrow\theta=(X^{T}X)^{-1}X\vec{y}$$
###cost function的概率解释
 
 假设$\epsilon^{(i)}=y^{(i)}-\theta^{T}x^{(i)}$服从独立同分布的高斯分布
 则
 $$p(\vec{y}|X;\theta)=L(\theta|X,\vec{y})=L(\theta)=\prod_{i=1}^{m}\frac{1}{\sqrt{2\pi}\delta}e^{\frac{(y^{(i)}-\theta^{T}x^{(i)})^2}{2\delta^{2}}}$$
| 函数      |    表达式 | 含义  |
| :--------: | :--------:| :--: |
| 概率函数  | $p(\vec{y}|X;\theta)$ |  以$X,\theta$为参数，关于$\vec{y}$的函数   |
| 似然函数  | $L(\theta|X,\vec{y})$ |  以$X,\vec{y}$为参数，关于$\theta$的函数   |
 求似然函数的最大值$\leftrightarrow$求概率函数的最大值，也$\leftrightarrow$求$\frac{1}{2}\sum_{i=1}^{m}(y^{i}-\theta^{T}x^{(i)})^{2}$的最小值(可推导)
 但为什么要求概率函数的最大值呢？
- 使每一个样本尽可能预测准确$\leftrightarrow$使每一个$p(y^{(i)}|x^{(i)};\theta)$尽可能大
- 也可以从使每一个$\epsilon^{(i)}$尽可能接近于0的角度来理解

##逻辑回归
 逻辑回归实际上是分类问题，名字上含有回归二字是因为算法思想是由线性回归演变而来的。
- 区别于线性回归问题，逻辑回归问题的样本中的$y^{(i)}$只能取0与1两个值，$y^{(i)}$称为样本的标签
- 逻辑回归问题依然可以用线性回归算法解决，但是问题在于当“无论的线性回归模型的$\theta$取何值，当样本的$x^{(i)}$趋近于无穷时，它的预测值也趋近于无穷大”，这与逻辑回归的标签只能取0,1两个值的规定不相符。
###假设函数
基于线性回归算法修改，很直观的容易想到增加一个可导的映射函数将线性回归的预测值映射到0，1之间。没错这个函数就是sigmoid function 与称为逻辑函数。
$$g(z)=\frac{1}{1+e^{-z}}$$
sigmoid function的另外一个优势在于${g}'(z)=g(z)(1-g(z))$，这样简洁的表示在反向传播过程中能提高效率。
 
将线性回归的预测值作为sigmoid function的输入得：
$$h_{\theta}(x)=g(\theta^{T}x)=\frac{1}{1+e^{-\theta^{T}x}}$$
###cost function
假设
$$P(y=1|x;\theta)=h_{\theta}(x)$$
 $$P(y=0|x;\theta)=1-h_{\theta}(x)$$
 则$$p(y^{(i)}|x^{(i)};\theta)=(h_{\theta})^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}$$
$$ l(\theta)=logL(\theta)=log\prod_{i=i}^{m}p(\vec{y}|X;\theta)$$
求似然函数$l(\theta)$的最大值$\leftrightarrow$求$J(\theta)=-\frac{1}{m}l(\theta)$的最小值,所以
\begin{align}
J(\theta) =
 -\frac{1}{m} \left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}
并且易推导求$H(\theta)$最小值的过程依然可以用LMS中的SGD公式，只不过$h_{\theta}$的含义不相同
#### 比较线性回归的$J(\theta)$推导过程与逻辑回归的$J(\theta)$推导过程
可以看出$J(\theta)$的推导过程基本一致：
1. 对样本的分布进行假设：
- 线性回归预测误差服从正态分布
- 逻辑回归服从0-1分布
2. 根据样本间独立同分布求整个样本的分布，再取对数求cost function
3. 根据cost function求模型loss function
 

