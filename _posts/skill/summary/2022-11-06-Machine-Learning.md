# 机器学习入门



@[toc]
## 0.写在前面

>**本文大多数内容来自《神经网络与深度学习》**（邱锡鹏，神经网络与深度学习，机械工业出版社，https://nndl.github.io/, 2020.）**周志华-机器学习**，**也参考了很多其他笔记博客**。**仅作为学习记录**。

<br />


## 1.机器学习基本概念

**① 机器学习是什么？**

机器学习就是让`计算机从数据中进行自动学习`，得到某种知识或规律。



**② 样本和数据集**

​我们可以将一个标记好`特征`以及`标签`看作一个样本。

​一组样本构成的集合称为`数据集`。一般将数据集分为两部分：`训练集和测试集`。

>​训练集中的样本是用来训练模型的，而测试集中的样本是用来检验模型好坏的。



**③ 学习与训练**

​	我们通常用一个𝐷 维向量$$𝒙 = [𝑥_1 , 𝑥_2 , ⋯ , 𝑥_𝐷] ^T$$ 表示所有特征构成的向量，称为`特征向量`，其中每一维表示一个特征。

​	假设训练集 𝒟 由 𝑁 个样本组成，其中每个样本都是独立同分布的，计为：$$𝒟 = {(𝒙^{(1)}, 𝑦^{(1)}), (𝒙^{(2)}, 𝑦^{(2)}), ⋯ , (𝒙^{(𝑁)}, 𝑦^{(𝑁)})}. $$

​	我们希望让计算机从一个函数集合$$ℱ = {𝑓_1 (𝒙), 𝑓_2 (𝒙), ⋯}$$ 中自动寻找一个“最优”的函数`𝑓∗(𝒙) `来近似每个样本的特征向量 𝒙 和标签 𝑦 之间的`真实映射关系`，找到这个最优函数的过程就叫做`学习`或`训练`。

>以下给出一个学习的基本流程：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bf1ff44cc641f3be691829d0bd0edef4.png#pic_center =710x250)


​	**通过训练集，不断识别特征，不断建模，最后形成有效的模型，这个过程就叫“机器学习”！**

<br />

## 2.机器学习算法的类型



### 2.1 监督学习 

​如果机器学习的目标是建模样本的特征 𝒙 和标签 𝑦 之间的关系，并且训练集中每个样本都有`标签`，那么这类机器学习称为监督学习。


​根据标签类型的不同，监督学习又可以分为`回归问题`、`分类问题`和`结构化学习`问题。

​	（1） **回归问题**中的标签 𝑦 是**连续值**， 𝑓(𝒙; 𝜃)的输出也是连续值。比如未来几年预测房屋价格的走势，价格是一个连续的值。最后会按照顺序把输出值串接起来，构成一个曲线。
>**一元线性回归**
>一元线性回归是最简单，最基础的一种模型，是探究两个变量之间关系的一种统计分析方法。其模型为为：
>$$y=ax+b$$其中$x$为自变量，$y$为因变量，$a$和$b$为回归系数
>求解方法为最小二乘法
>$$
>\left\{\begin{array}{l}
a=\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} \\
b=\bar{y}-a \bar{x}
\end{array}\right.
$$



>**多元线性回归**
>探究一个因变量和多个自变量之间关系的一种统计分析方法​，其模型为：
>$$
>y=a_{l} x_{1}+a_{2} x_{2}+\ldots+a_{n} x_{n}+b$$其中$y$为因变量，${x_1,x_2,…,x_n为自变量}$，$a_1,a_2,a_3…,a_n,b$ 为回归系数
>求解方法类似一元线性回归，使用最小二乘法
>利用线性代数的形式，对多元线性回归的误差公式求导为：
>$$
>\begin{aligned}
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} &=\frac{1}{2} \frac{\partial\left\|\boldsymbol{y}-\boldsymbol{X}^{\top} \boldsymbol{w}\right\|^{2}}{\partial \boldsymbol{w}} \\
&=-\boldsymbol{X}\left(\boldsymbol{y}-\boldsymbol{X}^{\top} \boldsymbol{w}\right)
\end{aligned}
>$$令  $\frac{\partial}{\partial \boldsymbol{w}} \mathcal{R}(\boldsymbol{w})=0$ , 得到最优的参数 $\boldsymbol{w}^{*}$ 为：$\boldsymbol{w}^{*}=\left(\boldsymbol{X} \boldsymbol{X}^{\top}\right)^{-1} \boldsymbol{X} \boldsymbol{y}$


>**用最小二乘法来进行线性回归参数学习的图示**
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ee89297d29bf5b6def09f598afacf7bb.png#pic_center =300x260)
>**核心代码**
>```python
>from sklearn.linear_model import LinearRegression
>#调用最小二乘法函数求回归系数
>model=LinearRegression()
>model.fit(x_train,y_train)
># 显示斜率
>a = model.coef_[0]
># 显示截距
>b = model.intercept_
># 预测结果
>predict = model.predict(x_test)


（2）**分类问题**中的标签 𝑦 是**离散的类别**。在分类问题中，学习到的模型也称为分类器。分类问题根据其类别数量又可分为二分类多分类问题。猫狗识别例子就是一个二分类的问题，因为它输出的结果不是猫就是狗。
其中分类问题中最常用的算法就是`KNN`	，`(k近邻算法)`
>**KNN**
>**KNN的原理就是当预测一个新的值x的时候，选择它距离最近的k个点,这k个点中属于哪个类别的点最多,x就属于哪个类别**
>&emsp;
>其中K的选择很重要！！！，如图所示，**不同的K计算出的结果是不一样的！！**
>&emsp;
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/433bc2d3caca98f51ade6a53b76de099.png#pic_center)
>&emsp;
>**KNN核心代码**
>```python
>from sklearn import  preprocessing 
>from sklearn import neighbors
>#数据标准化
>scaler = preprocessing.StandardScaler().fit(X_train)
>X_train = scaler.transform(X_train)
>X_test = scaler.transform(X_test) 
> 模型计算
># 调用KNN算法包训练模型
>knn = neighbors.KNeighborsClassifier(n_neighbors=5)
>knn.fit(X_train,y_train)
># 检验模型
>y_pred = knn.predict(X_test)



​	（3）**结构化学习**是一种特殊的分类问题。我们之前学习到的学习模型的输入与输出一直以来都是向量，但是在实际问题中，我们的输入输出可能是别的结构。比如，我们可能会需要输入输出是**序列、列表或者树**。它的输出空间比较大，通常用动态规划的方式解决。（在入门阶段简单了解概念就好）


<br />

### 2.2 无监督学习 

​无监督学习是指`从不包含目标标签`的训练样本中自动学习到一些有价值的信息。



 - 典型的无监督学习问题有`聚类`、 `密度估计`、`特征学习`、`降维`等，其中`聚类`和`降维`比较常用。


（1）**聚类**就是按照某个特定标准把一个数据集分割成不同的类或簇。最经典的就是`k-means算法`聚类，它的流程为：

>1. 随机地选择k个点，每个点代表一个簇的中心； 
>2. 将其他所有对象根据其与各簇中心的距离，将它赋给最近的簇； 
>3. 重新计算每个簇的平均值，更新为新的簇中心； 
>4. 不断重复2、3，直到准则函数收敛。
>&emsp;
>以下这个图很好地表示了**k-means聚类的过程**：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e6660e531533c3bf9c06f06b5090b568.png#pic_center =500x300)
>**k-means核心代码**
>```python
>from sklearn import  preprocessing 
>from sklearn.cluster import KMeans
># 将属性缩放到一个指定范围,即(x-min)/(max-min)
>scaler = preprocessing.MinMaxScaler().fit(X_train)
>X_train = scaler.transform(X_train)
> 模型计算
># 调用k-means算法包训练模型
>model_km = KMeans(n_clusters=3)#指定分类
>model_km.fit(X_train)
>label_pred = model_km.labels_ #获取聚类标签
>centroids = model_km.cluster_centers_ #获取聚类中心

(2) **降维**一种能在减少数据集中特征数量的同时，避免丢失太多信息并保持/改进模型性能的方法, 其中最常用的一种算法就是`PCA`, 也称`主成分分析法`.
>**PCA**
>PCA的主要思想是降维,**把多指标转化为少数几个综合指标**.
>我们从**几何角度**来理解一下降维
>&emsp;
>以二维空间为例,有这样一些点分布在直角坐标系中,我们观察可以发现,无论是丢弃$x_1$选择$x_2$, 还是丢弃$x_2$选择$x_1$都不能很好地体现这些点的特征.![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7edc8ac7dc1e2b0cb1210063bf1fdf4e.png#pic_center =260x260)
>&emsp;
>但是,如果我们将$x_1$轴和$x_2$轴平移和旋转一下, 变成$F_1$轴和$F_2$轴
>$$
>\left\{\begin{array}{l}
F_{1}=x_{1} \cos \theta+x_{2} \sin \theta \\
F_{2}=-x_{1} \sin \theta+x_{2} \cos \theta
\end{array}\right.
>$$
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7863281b8af411e5768551ad9982d174.png#pic_center =260x260)
观察可以发现, **旋转变换后n个样品点在$F_1$轴方向上的离散程度最大**, **变量$F_1$代表了原始数据的绝大部分信息**，在研究某些问题时，即使不考虑变量$F_2$也无损大局。
>&emsp;
>我们将其推广到更高的维度上, 有:
>$$
>\left[\begin{array}{rl}
F_{1}= & u_{11} X_{1}+u_{21} X_{2}+\cdots+u_{p 1} X_{p} \\
F_{2}= & u_{12} X_{1}+u_{22} X_{2}+\cdots+u_{p 2} X_{p} \\
& \cdots \cdots \\
F_{p}= & u_{1 p} X_{1}+u_{2 p} X_{2}+\cdots+u_{p p} X_{p}
\end{array}\right.
$$其中$F_1$被成为第一主成分,$F_2$被称为第二主成分, 我们通常选择前k个主成分, 一般保留的信息大于85%.
>&emsp;
>**PCA核心代码**
>```python
>from sklearn.preprocessing import MinMaxScaler
>from sklearn.decomposition import PCA
>#将属性缩放到一个指定范围,即(x-min)/(max-min)
>scaler = MinMaxScaler()
>scale_data = pd.DataFrame(scaler.fit_transform(datanew))
># PCA降维'
>#选择保留85%以上的信息时，自动保留主成分
>pca = PCA(0.85)
>data_pca = pca.fit_transform(scale_data) #data_pca就是降维后的数据



<br />

### 2.3 监督学习和无监督学习的对比
**举个监督学习的栗子：**

>​我们把一大堆的猫和狗的照片给机器，让机器识别出哪些是猫哪些是狗。当我们使用监督学习的时候，需要给这些照片打上`标签`。
>&emsp;
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29da8347354cce54b657afd0beae83c7.png#pic_center =460x260)
​	我们给照片打的标签就是“正确答案”，机器通过大量学习，就可以学会在新照片中认出猫和狗。
&emsp;
![<img src="https://gitee.com/Anonymous-cxb/image/raw/master/202210272331238.png" alt="image-20221027233119196" style="zoom: 50%;" />](https://i-blog.csdnimg.cn/blog_migrate/11d28b9ee9e2c2887eba2e79c58e8729.png#pic_center =460x260)

**举个无监督学习的栗子：**

>​我们还是一堆猫和狗的照片给机器，**不给这些照片打任何标签**，但是我们希望机器能够将这些照片分分类。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/423c2aa21f81468ab40428e914b71b2d.png#pic_center =530x260)
​通过学习，机器会把这些照片分为2类，一类都是猫的照片，一类都是狗的照片。虽然跟上面的监督学习看上去结果差不多，但是有着本质的差别：
&emsp;
​**非监督学习中，虽然照片分为了猫和狗，但是机器并不知道哪个是猫，哪个是狗。对于机器来说，相当于分成了 A、B 两类。**


1. 监督学习是一种目的明确的训练方式，你知道得到的是什么；而**无监督学习则是没有明确目的的训练方式，你无法提前知道结果是什么**。
2. 监督学习需要给数据打标签；而**无监督学习不需要给数据打标签**。一般而言， 监督学习通常需要大量的有标签数据集，这些数据集一般都需要由人工进行标 注，成本很高。
3. 监督学习由于目标明确，所以可以衡量效果；而**无监督学习几乎无法量化效果如何**。

<br />

### 2.4 强化学习
强化学习是一类通过交互来学习的。它有两个重要的特征：**反复实验和延迟奖励**。

>比如说：你想让小健同学好好学习算法。在这个过程中，要经历很多东西，比如看视频学习，刷题，打练习赛等等，要经过一段时间的ACM正式比赛才能知道小健同学的成果好不好。若正式比赛好成绩作为小健同学学习算法的奖励，这个不是一蹴而成的，需要正式比赛后才能给出反馈这个执行的过程对不对。因此，小健同学要根据比赛的结果多次摸索练习才能获得最优的学习方法。

<br />

## 3.机器学习的三个基本要素
机器学习方法可以粗略地分为三个基本要素：`模型`、`学习准则`、`优化算法`．
### 3.1 模型
对于一个机器学习任务，首先要确定其`输入空间𝒳` 和`输出空间𝒴`．不同机器
学习任务的主要区别在于输出空间不同．在二分类问题中`𝒴 = {+1, −1}`，在𝐶分类问题中`𝒴 = {1, 2, ⋯ , 𝐶}`，而在回归问题中`𝒴 = ℝ`． 
​	
>我们假设有一个“最优”的函数`𝑓∗(𝒙) `可以描述`输入空间𝒳` 和`输出空间𝒴`的真实映射关系，在一个我们假设的参数化的函数族`ℱ = {𝑓(𝒙; 𝜃)|𝜃 ∈ ℝ𝐷}`中取得。其中𝑓(𝒙; 𝜃)是参数为𝜃 的函数，也称为`模型`，𝐷 为参数的数量。
​
其中模型大致可以分为**线性模型**和**非线性模型**

<br />

### 3.2 学习准则
一个好的模型 𝑓(𝒙, 𝜃∗) 应该在所有 (𝒙, 𝑦) 的可能取值上都与真实映射函数𝑦 = 𝑔(𝒙)一致，即`|𝑓(𝒙, 𝜃∗
) − 𝑦| < 𝜖`, ∀(𝒙, 𝑦) ∈ 𝒳 × 𝒴, 

>这时候，我们还要引入一个很重要的概念：**损失函数**

#### 3.2.1 损失函数
损失函数是一个非负实数函数，用来`量化模型预测和真实标签之间的差异`。下面介绍几种常用的损失函数。

**0-1 损失函数** 

>最直观的损失函数是表现模型在训练集上的错误率，即0-1 损失函数：
>当预测值和真实值相等时，结果为0，否则为1
$$
\begin{aligned}
\mathcal{L}(y, f(\boldsymbol{x} ; \theta)) &=\left\{\begin{array}{cc}
0 & \text { if } y=f(\boldsymbol{x} ; \theta) \\
1 & \text { if } y \neq f(\boldsymbol{x} ; \theta)
\end{array}\right.\\
\end{aligned}
$$
>**优点：** 能够客观地评价模型的好坏
**缺点：** 不连续且导数为0，难以优化

**平方损失函数** 

>平方损失函数经常用在预测标签𝑦为实数值的任务中，定义为
$$\mathcal{L}(y, f(\boldsymbol{x} ; \theta))=\frac{1}{2}(y-f(\boldsymbol{x} ; \theta))^{2}$$
>平方损失函数一般不适用于分类问题． 

**交叉熵损失函数**
>对于两个概率分布，一般可以用`交叉熵`来衡量它们的差异．
𝒚和模型预测分布𝑓(𝒙; 𝜃)之间的交叉熵为
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{y}, f(\boldsymbol{x} ; \theta)) &=-\boldsymbol{y}^{\top} \log f(\boldsymbol{x} ; \theta) \\
&=-\sum_{c=1}^{C} y_{c} \log f_{c}(\boldsymbol{x} ; \theta)
\end{aligned}
$$
>比如对于三分类问题，一个样本的标签向量为$𝒚 = [0, 0, 1]^T$，
模型预测的标签分布为 $f(\boldsymbol{x} ; \theta)=[0.3,0.3,0.4]^{\top}$，
则它们的交叉熵为$−(0 × log(0.3) + 0 ×log(0.3) + 1 × log(0.4)) = − log(0.4)$

**Hinge 损失函数**
>对于二分类问题，假设 𝑦 的取值为 {−1, +1}，𝑓(𝒙; 𝜃) ∈ ℝ．
`Hinge损失函数`为
$\mathcal{L}(y, f(\boldsymbol{x} ; \theta))=\max (0,1-y f(\boldsymbol{x} ; \theta))$

#### 3.2.2 欠拟合和过拟合（包含正则化）

**欠拟合**：泛化能力差，训练样本集准确率低，测试样本集准确率低。
**过拟合**：泛化能力差，训练样本集准确率高，测试样本集准确率低。
**合适的拟合程度**：泛化能力强，训练样本集准确率高，测试样本集准确率高
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/55b6ec254c6d8c2f9a4dc8c7dc18217a.png#pic_center)

`欠拟合问题`往往是由于模型复杂度过低，特征量过少造成的，可以采用提高样本数量和提高模型复杂度等方法解决。
`过拟合问题`往往是由于训练数据少和噪声以及模型能力强等原因造成的．为了解决过拟合问题，一般会引入参数的`正则化`。
>**这时候我们又要引入一个很常见的概念了，什么是正则化？**

>我们引入一条公式来举例子
>$h_{w}(x)=w_{0}+w_{1} x_{1}+w_{2} x_{2}^{2}+w_{3} x_{3}^{3}+w_{4} x_{4}^{4}$
我们可以看到，这条公式中的**参数太多**了，这就会导致如上图三出现的那种弯弯曲曲的情况
>其实，像上图二，用公式$h_{w}(x)=w_{0}+w_{1} x_{1}+w_{2} x_{2}^{2}$就能**很好地拟合**了
>当然也可以写成$h_{w}(x)=w_{0}+w_{1} x_{1}+w_{2} x_{2}^{2}+0x_{3}^{3}+0 x_{4}^{4}$
>这时候**正则化**就能派上用场了


>假设我们的预测值$y_{pre_i}=w_{i}  x_{i}$
>再假设我们本来求误差的方式是这样的：
>$J(x)= \sum_{i=1}^{N}\left(w_{i}  x_{i}-y_{i}\right)^{2}.$
>我们当然想要求得一个$minJ(x)$
>下面我们就要利用**正则化**改进这个误差公式，对这些**参数进行约束**了，也就是一种**对参数的惩罚**
>正则化又很多种，常见的有$l_1$正则和$l_2$正则

>**以下是$l_1$正则，以$l_1$范数为约束**
>$$
>\left\{\begin{array}{l}
\min \sum_{i=1}^{N}\left(w_{i}  x_{i}-y_{i}\right)^{2} \\
\left|w_{1}\right|+\left|w_{2}\right|+… \left|w_{N}\right|\leqslant  m
\end{array}\right.
$$
>假设我们现在只有$w_1$和$w_2$两个系数，画成图是酱紫的![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b3e4230f503b168e74c7e7680d67323b.png#pic_center =160x160)
>
>
>**以下是$l_2$正则，以$l_2$范数为约束**
>$$
>\left\{\begin{array}{l}
\min \sum_{i=1}^{N}\left(w_{i}  x_{i}-y_{i}\right)^{2} \\
w_{1}^2+w_{2}^2+…w_{N}^2\leqslant  m
\end{array}\right.
$$
>假设我们现在只有$w_1$和$w_2$两个系数，画成图是酱紫的![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/9b3c12899f36c53f6145f508cf3e44c7.png#pic_center =160x160)

>聪明的你应该能看出来$l_1$和$l_2$正则的区别叭！！！
>当然，它们的共同点，也是最重要的点，就是在**第二个不等式把参数限制在了一定的范围内**啦，这样就可以**防止一些参数过多过大**了
>在上图中的几何意义就是，在**取一个点在左下角黑框图形的范围内(包括边界），使它和右上角图形的中心的距离最小，那么这个点一定在两个图形相切的地方取到！！！也就是彩色等值线与黑色图形首次相交的地方！！！**
>

>这时候我们引入一个系数$\lambda$表示左下角黑框图形的大小，这个$\lambda$的取值完全由我们来决定（$\lambda \ge  0$）。
>**λ越小，L图形越大,参数可以取得的范围就越广；λ 越大，图形就越小，参数取得的范围就越小。**
>&emsp;
>因此，在有约束条件的情况下：
>$l_1$正则化等价于求$\sum_{i=1}^{N}\left(w_{i}  x_{i}-y_{i}\right)^{2}+\lambda((\left|w_{1}\right|+\left|w_{2}\right|+… \left|w_{N}\right|)-m)$**导数为0**的点
>$l_2$正则化等价于求$\sum_{i=1}^{N}\left(w_{i}  x_{i}-y_{i}\right)^{2}+\lambda((w_{1}^2+w_{2}^2+…w_{N}^2)-m)$**导数为0**的点
>&emsp;
>**为啥可以这么等价呢？？？**
>设式子的第一个部分为$f(x)$，式子的第二个部分为$\lambda g(x)$，整个式子导数为0的点是$\nabla f(x) + \lambda \nabla g(x) = 0$ 的点，也就是**两个式子导数互为相反数**的点。在满足这个条件下，这个点就位于**两个图形相切的地方**。这是几何上的直观证明，数学公式证明我们后面再来和大家详细说说。

>**那么$l_1$正则化和$l_2$正则化有什么区别呢？？**
>&emsp;
>$l_1$函数更利于**稀疏化**（更多的参数取得0值），$l_2$函数**处处可导**，更易计算。
>&emsp;
>为什么$l_1$函数更利于**稀疏化**呢？？比较直观一点理解，因为$l_1$函数有很多突出的角（二维情况下四个，多维情况下更多）**，误差函数与这些角接触的机率会远大于与L其它部位接触的机率**，而在这些角上，会有**很多权值等于0**（比如图中的$w_1$等于0），这就是为什么L1正则化可以产生稀疏模型，进而可以用于特征选择。
>
>&emsp;
>下面再来看给大家看看不同范数的图示叭
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/521ba528ebe57f56ef426143342c710b.png#pic_center)

<br />

### 3.3 优化方法
&emsp;如何找到最优的模型𝑓(𝒙, 𝜃∗) 就成了一个最优化问题．机器学习的训练过程其实就是`最优化问题的求解过程`．


#### 3.3.1 梯度下降法（重要！）
&emsp;在机器学习中，最简单、常用的优化算法就是`梯度下降法`，我们可以不断往`负梯度`的方向搜索（也就是`导数`或者`偏导数`的相反方向），直到达到最低点。
&emsp;梯度下降可以理解为你站在山的某处，想要下山，此时最快的下山方式就是你环顾四周，**哪里最陡峭，朝哪里下山，一直执行这个策略**，在第N个循环后，你就到达了山的最低处。
&emsp;可以看出梯度下降有时得到的是`局部最优解`，如果损失函数是`凸函数`，梯度下降法得到的解就是`全局最优解`。
&emsp;凸函数只有一个最低点，也就是极值点就是最值点，任意两点的连线必然在函数的上方，而非凸函数有多个极值点。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6cbae48dfc18a9315513e648f4fcb759.png#pic_center)

**在二维中表示梯度下降的流程是这样子的：**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f8d73fdab0b4cd288c3a00abdcbc44a1.png#pic_center =300x260)
**在三维中表示梯度下降的流程是这样子的：**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/59abeb30da0d699ea74e22d82a4f06be.gif#pic_center =280x260)
比如下面这幅图`非凸函数`可能找到的就是`局部最优解`，而不是全局最优解。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd84a7ee249b609fed151693e6f35ecd.png#pic_center =380x260)

在高维空间中，`非凸优化的难点`并不在于如何逃离局部最优点，而是如何`逃离鞍点`，鞍点的梯度是0，**但是在一些维度上是最高点，在另一些维度上是最低点**，如图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cccba6bb16250e8c2a89f32a09806a42.png#pic_center)

>解决这个的方法通常为通过在梯度方向上引入随机性


但幸好，我们上面提到的`平方损失函数`就是`凸函数`，可以找到全局最优解。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a6c28311f0923fdcdbeaa95f59510238.jpeg#pic_center =260x260)
**损失函数的迭代公式为**
**$$\Theta^{1}=\Theta^{0}-\alpha \nabla J(\Theta)$$
其中$\Theta^{t}$ 为第 𝑡 次迭代时的参数值，𝛼为搜索步长．在机器学习中，𝛼一般称为学习率。**
>**这个学习率的大小是很有讲究的，如果学习率太小，要很慢才能到达最低点；如果学习率太大，很容易错过最低点。** 
>&emsp;
以下这幅图展现了不同学习率的情况。
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/24d3a29151bcb67491b44d31b6b259bd.png#pic_center =750x300)
>&emsp;
>所以，选择一个**合适的学习率**是很重要的！！！‘

>
>**通常，这个学习率是不断调整的，学习率在一开始要保持大些来保证收敛速度，在收敛到最优点附近时要小些以避免来回振荡。**
>&emsp;
>常见的学习率衰减的方法有：`分段常数衰减`，`逆时衰减`，`指数衰减`，`自然指数衰减`，`余弦衰减`
>&emsp;
>**① 分段常数衰减**：通俗易懂地理解为`阶梯衰减`，**即每经过 $𝑇_1, 𝑇_2, ⋯ , 𝑇_𝑚$ 次迭代将学习率衰减为原来的 $\beta_1, \beta_2, ⋯ , \beta_𝑚$ 倍**，其中 $𝑇_𝑚$ 和 $\beta_𝑚 < 1$ 为根据经验设置的超参数.
>&emsp;
>**② 逆时衰减**
>$$
>\alpha_{t}=\alpha_{0} \frac{1}{1+\beta \times t} \text {, }
>$$其中𝛽为衰减率
>&emsp;
>**③ 指数衰减:**
>$$
>\alpha_{t}=\alpha_{0} \beta^{t},
>$$其中𝛽 < 1为衰减率．
>&emsp;
>**④ 自然指数衰减:**
>$$
>\alpha_{t}=\alpha_{0} e^ {(-\beta \times t)}
>$$其中𝛽 为衰减率．
>&emsp;
>**⑤ 余弦衰减:**
>$$
>\alpha_{t}=\frac{1}{2} \alpha_{0}\left(1+\cos \left(\frac{t \pi}{T}\right)\right),
>$$其中𝑇为总的迭代次数
>&emsp;
>说了这么多学习率衰减的方法，是不是有点头晕呢？那我们来看看不同学习率衰减的图示叭！（假设学习率初始为1）
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/61e22a8b0df37a9f2c49d76832cb63fd.png#pic_center =460x300)

>其实呢，`学习率衰减`只是调整学习率中比较常见简单的一种方法，其它调整学习率的方法还有：`学习率预热`，`周期性学习率调整`，`自适应学习率`，等，这里就不一一介绍辽，给大家看看一些常见的优化方法叭：
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/401a595b47972d8b16f2fa0f05b460ea.png#pic_center =520x260)
#### 3.3.2 数据预处理
&emsp;下面，我们将介绍一种特别常见的数据预处理方法：`归一化`

&emsp;**归一化方法**泛指把数据特征转换为相同尺度的方法，比如把数据特征映射到[0, 1]或[−1, 1]区间内，或者映射为服从均值为0、方差为1的标准正态分布.

>**通俗易懂一点理解，为什么要归一化呢？**
>&emsp;
>比如说我们要衡量一个人，Ta有三个特征: 年龄（岁）、身高（厘米）、体重（千克）
>小明 = [18, 185, 80] &emsp;小红 = [13, 156, 50]
>很明显可以看出，每个数值的单位是不一样的，它们的数据范围也不一样，所以**数值的大小没有可比性**，所以要**把某个特征（某一列）统一压缩映射到一个范围才有计算和比较的意义**。

>下面介绍几种经常使用的归一化方法：`最小最大值归一化 `、`标准化`


**最小最大值归一化** 
&emsp;最小最大值归一化是一种非常简单的归一化方法，**将训练集中某一列数值特征（假设是第  i 列）的值缩放到0和1之间**，公式为：
$$
\frac{x_{\mathrm{i}}-\min \left(\mathrm{x}_{\mathrm{i}}\right)}{\max \left(\mathrm{x}_{\mathrm{i}}\right)-\min \left(\mathrm{x}_{\mathrm{i}}\right)}
$$

**标准化** 
&emsp;标准化也叫Z值归一化，来源于统计上的标准分数．**将每一个维特征都调整为均值为0，方差为1**
&emsp;对于每一维特征$x$，它的均值为：$\mu=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}$，它的方差为：$\sigma^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{(n)}-\mu\right)^{2}$
&emsp;然后，将特征$x^{(n)}$ 减去均值，并除以标准差，得到新的特征值 $\hat{x}^{(n)}$：
$$\hat{x}^{(n)}=\frac{x^{(n)}-\mu}{\sigma}$$&emsp;其中标准差 𝜎 不能为 0．如果标准差为 0，说明这一维特征没有任何区分性，可以直接删掉．
<br/>


#### 3.3.3 提前停止
为了防止`过拟合`问题。除了训练集和测试集之外，有时也会使用一个`验证集`来进行模型选择，测试模型在验证集上是否最优．在每次迭代时，把新得到的模型 𝑓(𝒙; 𝜃) 在验证集上进行测试，并计算错误率．**如果在验证集上的错误率不再下降，就停止迭代。** 这种策略叫`提前停止	`。
下图是`提前停止`的示例
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ef76af728d5e088c01df0cc75d8293b4.png#pic_center =380x260)
#### 3.3.4 超参数优化
在机器学习中，优化又可以分为`参数优化`和`超参数优化`．模型𝑓(𝒙; 𝜃)中的𝜃 称为模型的参数，可以通过优化算法进行学习．除了可学习的参数𝜃 之外，还有一类参数是用来`定义模型结构`或`优化策略`的，这类参数叫作超参数。
>**常见的超参数包括**：`聚类算法中的类别个数`、`梯度下降法中的步长`、`正则化分布的参数`．`项的系数`、`神经网络的层数`、`支持向量机中的核函数`等．超参数的选取一般都是组合优化问题，很难通过优化算法来自动学习．因此，超参数优化根据人的经验不断试错得到．
>
<br/>

## 4.线性模型
&emsp;线性模型是机器学习中应用最广泛的模型，指通过`样本特征的线性组合`来进行预测的模型．给定一个 𝐷 维样本$𝒙 = [𝑥_1, ⋯ , 𝑥_𝐷]^T$，其线性组合函数为
$$
\begin{aligned}
f(\boldsymbol{x} ; \boldsymbol{w}) &=w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{D} x_{D}+b \\
&=\boldsymbol{w}^{\top} \boldsymbol{x}+b
\end{aligned}
$$
>&emsp;其中 $𝒘 = [𝑤_1, ⋯ , 𝑤_𝐷]^T$ 为 𝐷 维的权重向量，𝑏 为偏置。

&emsp;在分类问题中，由于输出目标 𝑦 是一些离散的标签，而 𝑓(𝒙; 𝒘) 的值域为实数，因此无法直接用 𝑓(𝒙; 𝒘) 来进行预测，需要引入**一个非线性的决策函数𝑔(⋅)来预测输出目标**，**其中𝑓(𝒙; 𝒘)也称为判别函数**。
$$y=g(f(\boldsymbol{x} ; \boldsymbol{w}))$$

>&emsp;**一个线性分类模型或线性分类器，是由一个（或多个）线性的判别函数 $𝑓(𝒙; 𝒘) =𝒘^T𝒙 + 𝑏$ 和非线性的决策函数 𝑔(⋅) 组成。**

>**这时候，可能很多同学又有一个问题了，为什么引入了一个非线性的决策函数 𝑔(⋅) ，还要说它是线性模型呢？**
>&emsp;
>其实，区分线性模型和非线性模型主要是看**它的决策边界是否为线性**的，所谓决策边界就是能够把样本正确分类的一条边界。如图所示：
>&emsp;![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bd2f9b42ef18a30cb10f3175d9580b4e.png#pic_center =460x260)



>&emsp;接下来我们继续讲分类问题，其中分类问题大致可以分为`二分类`和`多分类`问题，二分类问题的类别标签 𝑦 只有**两种取值**，通常可以设为 {+1, −1} 或 {0, 1}．多分类问题是指**分类的类别数 𝐶 大于 2．**

&emsp;下面介绍四种不同线性分类模型：`感知器`、`Logistic回归`、`Softmax回归`和`支持向量机`，这些模型的区别主要在于**使用了不同的损失函数**．

<br />

### 4.1 感知器
&emsp;感知器是一种广泛使用的线性分类器，可谓是最简单的人工神经网络，只有一个神经元。
&emsp;它的分类准则是一个简单的符号函数：
$$
\begin{array}{l}
g(f(\boldsymbol{x} ; \boldsymbol{w}))=\operatorname{sgn}(f(\boldsymbol{x} ; \boldsymbol{w}))\\
\triangleq\left\{\begin{array}{cl}
+1 & \text { if } \quad f(\boldsymbol{x} ; \boldsymbol{w})>0 \\
-1 & \text { if } \quad f(\boldsymbol{x} ; \boldsymbol{w})<0
\end{array}\right.
\end{array}
$$
当𝑓(𝒙; 𝒘) = 0时不进行预测，它的结构图如下所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/23a55fa732c97b8419162682bb9d9492.png#pic_center =410x260)

给定  N  个样本的训练集:  $\left\{\left(\boldsymbol{x}^{(n)}, y^{(n)}\right)\right\}_{n=1}^{N} ,$ 其中  $y^{(n)} \in\{+1,-1\}$, 感知器学习算法试图找到一组参数  $\boldsymbol{w}^{*}$ , 使得对于每个样本  $\left(\boldsymbol{x}^{(n)}, y^{(n)}\right)$  有$$y^{(n)} \boldsymbol{w}^{*^{\top}} \boldsymbol{x}^{(n)}>0, \quad \forall n \in\{1, \cdots, N\}$$

因此感知器的损失函数为：
$$
\mathcal{L}(\boldsymbol{w} ; \boldsymbol{x}, y)=\max \left(0,-y \boldsymbol{w}^{\top} \boldsymbol{x}\right)
$$
采用梯度下降法进行更新，其每次更新的梯度为：
$$
\frac{\partial \mathcal{L}(\boldsymbol{w} ; \boldsymbol{x}, y)}{\partial \boldsymbol{w}}=\left\{\begin{array}{ll}
0 & \text { if } \quad y \boldsymbol{w}^{\top} \boldsymbol{x}>0 \\
-y \boldsymbol{x} & \text { if } \quad y \boldsymbol{w}^{\top} \boldsymbol{x}<0
\end{array}\right.
$$
>下图给出了感知器参数学习的更新过程，其中被圈中的点为随机选取的要学习的点，红色实心点为正例，蓝色空心点为负例．黑色箭头表示当前的权重向量，红色虚线箭头表示权重的更新方向．
>&emsp;
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/43411da416b05cbe84989cfb3aeeb8cc.png#pic_center =460x440)

<br>


### 4.2 Logistic回归
&emsp;Logistic 回归是一种常用的处理二分类问题的线性模型。我们将采用𝑦 ∈ {0, 1}以符合Logistic回归的描述习惯．

&emsp;**为了把线性函数的值域从实数区间“挤压”到了(0, 1)之间，可以用来表示概率，我们将使用一个激活函数𝑔(⋅)** 
$$
p(y=t \mid \boldsymbol{x})=g(f(\boldsymbol{x} ; \boldsymbol{w}))
$$
>在Logistic回归中，我们使用`Logistic函数`来作为激活函数
>&emsp;
>为什么要用`Logistic函数`呢？
>在分类问题中，我们可以用最简单的`单位阶跃函数`来作为激活函数𝑔(⋅)：
>$$y=\left\{\begin{array}{cc}
0, & z<0 \\
0.5, & z=0 \\
1, & z>0
\end{array}\right.$$它长这个样子：
>&emsp;
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7e0e42363ea8aabf3af7220e0e551bc3.png#pic_center =400x200)
>但是很可惜，它有一个很大的缺点：**该函数在跳跃点上从0瞬间跳跃到1（不连续、不可微）**，所以我们想找到一个在一定程度上近似单位阶跃函数的“替代函数”，并希望它单调可微。
>
>&emsp;
>
>这时候，`Logistic函数`就登场了，其公式为：
>$$
>\sigma(z)=\frac{1}{1+e^{-z}}
>$$
>它长酱紫, 它在$z=0$附近变化会很陡：
>&emsp;
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/65fc48c10117e5083c7ca11d7c82d871.png#pic_center =380x260)
>&emsp;
>其导数为：
>$$
>\sigma^{\prime}(z)=\sigma(z)(1-\sigma(z))
>$$

>我们来对比一下`单位阶跃函数`和`Logistic函数`：
>&emsp;
>![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c8e6669d4059bf4ff233801076896d84.png#pic_center =420x200)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/053471392b960510f1996538652d6939.png#pic_center =420x200)
>长相和效果还蛮像的对叭


标签𝑦 = 1（也就是正例）的后验概率为:
$$
\begin{aligned}
p(y=1 \mid \boldsymbol{x}) &=\sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)
 \triangleq \frac{1}{1+e^{- \left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)}},
\end{aligned}
$$
标签𝑦 = 0（反例）的后验概率为
$$
\begin{aligned}
p(y=0 \mid \boldsymbol{x}) &=1-p(y=1 \mid \boldsymbol{x})=\frac{e^{-\left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)}}{1+e^{- \left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)}}
\end{aligned}
$$
然后给大家变个魔法，把上面的式子变形一下可以得到：
$$
\boldsymbol{w}^{\top} \boldsymbol{x}=\log \frac{p(y=1 \mid \boldsymbol{x})}{1-p(y=1 \mid \boldsymbol{x})}
=\log \frac{p(y=1 \mid x)}{p(y=0 \mid x)}
$$
我们仔细看看，其中 $\frac{p(y=1 \mid x)}{p(y=0 \mid x)}$ 为样本𝒙为正反例后验概率的比值，称为`几率`，`几率的对数`称为`对数几率`，因此 Logistic 回归可以看作预测值为`“标签的对数几率”`的线性回归模型。
****

**Logistic 回归采用交叉熵作为损失函数，并使用梯度下降法来对参数进行优化．**

设一个样本标签为1的`预测概率`为：
$$
\hat{y}^{(n)}=\sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right),
$$
设一个样本标签为1的`真实概率`为：
$$
p_{r}\left(y^{(n)}=1 \mid x^{(n)}\right)=y^{(n)}
$$
使用`交叉熵损失函数`，假设有`N`个样本,公式为：
$$R(𝒘) = =-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)} \log \hat{y}^{(n)}+\left(1-y^{(n)}\right) \log \left(1-\hat{y}^{(n)}\right)\right) .$$

ℛ(𝒘)关于参数𝒘的`偏导数`为：
$$
\begin{aligned}
\frac{\partial \mathcal{R}(\boldsymbol{w})}{\partial \boldsymbol{w}} &=-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)} \frac{\hat{y}^{(n)}\left(1-\hat{y}^{(n)}\right)}{\hat{y}^{(n)}} \boldsymbol{x}^{(n)}-\left(1-y^{(n)}\right) \frac{\hat{y}^{(n)}\left(1-\hat{y}^{(n)}\right)}{1-\hat{y}^{(n)}} \boldsymbol{x}^{(n)}\right) \\
&=-\frac{1}{N} \sum_{n=1}^{N}\left(y^{(n)}\left(1-\hat{y}^{(n)}\right) \boldsymbol{x}^{(n)}-\left(1-y^{(n)}\right) \hat{y}^{(n)} \boldsymbol{x}^{(n)}\right) \\
&=-\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(y^{(n)}-\hat{y}^{(n)}\right) .
\end{aligned}
$$
>𝑦̂ 为 Logistic 函 数，求导公式在上面。这时的 $log$ 是以 $e$ 为底的

然后我们就采取`梯度下降法`进行迭代更新了，初始化 𝒘0 ← 0，然后通过下式来迭代更新参数：
$$
\boldsymbol{w}_{t+1} \leftarrow \boldsymbol{w}_{t}+\alpha \frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(y^{(n)}-\hat{y}_{w_{t}}^{(n)}\right),
$$

### 4.3 Softmax回归
`Softmax 回归`，也称为多项或多类的Logistic回归，**是Logistic回归在多分类问题上的推广．** 

对于多类问题，类别标签𝑦 ∈ {1, 2, ⋯ , 𝐶}可以有𝐶 个取值．给定一个样本𝒙，Softmax回归预测的属于类别𝑐的条件概率为
$$
\begin{aligned}
p(y=c \mid \boldsymbol{x}) &=\operatorname{softmax}\left(\boldsymbol{w}_{c}^{\top} \boldsymbol{x}+b\right) \\
&=\frac{e^{ \left(\boldsymbol{w}_{c}^{\top} \boldsymbol{x}+b\right)}}{\sum_{c^{\prime}=1}^{C}e^{ \left(\boldsymbol{w}_{c^{\prime}}^{\top} \boldsymbol{x}+b\right)}},
\end{aligned}
$$
Softmax回归的决策函数可以表示为:
$$
\begin{array}{l}
\hat{y}=\underset{c=1}{\overset{C}{\arg \max }}  p(y=c \mid \boldsymbol{x})\\
=\underset{c=1}{\overset{C}{\arg \max }}\left( \boldsymbol{w}_{c}^{\top} \boldsymbol{x}+b\right) .
\end{array}
$$
>这时候你可能又要问了，**这个argmax到底是个什么东西呀？**
>&emsp;
>根据定义，argmax()是一种函数，是对函数求参数(集合)的函数，也就是**求自变量最大的函数**。
>&emsp;
>**举个例子：**
>$f(x=1)=20$&emsp;$f(x=2)=25$&emsp;$f(x=3)=22$
>可以得出：
> $y=\max (f(x))=25$
> $y=\operatorname{argmax}(f(x))=2$
>&emsp;
>回归到问题本身，**我们为什么用argmax而不用max呢**？
>因为我们要得到的答案是这个样本是哪一类的，而不是它属于这一类的概率是多少，所以我们要用argmax来得到概率最大值的自变量。


采用交叉熵损失函数，Softmax回归模型的风险函数为：
$$
\begin{aligned}
\mathcal{R}(\boldsymbol{W}) &=
-\frac{1}{N} \sum_{n=1}^{N}\left(\boldsymbol{y}^{(n)}\right)^{\boldsymbol{\top}} \log \hat{\boldsymbol{y}}^{(n)}
\end{aligned}
$$风险函数ℛ(𝑾 )关于𝑾 的梯度为
$$
\frac{\partial \mathcal{R}(\boldsymbol{W})}{\partial \boldsymbol{W}}=-\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(\boldsymbol{y}^{(n)}-\hat{\boldsymbol{y}}^{(n)}\right)^{\top}
$$
采用梯度下降法，Softmax回归的训练过程为：初始化$𝑾_0 ← 0$，然后通过下式进行迭代更新：
$$
\boldsymbol{W}_{t+1} \leftarrow \boldsymbol{W}_{t}+\alpha\left(\frac{1}{N} \sum_{n=1}^{N} \boldsymbol{x}^{(n)}\left(\boldsymbol{y}^{(n)}-\hat{\boldsymbol{y}}_{W_{t}}^{(n)}\right)^{\top}\right)
$$

### 4.4 支持向量机
`支持向量机（SVM）`是一个经典的二分类算法，其找到的分割超平面具有更好的鲁棒性。


给定一个二分类器数据集$\mathcal{D}=\left\{\left(\boldsymbol{x}^{(n)}, y^{(n)}\right)\right\}_{n=1}^{N}$，其中$y_{n} \in\{+1,-1\}{\scriptsize }$,如果两类样本是线性可分的，即存在一个超平面 
$$
\boldsymbol{w}^{\top} \boldsymbol{x}+b=0$$将两类样本分开，那么对于每个样本都有$y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)>0$，数据集𝒟 中每个样本$𝒙^{(𝑛)}$到分割超平面的距离为：
$$\gamma^{(n)}=\frac{\left|\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right|}{\|\boldsymbol{w}\|}=\frac{y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)}{\|\boldsymbol{w}\|} .$$
我们定义间隔 𝛾 为整个数据集 𝐷 中所有样本到分割超平面的最短距离：$$\gamma=\min _{n} \gamma^{(n)}$$如下图所示，我们可以找到很多点来分隔，但能将训练样本分开的划分超平面可能有很多，我们要找到哪一个呢？
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/051e0a2a2fdcafa003e567ea13e5dd4d.png#pic_center =360x300)

显然，我们要找到一个最稳定的平面。**如果间隔𝛾越大**，其分割超平面对两个数据集的划分越稳定，不容易受噪声等因素影响．支持向量机的目标是寻找一个超平面(𝒘∗, 𝑏∗)使得𝛾最大，即
$$
\begin{aligned}
\max _{\boldsymbol{w}, b} & \gamma \\
\text { s.t. } & \frac{y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)}{\|\boldsymbol{w}\|} \geq \gamma, \forall n \in\{1, \cdots, N\} .
\end{aligned}
$$由于同时缩放𝒘 → 𝑘𝒘和𝑏 → 𝑘𝑏不会改变样本$𝒙^{(𝑛)}$ 到分割超平面的距离，我们可以限制‖𝒘‖ ⋅ 𝛾 = 1，则上述公式等价于
$$
\begin{aligned}
\max _{\boldsymbol{w}, b} & \frac{1}{\|\boldsymbol{w}\|^{2}} \\
\text { s.t. } & y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right) \geq 1, \forall n \in\{1, \cdots, N\} .
\end{aligned}
$$**数据集中所有满足$y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)=1$ 的样本点，都称为支持向量**

如图支持向量机的最大间隔分割超平面的示例，其中轮廓线加粗的样本点为支持向量．
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f5f66c00192a2fcf0ee84c42a53ed631.png#pic_center =360x300)
#### 4.4.1 参数学习
为了找到最大间隔分割超平面，将目标函数写为`凸优化问题`
$$\begin{aligned}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & 1-y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right) \leq 0, \quad \forall n \in\{1, \cdots, N\} .
\end{aligned}$$
>**这时候可能又有疑问了，什么是凸优化问题呢？**
>&emsp;
>先记住凸优化问题一个重要的特点：**局部最优是全局最优**
>&emsp;
>凸优化问题是想要找到一个酱紫的解：
>$$
>\begin{array}{ll}
min  & f_0(x)\\
\text { subject to } & f_{i}(x) \leq 0, \quad i=1, \cdots, m \\
& h_{i}(x)=0, \quad i=1, \cdots, p
\end{array}$$
>$下面两个式子分别表示不等式约束和等式约束$
>$\text {其中 } f_0 \text { 为凸函数， } f_{i} \text { 为凸函数， } h_{i} \text { 为仿射函数， } x \text { 为优化变量。 }$
>$注意上面的 min 以及约束条件的符号均要符合规定！！！$
>&emsp;
>仿射函数：$: a^{T} x+b$，它同时满足凸函数和凹函数


>**在上面学习正则化的时候，我们讲了拉格朗日乘数法的几何意义，下面我们来数学证明一下。**
>&emsp;
>根据上面的在约束条件求最值的条件下，我们化成`拉格朗日函数`
>$$L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})=f_{0}(\boldsymbol{x})+\sum \lambda_{i} f_{i}(\boldsymbol{x})+\sum v_{i} h_{i}(\boldsymbol{x})$$
>然后我们化成`原问题`（即先将$\lambda$和$v$看成变量，$x$看成常量求最大值，再把$x$看成变量求最小值）：
>$$
>\begin{array}{l}
\min _{\boldsymbol{x}} \max _{\lambda, v} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v}) \\
\text { s.t. } \lambda \geq 0
\end{array}$$
>然后我们**证明一下**为什么可以化成这样：
>$$\left\{\begin{array}{l}
\boldsymbol{x} \text { 不在可行域内: } \max _{\lambda, v} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})=f_{0}(\boldsymbol{x})+\infty+\infty=\infty \\
\boldsymbol{x} \text { 在可行域内 }: \max _{\lambda, v} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})=f_{0}(\boldsymbol{x})+0+0=f_{0}(\boldsymbol{x})
\end{array}\right.$$$$\min _{\boldsymbol{x}} \max _{\lambda, \boldsymbol{v}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})=\min _{\boldsymbol{x}}\left\{f_{0}(\boldsymbol{x}), \infty\right\}= \min  f_0(x)$$

>下面再来介绍一下`对偶函数`和`对偶问题`：
>&emsp;
>**对偶函数**：
>$$g(\boldsymbol{\lambda}, \boldsymbol{v})=\min _{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})$$
>**对偶问题**（即先将$x$看成变量，把$\lambda$和$v$看成常量求最小值，再把$\lambda$和$v$看成变量求最大值：
>$$
>\begin{array}{l}
\max _{\lambda, v} g(\boldsymbol{\lambda}, \boldsymbol{v})=\max _{\lambda, v} \min _{x} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v}) \\
\text { s.t. } \lambda \geq \mathbf{0}
\end{array}$$上式等价于：
$$
\begin{array}{ll}
\max _{\boldsymbol{\lambda}, \boldsymbol{v}} g(\boldsymbol{\lambda}, \boldsymbol{v}) \\
\text { s.t.} &\nabla_{\boldsymbol{x}} L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{v})=\mathbf{0} \\
&\boldsymbol{\lambda} \geq \mathbf{0}
\end{array}$$
>看完后是不是觉得像上面的**原问题反过来换种方法**求解嘞？
>对偶问题有一个重要的特性：**无论原问题是什么，对偶问题都是一个凸问题**！！（注意，原问题化成对偶问题的方向是单向的！）
>&emsp;
>我超！！这也太神奇了吧，**为什么无论什么问题的对偶问题都是凸问题呢**？？
>我们先来看一下这个对偶函数，我们假设$x^*$是找到的拉格朗日函数的最小值：
>$$g(\boldsymbol{\lambda}, \boldsymbol{v})=f_{0}\left(x^{\star}\right)+\sum \lambda_{i} f_{i}\left(x^{\star}\right)+\sum v_{i} h_{i}\left(x^{\star}\right)$$这时候只有$\lambda$和$v$是变量，这时候$g(\boldsymbol{\lambda}, \boldsymbol{v})$就是一阶线性关系的，它可以看成是一条直线，这时候它既是凹函数又是凸函数，求它的最大值就是凸优化问题。
>
>

>我们很容易得到，**原问题的解是大于等于对偶问题的解的**（弱对偶定理），即：
>$$\min _{x} \max _{\lambda, v} L(x, \lambda, v) \geq \max _{\lambda, v} \min _{x} L(x, \lambda, v)$$
>当然，我们肯定想要**原问题的解是等于对偶问题的解的**（强对偶定理），这时候我们就要用KKT条件进行约束了：
>&emsp;
>**KKT条件：**
>$$
> \left.\begin{array}{l}f_{i}(\boldsymbol{x}) \leq 0 \\ h_{i}(\boldsymbol{x})=0\end{array}\right\}  原问题可行条件
> $$$$\left.\begin{array}{l}\nabla_{x} L(x, \lambda, v)=0 \\ \lambda \geq 0\end{array}\right\}  对偶可行条件$$$$\lambda_{i} f_{i}(\boldsymbol{x})=0 \quad \text {互补松弛条件 }$$

使用`拉格朗日乘数法`，将公式化为`拉格朗日函数`：
$$\Lambda(\boldsymbol{w}, b, \lambda)=\frac{1}{2}\|\boldsymbol{w}\|^{2}+\sum_{n=1}^{N} \lambda_{n}\left(1-y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)\right)$$
其中$\lambda_1 ≥ 0, ⋯ , \lambda_𝑁 ≥ 0$为`拉格朗日乘数`．计算Λ(𝒘, 𝑏, 𝜆)关于𝒘和𝑏的`导数`，并令其等于0，得到
$$
\begin{aligned}
\boldsymbol{w} &=\sum_{n=1}^{N} \lambda_{n} y^{(n)} \boldsymbol{x}^{(n)}, \\
0 &=\sum_{n=1}^{N} \lambda_{n} y^{(n)}
\end{aligned}
$$


公式代换，得到`拉格朗日对偶函数`：
$$\Gamma(\lambda)=-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} \lambda_{m} \lambda_{n} y^{(m)} y^{(n)}\left(\boldsymbol{x}^{(m)}\right)^{\top} \boldsymbol{x}^{(n)}+\sum_{n=1}^{N} \lambda_{n} .$$

**根据KKT条件中的互补松弛条件**，最优解满足$\lambda_{n}^{*}\left(1-y^{(n)}\left(\boldsymbol{w}^{* \top} \boldsymbol{x}^{(n)}+b^{*}\right)\right)=0$

在计算出 𝜆∗ 后，根据公式 $\boldsymbol{w} =\sum_{n=1}^{N} \lambda_{n} y^{(n)} \boldsymbol{x}^{(n)}$ 计算出最优权重 𝒘∗，最优偏置 𝑏∗ 可以通过任选一个支持向量(𝒙,̃𝑦)̃计算得到
$$b^{*}=\tilde{y}-\boldsymbol{w}^{*^{\top}} \tilde{\boldsymbol{x}} .$$
**最优参数的支持向量机的决策函数为**
$$
\begin{aligned}
f(\boldsymbol{x}) &=\operatorname{sgn}\left(\boldsymbol{w}^{*^{\top}} \boldsymbol{x}+b^{*}\right) \\
&=\operatorname{sgn}\left(\sum_{n=1}^{N} \lambda_{n}^{*} y^{(n)}\left(\boldsymbol{x}^{(n)}\right)^{\top} \boldsymbol{x}+b^{*}\right)
\end{aligned}$$

#### 4.4.2 核函数
支持向量机还有一个重要的优点是可以使用`核函数`隐式地将样本从原始特征空间**映射到更高维的空间**，并解决原始特征空间中的**线性不可分问题**。面例如图中左边的"异或“问题就不是线性可分的，但将其映射到如图右的高维空间是可以变成线性可分的。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3573649577bd7e3479ae678913bbf1da.png#pic_center)
再比如这样：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/accf1bb39623335d2950f7dd1db693cb.png#pic_center =360x260)

比如在一个变换后的特征空间𝜙中，支持向量机的决策函数为：
$$\begin{aligned}
f(\boldsymbol{x}) &=\operatorname{sgn}\left(\boldsymbol{w}^{* \top} \phi(\boldsymbol{x})+b^{*}\right) \\
&=\operatorname{sgn}\left(\sum_{n=1}^{N} \lambda_{n}^{*} y^{(n)} k\left(\boldsymbol{x}^{(n)}, \boldsymbol{x}\right)+b^{*}\right)
\end{aligned}
$$
其中 $𝑘(𝒙, 𝒛) = \phi(𝒙)^T\phi(𝒛)$ 为核函数．
>"**核函数选择**"成为支持向量机的最大变数.若核函数选择不合适，则意味着将样本映射到了一个不合适的特征空间，很可能导致性能不佳。

以下是一些常用的`核函数`：
| 名称| 表达式 |参数 |
|--|--|--|
| 线性核 | $\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}$ ||
| 多项式核 |$\kappa\left(x_{i}, x_{j}\right)=\left(x_{i}^{\mathrm{T}} x_{j}\right)^{d}$  |$d\ge1$为多项式的次数|
| 高斯核 | $\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(-\frac{\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}}{2 \sigma^{2}}\right)$|$σ>0$ 为高斯核的带宽(width)|
|  拉普拉斯核|$\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(-\frac{\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|}{\sigma}\right)$  |$σ>0$|
| Sigmoid核 |$\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\tanh \left(\beta \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}+\theta\right)$  |tanh 为双曲正切函数，$\beta>0, \theta<0$ |
#### 4.4.3 软间隔
在支持向量机的优化问题中，约束条件比较严格．如果训练集中的样本在特征空间中不是线性可分的，就无法找到最优解．为了能够容忍部分不满足约束的样本，我们可以引入松弛变量 𝜉，将优化问题变为
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{n=1}^{N} \xi_{n}\\
s.t.   \quad1-y^{(n)}\left(\boldsymbol{w}^{\top} \boldsymbol{x}^{(n)}+b\right)-\xi_{n} \leq 0,\quad \forall n \in\{1, \cdots, N\}  \\ \xi_{n} \geq 0, \quad \forall n \in\{1, \cdots, N\} $$

如图所示：红色圈出了一些不满足约束的样本.
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/49d7afd23d464969a7c0184604462bed.png#pic_center)
### 4.5 损失函数对比
&emsp;损失函数的不同，会让模型它们在实际任务上的表现存在一定的差异。
&emsp;**下图给出了不同损失函数的对比**．对于二分类来说，当𝑦𝑓(𝒙; 𝒘) > 0时，分类器预测正确，并且𝑦𝑓(𝒙; 𝒘)越大，模型的预测越正确；当𝑦𝑓(𝒙; 𝒘) < 0时，分类器预测错误，并且𝑦𝑓(𝒙; 𝒘)越小，模型的预测越错误．因此，一个好的损失函数应该随着𝑦𝑓(𝒙; 𝒘)的增大而减少．
&emsp;从下图中看出，除了平方损失，其他损失函数都比较适合于二分类问题．
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5a1881497fc1871b3cc43d33d78ac603.png#pic_center)

## 5. 评价指标
为了衡量一个机器学习模型的好坏，需要给定一个测试集，用模型对测试集
中的每一个样本进行预测，并根据预测结果计算评价分数．

对于分类问题，常见的评价标准有`准确率`、`精确率`、`召回率`和`F值`等．

给定测试集$\mathcal{T}=\left\{\left(\boldsymbol{x}^{(1)}, y^{(1)}\right), \cdots,\left(\boldsymbol{x}^{(N)}, y^{(N)}\right)\right\}$.假设标签 $y^{(n)} \in\{1, \cdots, C\}$，，用学习好的模型𝑓(𝒙; 𝜃∗)对测试集中的每一个样本进行预测，结果为$\left\{\hat{y}^{(1)}, \cdots, \hat{y}^{(N)}\right\}$．

**`准确率`**  最常用的评价指标为准确率：
$$\mathcal{A}=\frac{1}{N} \sum_{n=1}^{N} I\left(y^{(n)}=\hat{y}^{(n)}\right)$$
其中𝐼(⋅)为指示函数．

<br>

**`错误率`** 和准确率相对应的就是错误率

$$\begin{aligned}
\mathcal{E} & =1-\mathcal{A} \\
& =\frac{1}{N} \sum_{n=1}^{N} I\left(y^{(n)} \neq \hat{y}^{(n)}\right)
\end{aligned}
$$

**`精确率和召回率 `**

准确率是所有类别整体性能的平均，如果希望对每个类都进行性能估计，就需要计算`精确率`和`召回率`

**对于类别𝑐来说，模型在测试集上的结果可以分为以下四种情况：**

（1） **`真正例（True Positive，TP）`：一个样本的真实类别为𝑐并且模型正确地预测为类别𝑐．** 这类样本数量记为

$$T P_{c}=\sum_{n=1}^{N} I\left(y^{(n)}=\hat{y}^{(n)}=c\right)$$

（2） **`假负例（False Negative，FN）`：一个样本的真实类别为𝑐，模型错误地预测为其他类**. 这类样本数量记为

$$F N_{c}=\sum_{n=1}^{N} I\left(y^{(n)}=c \wedge \hat{y}^{(n)} \neq c\right)$$

（3） **`假正例（False Positive，FP）`：一个样本的真实类别为其他类，模型错误地预测为类别𝑐**．这类样本数量记为

$$F P_{c}=\sum_{n=1}^{N} I\left(y^{(n)} \neq c \wedge \hat{y}^{(n)}=c\right)$$

（4） **`真负例（True Negative，TN）`：一个样本的真实类别为其他类，模型也预测为其他类．这类样本数量记为𝑇𝑁𝑐**．对于类别𝑐来说，这种情况一般不需要关注．



有 TP+FP+TN+FN=样例总数,  这四种情况的关系可以用`混淆矩阵`来表示．
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/56417d5e16a751a5f3fe5ddc13b4e450.png#pic_center)
**`精确率（Precision）`，也叫精度或查准率，类别 𝑐 的查准率是所有预测为类别𝑐的样本中预测正确的比例：**

$$\mathcal{P}_{c}=\frac{T P_{c}}{T P_{c}+F P_{c}}$$

**`召回率（Recall）`，也叫查全率，类别𝑐的查全率是所有真实标签为类别𝑐的样本中预测正确的比例：**

$$\mathcal{R}_{c}=\frac{T P_{c}}{T P_{c}+F N_{c}}$$

**查准率和查全率是一对矛盾的度量**.一般来说，查准率高时，查全率往往偏低;而查全率高时，查准率往往偏低.

**`F值（F Measure）`是一个综合指标，为精确率和召回率的调和平均：**

$$\mathcal{F}_{c}=\frac{\left(1+\beta^{2}\right) \times \mathcal{P}_{c} \times \mathcal{R}_{c}}{\beta^{2} \times \mathcal{P}_{c}+\mathcal{R}_{c}}$$

其中 𝛽 用于平衡精确率和召回率的重要性，一般取值为1．𝛽 = 1时的F值称为F1值，是精确率和召回率的调和平均．

$$\mathcal{F}_{1}=\frac{2\times \mathcal{P}_{c} \times \mathcal{R}_{c}}{ \mathcal{P}_{c}+\mathcal{R}_{c}}$$


<br>

## 6. 写在最后
**好了，今天的机器学习的内容就分享到这里了**
哈哈，是不是看到这一堆公式就头昏脑胀的。**但我告诉你，这只是开始。**
继续加油叭！！！
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b0142313cfcd9fbd9f73912fbdbae51c.png#pic_center)

