# Hello World
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
# print(tf.__version__)
# print(sess.run(hello))

# 节点（操作）和边（张量）
# （1）张量
# 张量是TensorFlow的主要数据结构，用于操作计算图。
# 一个张量（Tensor）可以简单地理解为任意维的数组，张量的秩表示其维度数量。张量的秩不同，名称也不相同。
# a、标量：维度为0的Tensor，也就是一个实数
# b、向量：维度为1的Tensor
# c、矩阵：维度为2的Tensor
# d、张量：维度达到及超过3的Tensor

# 创建张量有以下主要4种方法：
# a、创建固定张量
# 创建常数张量：
constant_ts=tf.constant([1,2,3,4,5])

# 创建零张量：
# zero_ts=tf.zeros([row_dim, col_dim])

# 创建单位张量：
# ones_ts=tf.ones([row_dim,col_dim])

# 创建张量，并用常数填充：
# filled_ts=tf.fill([row_dim,col_dim],123)

# b、创建相似形状张量
# 创建相似的零张量：
zeros_like=tf.zeros_like(constant_ts)


# 创建相似的单位张量：
ones_like=tf.ones_like(constant_ts)

# c、创建序列张量
# 指定起止范围
linear_ts=tf.linspace(start=.0, stop=2, num=6)
# 结果返回[0.0,0.4,0.8,1.2,1.6,2.0]，注意该结果包括stop值

# 指定增量
seq_ts=tf.range(start=4, limit=16, delta=4)
# 结果返回[4,8,12]，注意该结果不包括stop值

# d、随机张量
# 生成均匀分布的随机数
# randunif_ts=tf.random_uniform([row_dim,col_dim],minval=0,maxval=1)
# 结果返回从minval（包含）到maxval（不包含）的均匀分布的随机数

# 生成正态分布的随机数
# randnorm_ts=tf.random_normal([row_dim,col_dim],mean=0.0,stddev=1.0)
# 其中mean表示平均值，stddev表示标准差

# （2）占位符和变量
# 占位符和变量是使用TensorFlow计算图的关键工具，两者是有区别的
# a、变量：是TensorFlow算法中的参数，通过调整这些变量的状态来优化模型算法；
# b、占位符：是TensorFlow对象，用于表示输入输出数据的格式，允许传入指定类型和形状的数据。

# 创建变量
# 通过tf.Variable()函数封装张量来创建变量，例如：
# my_var=tf.Variable(tf.zeros([row_dim,col_dim]))

# 【注意】声明变量后需要进行初始化才能使用，最常使用以下函数一次性初始化所有变量，使用方式如下：
# init_op=tf.global_variables_initializer()

# 创建占位符
# 占位符仅仅是声明数据位置，也即先占个位，后面在会话中通过feed_dict传入具体的数据。示例代码如下：
a=tf.placeholder(tf.float32,shape=[1,2])
b=tf.placeholder(tf.float32,shape=[1,2])
adder_node = a + b   #这里的“+”是tf.add(a,b)的简洁表达
# print(sess.run(adder_node,feed_dict={a:[[2,4]],b:[[5.2,8]]}))

c = 4.0
d = tf.rsqrt(c)
print(sess.run(d))
