# 2.怎样使用激励函数
# (0).创建一个会话，调用默认计算图
import tensorflow as tf
sess = tf.Session()

# (1)ReLU函数
df = tf.nn.relu([-5., 0., 5., 10.])
print(sess.run(df))

# (2)ReLU6函数
df = tf.nn.relu6([-5., 0., 5., 10.])
print(sess.run(df))

# (3)Leaky ReLU函数
df = tf.nn.leaky_relu([-3., 0., 5.])
print(sess.run(df))

# (4)sigmoid函数
df = tf.nn.sigmoid([-1., 0., 1.])
print(sess.run(df))

# (5)tanh函数
df = tf.nn.tanh([-1., 0., 1.])
print(sess.run(df))

# （6)ELU函数
df = tf.nn.elu([-1., 0., 1.])
print(sess.run(df))

# (7)softsign函数
df = tf.nn.softsign([-1., 0., 1.])
print(sess.run(df))

# (8)softplus函数
df = tf.nn.softplus([-1., 0., 1.])
print(sess.run(df))