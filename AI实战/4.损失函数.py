# 1.回归模型的损失函数
import tensorflow as tf
sess = tf.Session()
y_pred = tf.linspace(-1., 1., 100)
y_target = tf.constant(0.)

# (1)L1正则损失函数
loss_l1_vals = tf.abs(y_pred - y_target)
loss_l1_out = sess.run(loss_l1_vals)
print(loss_l1_out)

# (2)L2正则损失函数
# L2损失
loss_l2_vals = tf.square(y_pred - y_target)
loss_l2_out = sess.run(loss_l2_vals)

# 均方误差
loss_mse_vals = tf.reduce.mean(tf.square(y_pred - y_target))