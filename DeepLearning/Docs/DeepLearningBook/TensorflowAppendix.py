import tensorflow as tf
import numpy as np

# We construct the graph and prepare the operation.
# We can have operations with dependencies, tensorflow will figure it out.
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.constant([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	col_sum = tf.reduce_sum(tf_x, axis=0)
	col_sum_times_2 = col_sum * 2
print('tf_x: ', tf_x)
print('col_sum: ', col_sum)

# Now we execute the operation
with tf.Session(graph=g) as sess:
	mat, csum = sess.run([tf_x, col_sum])
print('mat: ', mat)
print('csum: ', csum)

# We don't have to run the dependencies
with tf.Session(graph=g) as sess:
	csum2 = sess.run(col_sum_times_2)
print('csum2: ', csum2)

# We define a variable tensor, we must initialize variables in the active session
g=tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	tf_x = tf_x + x
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(tf_x)
print(result)

# Evaluating the same graph twice does not affect the numerical values fetched
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(tf_x)
	result = sess.run(tf_x)
print(result)

# To assign new values to a variable, we use tensorflow.assign
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(update_tf_x)
	result = sess.run(update_tf_x)
print(result)

# Placeholder variables allow us to feed the graph in an active session
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.placeholder(dtype=tf.float32, shape=(3,2))
	output = tf.matmul(tf_x, tf.transpose(tf_x))
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	np_ary = np.array([[3., 4.], [5., 6.], [7., 8.]])
	feed_dict = {tf_x: np_ary}
	print(sess.run(output, feed_dict=feed_dict))

# We can save and reuse our tensorflow models
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver([tf_x])
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(update_tf_x)
	saver.save(sess, save_path='./my-model.ckpt')

# We can every 10 iterations for example
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver([tf_x], max_to_keep=3)
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	saver.save(sess, save_path='./my-model.ckpt')
	for epoch in range(100):
		result = sess.run(update_tf_x)
		if not epoch % 10:
			saver.save(sess, save_path='./my-model_multiple_ckpts.ckpt', global_step=epoch)

# We can restore the variables
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver([tf_x], max_to_keep=3)
with tf.Session(graph=g) as sess:
	saver.restore(sess, save_path='./my-model.ckpt')
	result = sess.run(update_tf_x)
	print(result)

with tf.Session(graph=g) as sess:
	saver.restore(sess, save_path='./my-model_multiple_ckpts.ckpt-90')
	result = sess.run(update_tf_x)
	print(result)

# We can give a name to Variables, not doing so might create problems such as this one
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver()
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(update_tf_x)
	saver.save(sess, save_path='./my-model.ckpt')

# We didn't name our variables so tf_y was restored into tf_x because they are switched
g = tf.Graph()
with g.as_default() as g:
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], dtype=tf.float32)
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver()
with tf.Session(graph=g) as sess:
	saver.restore(sess, save_path='./my-model.ckpt')
	result = sess.run(update_tf_x)
	print(result)

# We fix the problem by naming our variables
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], name='tf_x_0', dtype=tf.float32)
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], name='tf_y_0', dtype=tf.float32)
	x = tf.constant(1., dtype=tf.float32)
	update_tf_x = tf.assign(tf_x, tf_x + x)
	saver = tf.train.Saver()
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	result = sess.run(update_tf_x)
	saver.save(sess, save_path='./my-model.ckpt')

# With gpu version, tensorflow uses gpu by default, we can specify the processing unit
# with tf.Session() as sess:
	# with tf.device('/gpu:1')

# We can list all devices
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
tf.test.gpu_device_name()

# We can use tensorboard to visualize our graph / debug our model
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], name='tf_x_0', dtype=tf.float32)
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], name='tf_y_0', dtype=tf.float32)
	output = tf_x + tf_y
	output = tf.matmul(tf.transpose(tf_x), output)
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	# run command tensorboard --logdir logs/1 to check model (0.0.0.0:6006)
	file_writer = tf.summary.FileWriter(logdir='logs/1', graph=g)
	result = sess.run(output)
	print(result)

# Name scopes let us organize our graph
g = tf.Graph()
with g.as_default() as g:
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], name='tf_x_0', dtype=tf.float32)
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], name='tf_y_0', dtype=tf.float32)
	with tf.name_scope('addition'):
		output = tf_x + tf_y
	with tf.name_scope('matrix_multiplication'):
		output = tf.matmul(tf.transpose(tf_x), output)
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	file_writer = tf.summary.FileWriter(logdir='logs/2', graph=g)
	result = sess.run(output)
	print(result)

# With tensorboard we can track scalar values over time using summaries
g = tf.Graph()
with g.as_default() as g:
	some_value = tf.placeholder(dtype=tf.int32, shape=None, name='some_value')
	tf_x = tf.Variable([[1., 2.], [3., 4.], [5., 6.]], name='tf_x_0', dtype=tf.float32)
	tf_y = tf.Variable([[7., 8.], [9., 10.], [11., 12.]], name='tf_y_0', dtype=tf.float32)
	with tf.name_scope('addition'):
		output = tf_x + tf_y
	with tf.name_scope('matrix_multiplication'):
		output = tf.matmul(tf.transpose(tf_x), output)
	with tf.name_scope('update_tensor_x'):
		tf_const = tf.constant(2., shape=None, name='some_const')
		update_tf_x = tf.assign(tf_x, tf_x * tf_const)
	tf.summary.scalar(name='some_value', tensor=some_value)
	tf.summary.histogram(name='tf_x_values', values=tf_x)
	merged_summary = tf.summary.merge_all()
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
	file_writer = tf.summary.FileWriter(logdir='logs/3', graph=g)
	for i in range(5):
		result, summary = sess.run([update_tf_x, merged_summary], feed_dict={some_value: i})
		file_writer.add_summary(summary=summary, global_step=i)
		file_writer.flush()
