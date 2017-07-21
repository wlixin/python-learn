import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib 
matplotlib.use("Agg")



Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()


fig, ax = plt.subplots()
ax.plot(x_data,y_data,'k.')
line, = ax.plot(x_data,y_data,'r-',linewidth=2.0)
fw = np.zeros(201)
fb = np.zeros(201)


# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
#fig, ax = plt.subplots() 
#ax.plot(x_data,y_data,'k.')
#line, = ax.plot(x_data,y,'r-',linewidth=2.0)

#fw = np.array(201)
#fb = np.array(201)
for step in range(201):
    sess.run(train)
    fw[step] = sess.run(W)
    fb[step] = sess.run(b)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
	#plt.plot(x_data,sess.run(W)*x_data+sess.run(b),label=str(step))
def update(i):
	line.set_ydata(fw[i]*x_data+fb[i])
	return line
anim = FuncAnimation(fig, update, frames=np.arange(0, 200))

anim.save('lines.mp4', writer=writer)
		
#plt.legend()
plt.show()
