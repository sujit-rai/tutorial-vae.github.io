import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import os
import sys
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True);

tf.set_random_seed(0);

#tuning_knobs
learning_rate = 0.001;
batch_size = 128;
n_epochs = 500; 

#model params
z_dim = 32;

tfd = tf.contrib.distributions #we will use this to calculate kl-divergence

X = tf.placeholder(tf.float32,[None,784]);
epoch_number = tf.placeholder(tf.float32,[]);

def get_activations(name,input_image,session):
	actvtns = session.run(name,feed_dict={X:input_image});
	return actvtns;

def encoder_dist(X,isTrainable=True,reuse=False,name='encoder'):
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
		X = tf.reshape(X,[-1,28,28,1]);
		outputs={};
		conv1 = tf.layers.conv2d(X,filters=16,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='conv1_layer');
		conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',trainable=isTrainable,reuse=reuse);
		conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='conv1_layer_maxpool');
		
		outputs['conv1'] = conv1;
		
		conv2 = tf.layers.conv2d(conv1,filters=32,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='conv2_layer');
		conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',trainable=isTrainable,reuse=reuse);
		conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='conv2_layer_maxpool');
		
		outputs['conv2'] = conv2;

		conv3 = tf.layers.conv2d(conv2,filters=32,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='conv3_layer');
		conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',trainable=isTrainable,reuse=reuse);
		conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='conv3_layer_maxpool');
		
		outputs['conv3'] = conv3;
		conv3_flt = tf.layers.flatten(conv3,name='flattened_conv3');
	
	with tf.variable_scope('VAE_mean_and_var_'+name) as scope:

		mean_fc = tf.layers.dense(conv3_flt,z_dim,trainable=True,reuse=False,activation=tf.nn.tanh,name='mean_fully_connected');
		var_fc = tf.layers.dense(conv3_flt,z_dim,activation=tf.nn.softplus,trainable=True,reuse=False,name='var_fully_connected');
		
		dist = tfd.MultivariateNormalDiag(mean_fc,var_fc);
		return dist,outputs;

def decoder(Z,isTrainable=True,reuse=False,name='decoder'):
	with tf.variable_scope(name) as scope:
		Z = tf.layers.dense(Z,4*4*32,activation=tf.nn.tanh,trainable=isTrainable,reuse=reuse,name='fully_connected_decoder_from_z_dim');
		outputs={};
		Z = tf.reshape(Z,[-1,4,4,32]);
		
		deconv1 = tf.image.resize_images(Z,size=[7,7],align_corners=False);
		deconv1 = tf.layers.conv2d_transpose(deconv1,filters=32,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv1_layer');
		deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',trainable=isTrainable,reuse=reuse);
		outputs['deconv1'] = deconv1;
		
		deconv2 = tf.image.resize_images(deconv1,size=[14,14],align_corners=False);
		deconv2 = tf.layers.conv2d_transpose(deconv2,filters=16,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv2_layer');
		deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',trainable=isTrainable,reuse=reuse);
		outputs['deconv2'] = deconv2;
		
		deconv3 = tf.image.resize_images(deconv2,size=[28,28],align_corners=False);
		deconv3 = tf.layers.conv2d_transpose(deconv3,filters=1,kernel_size=[3,3],strides=(1,1),padding='SAME',activation=tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='deconv3_layer');
		outputs['deconv3'] = deconv3;
		deconv3_reshaped = tf.reshape(deconv3,[-1,784]);
		return deconv3_reshaped,outputs;

prior_dist = tfd.MultivariateNormalDiag(tf.zeros(z_dim),tf.ones(z_dim));
posterior_dist,encoder_outputs = encoder_dist(X,isTrainable=False);

## z_sample = posterior_dist.sample();
epsilon_value = tfd.MultivariateNormalDiag(tf.zeros(z_dim),tf.ones(z_dim)).sample(tf.shape(X)[0]);
z_sample = tf.add(posterior_dist.mean(),tf.multiply(posterior_dist.stddev(),epsilon_value));
#z_sample = tf.placeholder(tf.float32,[None,z_dim]);
reconstruction,decoder_outputs = decoder(z_sample);
reconstruction_loss = tf.reduce_mean(tf.pow(X - reconstruction,2));
KL_loss = tf.reduce_mean(tfd.kl_divergence(posterior_dist,prior_dist));

kl_weight = 1.0 / (1.0 + tf.exp(-epoch_number/3+4));

kl_weight *= 0.001;
loss = kl_weight*KL_loss + reconstruction_loss;

#####changes#####


#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss);


## FOR TENSORBOARD ##

enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder');
VAE_mean_and_var_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='VAE_mean_and_var_'+'encoder');
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate);

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
	gradsVars = optimizer.compute_gradients(loss, tf.trainable_variables());
	train_optimizer = optimizer.apply_gradients(gradsVars);

tf.summary.scalar("reconstruction_loss",reconstruction_loss);
tf.summary.scalar("KL_loss",KL_loss);

merged_all = tf.summary.merge_all();
log_directory = 'Modified-VAE-dir';
model_directory='Modified-VAE-model_dir';

save_model_directory = 'Reconstruction-VAE-model_dir';

if not os.path.exists(log_directory):
    os.makedirs(log_directory);
if not os.path.exists(model_directory):
    os.makedirs(model_directory);
####################################



#code = encoder(X);

def train_model():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer());

		n_batches = mnist.train.num_examples/batch_size;
		n_batches = int(n_batches);

		saver = tf.train.Saver();

		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder');
		saver = tf.train.Saver(var_list=params);

		print('----------------PARAMS-----------------');
		for var in params:
		    print (var.name+"\t");
		print('---------------------------------');

		string = save_model_directory+'/model_'+str(98); 

		try:
		    saver.restore(sess, string);
		except:
		    print("Previous weights not found of encoder"); 
		    sys.exit(0);

		print('---------------------------------');
		print ("Model loaded");
		print('---------------------------------');
		saver = tf.train.Saver();
		
		writer = tf.summary.FileWriter(log_directory,sess.graph);
		
		train_list = tf.trainable_variables();

		print('----------------TRAINABLE_VARIABLES----------------');
		for it in train_list:
			print(it.name+"\t");

		print('---------------------------------------------------');
		
		for epoch in range(n_epochs):
			epoch_loss = 0;
			epoch_KL_loss = 0;
			epoch_reconstruction_loss = 0;
			for batch in range(n_batches):
				X_batch,_ = mnist.train.next_batch(batch_size);
				_,batch_cost,merged,batch_KL_loss,batch_reconstruction_loss = sess.run([train_optimizer,loss,merged_all,KL_loss,reconstruction_loss],feed_dict={X:X_batch,epoch_number:epoch});
				epoch_loss += batch_cost;
				epoch_KL_loss += batch_KL_loss;
				epoch_reconstruction_loss += batch_reconstruction_loss;
				writer.add_summary(merged,epoch*n_batches+batch);
			print('At epoch #',epoch,' loss is ',epoch_loss ,' where recons loss : ',epoch_reconstruction_loss,' and KL_loss : ',epoch_KL_loss);
			if(epoch % 2) == 0:
				save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
				print("At epoch #",epoch," Model is saved at path: ",save_path);
		print('Optimization Done !!');
		n = 5;
		
		reconstructed = np.empty((28*n,28*n));
		original = np.empty((28*n,28*n));

		for i in range(n):
			
			batch_X,_ = mnist.test.next_batch(n);
			recons = sess.run(reconstruction,feed_dict={X:batch_X});
			print ('recons : ',recons.shape);
			recons = np.reshape(recons,[-1,784]);
			print ('recons : ',recons.shape);

			for j in range(n):
		        	original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28]);

			for j in range(n):
				reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28]);

		print("Original Images");
		plt.figure(figsize=(n, n));
		plt.imshow(original, origin="upper", cmap="gray");
		plt.savefig('original_new_vae.png');

		print("Reconstructed Images");
		plt.figure(figsize=(n, n));
		plt.imshow(reconstructed, origin="upper", cmap="gray");
		plt.savefig('reconstructed_new_vae.png');
		
		

def generateSample():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver();

		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder');
		saver = tf.train.Saver(var_list=params);

		for var in params:
		    print (var.name+"\t");

		string = model_directory+'/model_'+str(n_epochs-2); 

		try:
		    saver.restore(sess, string);
		except:
		    print("Previous weights not found of decoder"); 
		    sys.exit(0);

		print ("Model loaded");
		saver = tf.train.Saver();

		sample = tf.random_normal([1,z_dim]);
		recons = sess.run(reconstruction,feed_dict={z_sample:sample.eval()});
		plt.imshow(np.reshape(recons,[28,28]), interpolation="nearest", cmap="gray");
		plt.title('Generated Image');
		plt.savefig('gen-img.png');

def plot_conv_layers():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver();

		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder');
		saver = tf.train.Saver(var_list=params);

		for var in params:
		    print (var.name+"\t");

		string = model_directory+'/model_'+str(n_epochs-2); 

		try:
		    saver.restore(sess, string);
		except:
		    print("Previous weights not found of decoder"); 
		    sys.exit(0);

		print ("Model loaded");
		saver = tf.train.Saver();

		if not os.path.exists('out'):
			os.makedirs('out');

		x_img,y_img = mnist.train.next_batch(1);
		plt.imshow(np.reshape(x_img,[28,28]), interpolation="nearest", cmap="gray");
		plt.title('Original Image');
		plt.savefig('out/real-img');

		encoder_output = sess.run(encoder_outputs,feed_dict={X:x_img});
		print ('conv1 shape--> ',encoder_output['conv1'].shape);
		print ('conv2 shape--> ',encoder_output['conv2'].shape);

		#plt.subplots_adjust(left=0.125, bottom=0.01, right=0.9, top=0.9, wspace=0.5, hspace=0.001)

		op = encoder_output['conv1'];
		filters = op.shape[3];
		plt.figure(1, figsize=(20,20));
		n_columns = 8;
		n_rows = math.ceil(filters / n_columns) + 1;
		
		for i in range(filters):
			plt.subplot(n_rows, n_columns, i+1);
			plt.title('Filter ' + str(i));
			plt.imshow(op[0,:,:,i], interpolation="nearest", cmap="gray");
		plt.tight_layout();
		plt.savefig('out/conv1-op');

		op = encoder_output['conv2'];
		filters = op.shape[3];
		plt.figure(2, figsize=(30,30));
		n_columns = 8;
		n_rows = math.ceil(filters / n_columns) + 1;
		for i in range(filters):
			plt.subplot(n_rows, n_columns, i+1);
			plt.title('Filter ' + str(i));
			plt.imshow(op[0,:,:,i], interpolation="nearest", cmap="gray");
		plt.tight_layout();
		plt.savefig('out/conv2-op');

		op = encoder_output['conv3'];
		filters = op.shape[3];
		plt.figure(3, figsize=(30,30));
		n_columns = 8;
		n_rows = math.ceil(filters / n_columns) + 1;
		for i in range(filters):
			plt.subplot(n_rows, n_columns, i+1);
			plt.title('Filter ' + str(i));
			plt.imshow(op[0,:,:,i], interpolation="nearest", cmap="gray");
		plt.tight_layout();
		plt.savefig('out/conv3-op');


#train_model();
generateSample();	
plot_conv_layers();
