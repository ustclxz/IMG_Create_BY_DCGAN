#! coding: Shift_JIS
import time
import os
import random
from flask import Flask, render_template, request,abort, redirect, url_for
from glob import glob
from pprint import pprint
import json
from lib import nn
import scipy.misc
import numpy as np
from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [4]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./static/images/samples/", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def Test():
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir)

    show_all_variables()
    print('variables OK test')
    print('FLAGS.train is',FLAGS.train)

    if FLAGS.train:
      print('variables OK test01')
      dcgan.train(FLAGS) #DEMO2      
    else:
      print('variables OK test02')
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    #この処理を飛ばす
    #OPTION = 1
    #visualize(sess, dcgan, FLAGS, OPTION)

#add e

app = Flask(__name__)
predictor = None

def initialize():
    #app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10MiB
    global predictor
    if not predictor:
        predictor = nn.Predictor()

@app.route('/')
def index():
	#test
    #initialize()    
    #return render_template('index.html', items=predictor.get_sample_items())
    return render_template('index.html')
    
@app.route('/demo')
def demo():
    Test()
    initialize()    
    return render_template('index.html', items=predictor.get_sample_items())

@app.route('/upload', methods=['POST'])
def fungo():
    #return redirect(url_for('demo'))
    Test()
    initialize()    
    item = predictor.get_sample_items()
    return json.dumps(item)
    #return render_template('index.html', items=predictor.get_sample_items())

#WEB画面からサーバーへのリクエスト（POST）
@app.route("/mockservice",methods=['GET','POST'])
def MockController():
    Test()
    initialize()    
    return render_template('index.html', items=predictor.get_sample_items())
	
if __name__ == '__main__':
    app.run()
