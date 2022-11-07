# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import sys
import _thread
from playsound import playsound


import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import new_input_test
import math
import numpy as np
from i3d import InceptionI3d
from new_utils import *

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 4, 'The num of class')
FLAGS = flags.FLAGS
action = {0: "no_act", 1: "chin_hand", 2: "finger_nail", 3: "lip_bite"}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cam_images_path = "images"

working_path  = os.getcwd()
music_path = {"katalk": os.path.join( working_path,"music\\katalk.mp3"),
              "galaxy": os.path.join( working_path,"music\\galaxy.mp3"),
              "iphone": os.path.join( working_path,"music\\iphone.mp3")}




def RunControl():
    calc = ".\\cam_receiver\\cam_receiver.exe"
    os.popen(calc)


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    rgb_model_save_dir = "./models"
    flow_model_save_dir = "./models"

    with tf.Graph().as_default():
        rgb_images_placeholder, flow_images_placeholder, is_training = placeholder_inputs(
            FLAGS.batch_size * gpu_num,
            FLAGS.num_frame_per_clib,
            FLAGS.crop_size,
            FLAGS.rgb_channels
        )

        with tf.variable_scope('RGB'):
            rgb_logit, _ = InceptionI3d(
                num_classes=FLAGS.classics,
                spatial_squeeze=True,
                final_endpoint='Logits',
                name='inception_i3d'
            )(rgb_images_placeholder, is_training)
        with tf.variable_scope('Flow'):
            flow_logit, _ = InceptionI3d(
                num_classes=FLAGS.classics,
                spatial_squeeze=True,
                final_endpoint='Logits',
                name='inception_i3d'
            )(flow_images_placeholder, is_training)
        norm_score = tf.nn.softmax(tf.add(rgb_logit, flow_logit))

        # Create a saver for writing training checkpoints.
        rgb_variable_map = {}
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB' and 'Adam' not in variable.name.split('/')[-1]:
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow' and 'Adam' not in variable.name.split('/')[-1]:
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
        saver = tf.train.Saver()  # can remove
        init = tf.global_variables_initializer()  # can remove

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True)
        )
        sess.run(init)

    # load pre_train models
    # ckpt = tf.train.get_checkpoint_state(rgb_model_save_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    print("loading checkpoint %s,waiting......" % ".\\models\\rgb\\i3d_ucf_model-2399")
    rgb_saver.restore(sess, ".\\models\\rgb\\i3d_ucf_model-2399")
    print("load complete!")

    # ckpt = tf.train.get_checkpoint_state(flow_model_save_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    print("loading checkpoint %s,waiting......" % ".\\models\\flow\\i3d_ucf_model-2399")
    flow_saver.restore(sess, ".\\models\\flow\\i3d_ucf_model-2399")
    print("load complete!")

    top1_list = []

    start_time = time.time()


    predicts = []
    top1 = False
    while True:
        rgb_images, flow_images = new_input_test.read_clip_and_label(
            filename=cam_images_path,
            num_frames_per_clip=FLAGS.num_frame_per_clib,
            crop_size=FLAGS.crop_size,
        )
        predict = sess.run(norm_score,
                           feed_dict={
                               rgb_images_placeholder: rgb_images,
                               flow_images_placeholder: flow_images,
                               is_training: False
                           })
        predicts.append(np.array(predict).astype(np.float32).reshape(FLAGS.classics))

        out_predictions = predict[0]
        sorted_indices = np.argsort(out_predictions)[::-1]
        print('\nTop classes and probabilities')
        for index in sorted_indices[:4]:
            print(out_predictions[index], action[index])


        #if act != noact
        if sorted_indices[0] == 1:
            playsound(music_path["galaxy"])
            time.sleep(5)
        if sorted_indices[0] == 2:
            playsound(music_path["katalk"])
            time.sleep(5)
        if sorted_indices[0] == 3:
            playsound(music_path["iphone"])
            time.sleep(5)



def main(_):

    os.popen("cam_receiver\\cam_receiver.exe")
    time.sleep(20) # image set wait time
    run_training()


if __name__ == '__main__':
    tf.app.run()
