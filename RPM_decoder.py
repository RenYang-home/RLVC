import argparse
import numpy as np
import tensorflow as tf
import os
from scipy import misc
import CNN_recurrent
import helper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# use CPU for RPM to ensure the determinism
config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--f_P", type=int, default=6)
parser.add_argument("--b_P", type=int, default=6)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

# settings
_, GOP_size, GOP_num, \
path, path_com, path_bin, path_lat = helper.configure_decoder(args)

F1 = misc.imread(path_com + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)
batch_size = 1

# Placeholder
prior_tensor = tf.placeholder(tf.float32, [batch_size, Height//16, Width//16, args.M]) # previous latent
hidden_states = tf.placeholder(tf.float32, [2, batch_size, Height//16, Width//16, args.N]) # hidden states in RPM

c_prob, h_prob = tf.split(hidden_states, 2, axis=0)

# RPM network
prob_latent, c_prob_out, h_prob_out \
    = CNN_recurrent.rec_prob(prior_tensor, args.N, Height, Width, c_prob[0], h_prob[0])

sigma, mu = tf.split(prob_latent, [args.N, args.N], axis=-1)

hidden_states_out = tf.stack([c_prob_out, h_prob_out], axis = 0)

# start RPM decoder

latents = ['mv', 'res'] # two kinds of latents

for lat in latents:

    # load model
    model_path = './model/RPM_' + args.mode + '_' + str(args.l) + '_' + lat
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, save_path=model_path + '/model.ckpt')

    # decode GOPs
    for g in range(GOP_num):

        # forward P frames (only if more than 2 P frames exist)
        if args.f_P >= 2:
            # load first prior
            frame_index = g * GOP_size + 2
            prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

            # init state
            h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

            for f in range(args.f_P - 1):

                # load latent
                frame_index = g * GOP_size + f + 3

                # run RPM
                sigma_value, mu_value, h_state \
                    = sess.run([sigma, mu, hidden_states_out],
                               feed_dict={prior_tensor: prior_value, hidden_states: h_state})

                latent_value = helper.entropy_decoding(frame_index, lat, path_bin, path_lat, sigma_value, mu_value)

                # the latent will be the prior for the next latent
                prior_value = latent_value

        # backward P frames (only if more than 2 P frames exist)
        if args.b_P >= 2:
            # load first prior
            frame_index = (g + 1) * GOP_size
            prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

            # init state
            h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

            for f in range(args.b_P - 1):

                # load latent
                frame_index = (g + 1) * GOP_size - f - 1

                # run RPM
                sigma_value, mu_value, h_state \
                    = sess.run([sigma, mu, hidden_states_out],
                               feed_dict={prior_tensor: prior_value, hidden_states: h_state})

                latent_value = helper.entropy_decoding(frame_index, lat, path_bin, path_lat, sigma_value, mu_value)

                # the latent will be the prior for the next latent
                prior_value = latent_value

    # decode rest frames (only if more than 2 P frames exist)
    rest_frame_num = args.frame - 1 - GOP_size * GOP_num

    if rest_frame_num >= 2:
        # load first prior
        frame_index = GOP_num * GOP_size + 2
        prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

        # init state
        h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

        for f in range(rest_frame_num - 1):

            # load latent
            frame_index = GOP_num * GOP_size + f + 3

            # run RPM
            sigma_value, mu_value, h_state \
                = sess.run([sigma, mu, hidden_states_out],
                           feed_dict={prior_tensor: prior_value, hidden_states: h_state})

            latent_value = helper.entropy_decoding(frame_index, lat, path_bin, path_lat, sigma_value, mu_value)

            # the latent will be the prior for the next latent
            prior_value = latent_value




