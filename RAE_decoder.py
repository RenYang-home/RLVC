import argparse
import numpy as np
import tensorflow as tf
import os
from scipy import misc
import CNN_recurrent
import functions
import helper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
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
activation, GOP_size, GOP_num, \
path, path_com, path_bin, path_lat = helper.configure_decoder(args)

F1 = misc.imread(path_com + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)
batch_size = 1
Channel = 3

# Placeholder
Y0_com_tensor = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel]) # reference frame
motion_latent_hat = tf.placeholder(tf.float32, [batch_size, Height//16, Width//16, args.M])
res_latent_hat = tf.placeholder(tf.float32, [batch_size, Height//16, Width//16, args.M])

hidden_states = tf.placeholder(tf.float32, [4, batch_size, Height//4, Width//4, args.N]) # hidden states in RAE decoder
c_dec_mv, h_dec_mv, c_dec_res, h_dec_res = tf.split(hidden_states, 4, axis=0)

# RAE decoder for motion
motion_hat, c_dec_mv_out, h_dec_mv_out = CNN_recurrent.MV_synthesis(motion_latent_hat, num_filters=args.N,
                                      Height=Height, Width=Width,
                                      c_state=c_dec_mv[0], h_state=h_dec_mv[0], act=activation)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com_tensor, motion_hat)
MC_input = tf.concat([motion_hat, Y0_com_tensor, Y1_warp], axis=-1)
Y1_MC = functions.MC_RLVC(MC_input)

# RAE decoder for residual
res_hat, c_dec_res_out, h_dec_res_out = CNN_recurrent.Res_synthesis(res_latent_hat, num_filters=args.N,
                                      Height=Height, Width=Width,
                                      c_state=c_dec_res[0], h_state=h_dec_res[0], act=activation)

# reconstructed frame
Y1_decoded = tf.clip_by_value(res_hat + Y1_MC, 0, 1)

# output hidden states
hidden_states_out = tf.stack([c_dec_mv_out, h_dec_mv_out, c_dec_res_out, h_dec_res_out], axis=0)

# load model
saver = tf.train.Saver(max_to_keep=None)
model_path = './model/RAE_' + args.mode + '_' + str(args.l)
saver.restore(sess, save_path=model_path + '/model.ckpt')

# decode GOPs
for g in range(GOP_num):

    # forward P frames

    # load I frame (compressed)
    frame_index = g * GOP_size + 1
    F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
    F0_com = np.expand_dims(F0_com, axis=0)

    for f in range(args.f_P):

        # load latents
        frame_index = g * GOP_size + f + 2
        latent_mv = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_mv.npy')
        latent_res = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_res.npy')

        # init hidden states
        if f % 6 == 0:
            h_state = np.zeros([4, batch_size, Height // 4, Width // 4, args.N], dtype=np.float)
            # since the model is optimized on 6 frames, we reset hidden states every 6 P frames

        # run RAE decoder
        F0_com, h_state = sess.run([Y1_decoded, hidden_states_out],
                          feed_dict={Y0_com_tensor: F0_com / 255.0,
                                     motion_latent_hat: latent_mv,
                                     res_latent_hat: latent_res,
                                     hidden_states: h_state})
        F0_com = F0_com * 255

        # save compressed frame
        misc.imsave(path_com + '/f' + str(frame_index).zfill(3) + '.png', np.uint8(np.round(F0_com[0])))

        print('Decoded P-frame', frame_index)

    # load I frame (compressed)
    frame_index = (g + 1) * GOP_size + 1
    F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
    F0_com = np.expand_dims(F0_com, axis=0)

    for f in range(args.b_P):

        # load latents
        frame_index = (g + 1) * GOP_size - f
        latent_mv = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_mv.npy')
        latent_res = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_res.npy')

        # init hidden states
        if f % 6 == 0:
            h_state = np.zeros([4, batch_size, Height // 4, Width // 4, args.N], dtype=np.float)
            # since the model is optimized on 6 frames, we reset hidden states every 6 P frames

        # run RAE decoder
        F0_com, h_state = sess.run([Y1_decoded, hidden_states_out],
                          feed_dict={Y0_com_tensor: F0_com / 255.0,
                                     motion_latent_hat: latent_mv,
                                     res_latent_hat: latent_res,
                                     hidden_states: h_state})
        F0_com = F0_com * 255

        # save compressed frame
        misc.imsave(path_com + '/f' + str(frame_index).zfill(3) + '.png', np.uint8(np.round(F0_com[0])))

        print('Decoded P-frame', frame_index)

# deencode rest frames
rest_frame_num = args.frame - 1 - GOP_size * GOP_num

# load I frame (compressed)
frame_index = GOP_num * GOP_size + 1
F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
F0_com = np.expand_dims(F0_com, axis=0)

for f in range(rest_frame_num):

    # load latents
    frame_index = GOP_num * GOP_size + f + 2
    latent_mv = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_mv.npy')
    latent_res = np.load(path_lat + 'f' + str(frame_index).zfill(3) + '_res.npy')

    # init hidden states
    if f % 6 == 0:
        h_state = np.zeros([4, batch_size, Height // 4, Width // 4, args.N], dtype=np.float)
        # since the model is optimized on 6 frames, we reset hidden states every 6 P frames

    # run RAE decoder
    F0_com, h_state = sess.run([Y1_decoded, hidden_states_out],
                               feed_dict={Y0_com_tensor: F0_com / 255.0,
                                          motion_latent_hat: latent_mv,
                                          res_latent_hat: latent_res,
                                          hidden_states: h_state})
    F0_com = F0_com * 255

    # save compressed frame
    misc.imsave(path_com + '/f' + str(frame_index).zfill(3) + '.png', np.uint8(np.round(F0_com[0])))

    print('Decoded P-frame', frame_index)
