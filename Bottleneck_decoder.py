import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
from scipy import misc
import CNN_recurrent
import motion
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
parser.add_argument("--python_path", default='path_to_python')
parser.add_argument("--CA_model_path", default='path_to_CA_EntropyModel_Test')
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

# settings
_, GOP_size, GOP_num, \
path, path_com, path_bin, path_lat = helper.configure_decoder(args)

# decode I frames
for g in range(GOP_num + 1):

    I_index = g * GOP_size + 1

    if I_index <= args.frame:
        helper.decode_I(args, I_index, path_com, path_bin)

F1 = misc.imread(path_com + 'f001.png')
Height = np.size(F1, 0)
Width = np.size(F1, 1)

# placeholder
string_mv_tensor = tf.placeholder(tf.string, [])
string_res_tensor = tf.placeholder(tf.string, [])

# decode motion latent
entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32, name='entropy_bottleneck')
motion_latent_hat = entropy_bottleneck.decompress(
    tf.expand_dims(string_mv_tensor, 0), [Height//16, Width//16, args.M], channels=args.M)

# decode residual latent
entropy_bottleneck_2 = tfc.EntropyBottleneck(dtype=tf.float32, name='entropy_bottleneck_1_1')
residual_latent_hat = entropy_bottleneck_2.decompress(
    tf.expand_dims(string_res_tensor, 0), [Height//16, Width//16, args.M], channels=args.M)

# load model
saver = tf.train.Saver(max_to_keep=None)
model_path = './model/RAE_' + args.mode + '_' + str(args.l)
saver.restore(sess, save_path=model_path + '/model.ckpt')


for g in range(GOP_num + 1):

    I_index = g * GOP_size + 1

    if I_index <= args.frame:

        # if there exists forward P frame(s), I_index + 1 is decoded by the bottleneck
        if args.f_P > 0 and I_index + 1 <= args.frame:

            with open(path_bin + 'f' + str(I_index + 1).zfill(3) + '.bin', "rb") as ff:
                mv_len = np.frombuffer(ff.read(2), dtype=np.uint16)
                string_mv = ff.read(np.int(mv_len))
                string_res = ff.read()

            latent_mv, latent_res = sess.run([motion_latent_hat, residual_latent_hat], feed_dict={
                string_mv_tensor: string_mv,
                string_res_tensor: string_res})

            np.save(path_lat + '/f' + str(I_index + 1).zfill(3) + '_mv.npy', latent_mv)
            np.save(path_lat + '/f' + str(I_index + 1).zfill(3) + '_res.npy', latent_res)

            print('Decoded latents frame', I_index + 1)

        # if there exists backward P frame(s), I_index - 1 is decoded by the bottleneck
        if args.b_P > 0 and I_index - 1 >= 1:

            with open(path_bin + 'f' + str(I_index - 1).zfill(3) + '.bin', "rb") as ff:
                mv_len = np.frombuffer(ff.read(2), dtype=np.uint16)
                string_mv = ff.read(np.int(mv_len))
                string_res = ff.read()

            latent_mv, latent_res = sess.run([motion_latent_hat, residual_latent_hat], feed_dict={
                string_mv_tensor: string_mv,
                string_res_tensor: string_res})

            np.save(path_lat + '/f' + str(I_index - 1).zfill(3) + '_mv.npy', latent_mv)
            np.save(path_lat + '/f' + str(I_index - 1).zfill(3) + '_res.npy', latent_res)

            print('Decoded latents frame', I_index - 1)






