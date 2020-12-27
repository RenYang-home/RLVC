import numpy as np
import tensorflow as tf
import os
from scipy import misc
from ms_ssim_np import MultiScaleSSIM
import arithmeticcoding

def configure(args):

    if args.l == 256:
        I_level = 37
    elif args.l == 512:
        I_level = 32
    elif args.l == 1024:
        I_level = 27
    elif args.l == 2048:
        I_level = 22

    elif args.l == 8:
        I_level = 3
    elif args.l == 16:
        I_level = 3
    elif args.l == 32:
        I_level = 5
    elif args.l == 64:
        I_level = 7

    path = args.path + '/'
    path_com = args.path + '_' + args.mode + '_' + str(args.l) + '/frames/'
    path_bin = args.path + '_' + args.mode + '_' + str(args.l) + '/bitstreams/'
    path_lat = args.path + '_' + args.mode + '_' + str(args.l) + '/latents/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_bin, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    F1 = misc.imread(path + 'f001.png')
    Height = np.size(F1, 0)
    Width = np.size(F1, 1)
    batch_size = 1
    Channel = 3

    if (Height % 16 != 0) or (Width % 16 != 0):
        raise ValueError('Height and Width must be a mutiple of 16.')

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + 1
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return I_level, Height, Width, batch_size, \
           Channel, activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat


def configure_decoder(args):

    path = args.path + '/'
    path_com = args.path + '_' + args.mode + '_' + str(args.l) + '/frames_dec/'
    path_bin = args.path + '_' + args.mode + '_' + str(args.l) + '/bitstreams/'
    path_lat = args.path + '_' + args.mode + '_' + str(args.l) + '/latents_dec/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + 1
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat


def encode_I(args, frame_index, I_level, path, path_com, path_bin):

    if args.mode == 'PSNR':
        os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(frame_index).zfill(3) + '.png '
                  '-o ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin -q ' + str(I_level))
        os.system('bpgdec ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin '
                  '-o ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/encode.py --model_type 1 '
                  '--input_path ' + path + 'f' + str(frame_index).zfill(3) + '.png' +
                  ' --compressed_file_path ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --quality_level ' + str(I_level))
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path '
                  + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    # bits = os.path.getsize(path_bin + str(frame_index).zfill(3) + '.bin')
    # bits = bits * 8

    F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
    F0_raw = misc.imread(path + 'f' + str(frame_index).zfill(3) + '.png')

    F0_com = np.expand_dims(F0_com, axis=0)
    F0_raw = np.expand_dims(F0_raw, axis=0)

    if args.metric == 'PSNR':
        mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
        quality = 10 * np.log10(1.0 / mse)
    elif args.metric == 'MS-SSIM':
        quality = MultiScaleSSIM(F0_com, F0_raw, max_val=255)

    print('Frame', frame_index, args.metric + ' =', quality)

    return quality


def decode_I(args, frame_index, path_com, path_bin):

    if args.mode == 'PSNR':
        os.system('bpgdec ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin '
                  '-o ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path '
                  + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    print('Decoded I-frame', frame_index)


def entropy_coding(frame_index, lat, path_bin, latent, sigma, mu):

    if lat == 'mv':
        bias = 50
    else:
        bias = 100

    bin_name = 'f' + str(frame_index).zfill(3) + '_' + lat + '.bin'
    bitout = arithmeticcoding.BitOutputStream(open(path_bin + bin_name, "wb"))
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

    for h in range(latent.shape[1]):
        for w in range(latent.shape[2]):
            for ch in range(latent.shape[3]):
                mu_val = mu[0, h, w, ch] + bias
                sigma_val = sigma[0, h, w, ch]
                symbol = latent[0, h, w, ch] + bias

                freq = arithmeticcoding.logFrequencyTable_exp(mu_val, sigma_val, np.int(bias * 2 + 1))
                enc.write(freq, symbol)

    enc.finish()
    bitout.close()

    bits_value = os.path.getsize(path_bin + bin_name) * 8

    return bits_value


def entropy_decoding(frame_index, lat, path_bin, path_lat, sigma, mu):

    if lat == 'mv':
        bias = 50
    else:
        bias = 100

    bin_name = 'f' + str(frame_index).zfill(3) + '_' + lat + '.bin'
    bitin = arithmeticcoding.BitInputStream(open(path_bin + bin_name, "rb"))
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

    latent = np.zeros([1, mu.shape[1], mu.shape[2], mu.shape[3]])

    for h in range(mu.shape[1]):
        for w in range(mu.shape[2]):
            for ch in range(mu.shape[3]):

                mu_val = mu[0, h, w, ch] + bias
                sigma_val = sigma[0, h, w, ch]

                freq = arithmeticcoding.logFrequencyTable_exp(mu_val, sigma_val, np.int(bias * 2 + 1))
                symbol = dec.read(freq)
                latent[0, h, w, ch] = symbol - bias

    bitin.close()

    np.save(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy', latent)
    print('Decoded latent_' + lat + ' frame', frame_index)

    return latent




