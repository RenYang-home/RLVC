import os
import argparse

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--f_P", type=int, default=6)
parser.add_argument("--b_P", type=int, default=6)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--python_path", default='python')
parser.add_argument("--CA_model_path", default='./CA_EntropyModel_Test/')
parser.add_argument("--l", type=int, default=1024, choices=[8, 16, 32, 64, 256, 512, 1024, 2048])

args = parser.parse_args()

os.system(args.python_path + ' Bottleneck_decoder.py --path ' + args.path + ' --frame ' + str(args.frame)
          + ' --f_P ' + str(args.f_P) + ' --b_P ' + str(args.b_P) + ' --mode ' + args.mode
          + ' --python_path ' + args.python_path + ' --CA_model_path ' + args.CA_model_path
          + ' --l ' + str(args.l))

os.system(args.python_path + ' RPM_decoder.py --path ' + args.path + ' --frame ' + str(args.frame)
          + ' --f_P ' + str(args.f_P) + ' --b_P ' + str(args.b_P) + ' --mode ' + args.mode
          + ' --l ' + str(args.l))

os.system(args.python_path + ' RAE_decoder.py --path ' + args.path + ' --frame ' + str(args.frame)
          + ' --f_P ' + str(args.f_P) + ' --b_P ' + str(args.b_P) + ' --mode ' + args.mode
          + ' --l ' + str(args.l))

