"""
Inference code for LMAFormer.
Based on VisTR (https://github.com/Epiphqny/VisTR)
and DETR (https://github.com/facebookresearch/detr)
and MED-VT (https://github.com/rkyuca/medvt)
"""
import sys
import argparse
import logging
import random
import numpy as np
import os
import torch
import MFIRSTD.utils.misc as utils_misc
from MFIRSTD.evals import run_inference

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args_parser():
    parser = argparse.ArgumentParser('LMAFormer', add_help=False)
    # Model name
    parser.add_argument('--st_model', default='LMAFormer', type=str,
                        help="LMAFormer")
    # Backbone
    parser.add_argument('--backbone', default='swinS', type=str,
                        help="backbone to use, [swinS, swinB]")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument('--enc_layers', default=(6, 1), type=tuple,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--encoder_cross_layer', default=True, type=bool,
                        help="Cross resolution attention")
    parser.add_argument('--dec_layers', default=9, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dec_multiscale', default='yes', type=str,
                        help="Multi-scale vs single scale decoder, for ablation")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of frames")
    parser.add_argument('--val_size', default=400, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
      # MFQA
    parser.add_argument('--is_MJQM', default=True, type=bool,
                        help="MJQM Switch")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots")
    parser.add_argument('--balance_weight', default=0.1, type=float,
                        help="Original-grouped query balance weight")
      # SC block
    parser.add_argument('--is_sc_block', default=True, type=bool,
                        help="MAFEM Switch")
    parser.add_argument('--flow_loss_coef', default=1, type=float)

    # Init Weights
    parser.add_argument('--is_train', default=0, type=int,
                             help='Choose 1 for train')
    parser.add_argument('--model_path', type=str,
                        default='./result/HIT-TSIRMT/swin_S_with_MFAQ_bottom_basic_dynamic_lamda_0.1_queriesnum_5_with_sc_top/checkpoint_best.pth',
                        help="Path to the model weights.")
    parser.add_argument('--swin_s_pretrained_path', type=str,
                        default="./ckpts/swin_init/swin_small_patch244_window877_kinetics400_1k.pth",
                        help="swin-s pretrained model path.")
    parser.add_argument('--swin_b_pretrained_path', type=str,
                        default="./ckpts/swin_init/swin_base_patch244_window877_kinetics400_22k.pth",
                        help="swin-b pretrained model path.")

    # LOSS
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    parser.add_argument('--pretrain_settings', default=None, nargs=argparse.REMAINDER, help='for two-stage train')
    # Segmentation
    parser.add_argument("--save_pred", action="store_true", default=True)
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--num_classes', default=1, type=int,
                             help="Train segmentation head if the flag is provided")
    parser.add_argument('--dataset', type=str, default='TSIRMT', help='TSIRMT,NUDT-MIRSDT,IRDST')
    parser.add_argument('--sequence_names', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='predict',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--msc', action='store_true')
    parser.add_argument('--flip', action='store_true', default=True)

    # Misc
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    device = torch.device(args.device)
    utils_misc.init_distributed_mode(args)
    seed = args.seed + utils_misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args.aux_loss = 0
    args.aux_loss_norm = 0

    from MFIRSTD.models.lmaf_swin import build_model_lmaf_swinbackbone as build_model
    model, _ = build_model(args)
    model.to(device)
    from MFIRSTD.metrics import SigmoidMetric,SamplewiseSigmoidMetric,PD0_FA0
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0)
    PD0_FA0 = PD0_FA0(nclass=1, thre=0)  #
    args.sequence_names = None
    run_inference(args, device, model, iou_metric, nIoU_metric, PD0_FA0)
    print('Thank You!')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('VisVOS inference script', parents=[get_args_parser()])
    parsed_args = args_parser.parse_args()
    if not hasattr(parsed_args, 'output_dir') or parsed_args.output_dir is None or len(parsed_args.output_dir) < 3:
        from MFIRSTD.evals import create_eval_save_dir_name_from_args
        out_dir_name = create_eval_save_dir_name_from_args(parsed_args)
        parsed_args.output_dir = os.path.join(os.path.dirname(parsed_args.model_path), out_dir_name)
    if not os.path.exists(parsed_args.output_dir):
        os.makedirs(parsed_args.output_dir)
    experiment_name = str(parsed_args.model_path).split('/')[-2]
    logging.basicConfig(
        filename=os.path.join(parsed_args.output_dir, 'out.log'),
        format='%(asctime)s %(levelname)s %(module)s-%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.debug(parsed_args)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.debug('output_dir: ' + str(parsed_args.output_dir))
    logger.debug('experiment_name:%s' % experiment_name)
    logger.debug('log file: ' + str(os.path.join(parsed_args.output_dir, 'out.log')))
    main(parsed_args)
