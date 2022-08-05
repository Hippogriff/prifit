import argparse

def parse_args():

    parser = argparse.ArgumentParser('Train PointNet++ PartSeg Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default=None, help='GPU to use [default: None]')
    parser.add_argument('--cudnn_off', action='store_true', default=False, help='disable CuDNN [default: False]')
    parser.add_argument('--seed', type=int, default=0, help='Seed for multiple runs [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--category', action='store_true', default=False, help='use category label information [default: False]')
    parser.add_argument('--l2_norm', action='store_true', default=False, help='unit-normalize features [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--rotation_z', action='store_true', default=False, help='use z-rotation jitter [default: False]')
    parser.add_argument('--rotation_z_45', action='store_true', default=False, help='use z-rotation 45 degrees [default: False]')
    parser.add_argument('--random_anisotropic_scale', action='store_true', default=False, help='anisotropic scaling jittering [default: False]')
    parser.add_argument('--modelnet_val', action='store_true', default=False, help='do validation on ModelNet40 [default: False]')
    parser.add_argument('--lr_clip', type=float,  default=1e-5, help='Minimum value of lr [default: 1e-5]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--dgcnn_k', type=int,  default=20, help='DGCNN k [default: 20]')
    parser.add_argument('--num_classes', type=int,  default=16, help='Number of shape classes [default: 16]')
    parser.add_argument('--num_parts', type=int,  default=50, help='Number of part classes [default: 50]')
    # self-supervised loss setting
    parser.add_argument('--selfsup', action='store_true', default=False, help='use self-sup data [default: False]')
    parser.add_argument('--margin', type=float,  default=0.5, help='contrastive loss margin [default: 0.5]')
    parser.add_argument('--lmbda', type=float,  default=10.0, help='weight on self-sup loss [default: 10]')
    parser.add_argument('--n_cls_selfsup', type=int,  default=-1, help='self-sup samples per class [default: -1, all samples]')
    parser.add_argument('--ss_dataset', type=str, default='acd', help='self-sup dataset [default: acd]')
    parser.add_argument('--ss_path', type=str, default='/srv/data2/mgadelha/ShapeNetACD/', help='self-sup dataset location [default: dummy]')
    parser.add_argument('--retain_overlaps', action='store_true', default=False, help='keep overlapping shapes with labeled data [default: False]')
    # annealing the weight of the self-sup loss (lambda)
    parser.add_argument('--anneal_lambda', action='store_true', default=False, help='anneal lambda value [default: False]')
    parser.add_argument('--anneal_step',type=int, default=5, help='anneal lambda epochs [default: 10]')
    parser.add_argument('--anneal_rate',type=float, default=0.5, help='anneal lambda epochs [default: 10]')
    # few-shot setting
    parser.add_argument('--k_shot', type=int,  default=-1, help='few shot samples [default: -1, all samples]')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pre-trained model path [default: None]')
    parser.add_argument('--init_cls', action='store_true', default=False, help='pre-train classifier layers [default: False]')
    parser.add_argument('--train_split', type=str, default='trainval', help='data split for training [default: trainval]')
    parser.add_argument('--eval_split', type=str, default='test', help='data split for evaluation [default: test]')
    parser.add_argument('--quantile', type=float, default='0.01', help='quantile in mean shift clustering [default: 0.01]')
    parser.add_argument('--msc_iterations', type=int, default='20', help='number of iterations in mean shift clustering [default: 20]')
    parser.add_argument('--max_num_clusters', type=int, default='25', help='Max clustering in mean shift clustering [default: 25]')
    parser.add_argument('--include_convex_loss', action='store_true', default=False, help='Surface Fitting [default: False]')
    parser.add_argument('--include_intersect_loss', action='store_true', default=False, help='Intersection Loss [default: False]')
    parser.add_argument('--include_entropy_loss', action='store_true', default=False, help='Entropy Loss [default: False]')
    parser.add_argument('--include_pruning', action='store_true', default=False, help='Pruning [default: False]')
    parser.add_argument('--alpha', type=float, default=1, help='Adjusts the Ellipsoid Intersection Loss [default: 1]')
    parser.add_argument('--beta', type=float, default=0.01, help='Adjusts the SelfSup Entropy Loss [default: 0.01]')
    parser.add_argument('--if_cuboid', action='store_true', default=False, help='Enable Cuboid Fitting [default: False')
    parser.add_argument('--reconstruct', action='store_true', default=False, help='Enable reconstruction loss [default: False')
    parser.add_argument('--extra_layers', action='store_true', default=False, help='Enable extra_layers [default: False')
    parser.add_argument('--num_charts', type=int, default=25, help='Number of charts for AtlasNet [default: 25')
    parser.add_argument('--num_points', type=int, default=128, help='Number of points for AtlasNet [default: 128')
    parser.add_argument('--embed', action='store_true', default=False, help='Enable to return embeddings [default: False')
    parser.add_argument('--ckpt', type=str, default=None, 
                            help='model checkpoint filename [default: None]')
    parser.add_argument('--num_point', type=int, default=1024, 
                            help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', 
                            help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, 
                            help='Whether to use normal information [default: False]')
    parser.add_argument('--sqrt', action='store_true', default=False, 
                            help='Whether to use sqrt normalization [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, 
                            help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--cross_val_svm', action='store_true', default=False, 
                            help='Whether to cross-validate SVM C [default: False]')
    parser.add_argument('--svm_c', type=float, default=220.0, 
                            help='Linear SVM `C` hyper-parameter [default: 220.0]')
    parser.add_argument('--val_svm', action='store_true', default=False, 
                            help='Whether to use test or val set for eval [default: False]')
    parser.add_argument('--svm_jitter', action='store_true', default=False, 
                            help='Whether to jitter data during SVM training [default: False]')
    parser.add_argument('--do_sa3', action='store_true', default=False, 
                            help='Use SA3 layer features of PointNet++ [default: False]')
    parser.add_argument('--random_feats', action='store_true', default=False, 
                            help='Use randomly initialized PointNet++ features [default: False]')
    return parser.parse_args()
