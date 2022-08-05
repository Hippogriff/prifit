import time
from src.utils import visualize_point_cloud_from_labels, visualize_point_cloud, save_point_cloud
import ipdb
import os
import os.path as osp
import data_utils
from data_utils.ShapeNetDataLoader import PartNormalDataset, SelfSupPartNormalDataset, ACDSelfSupDataset
from tensorboard_logger import configure, log_value
import itertools
import torch
from torch import nn
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import sys
from args_parser import parse_args
import open3d as o3d

torch.backends.cudnn.enabled = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
               'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23],
               'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37],
               'Pistol': [38, 39, 40], 'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
classes = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def evaluation(args, epoch = 0, classifier = None, metrics={}):
    def log_string(str):
        print(str)
        #sys.stdout.flush()

    '''CUDA ENV SETTINGS'''
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.cudnn_off:
        torch.backends.cudnn.enabled = False  # needed on gypsum!

    # --------------------------------------------------------------------------
    '''LOG'''
    # --------------------------------------------------------------------------
    if classifier is None:
        log_string('PARAMETERS ...')
        log_string(args)

    # --------------------------------------------------------------------------
    '''DATA LOADERS'''
    # --------------------------------------------------------------------------
    root = 'ShapeSelfSup/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal'

    print("Loading data from {} set".format(args.eval_split))
    sys.stdout.flush()

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split=args.eval_split, normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)

    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = args.num_classes
    num_part = args.num_parts
    seed = args.seed

    # --------------------------------------------------------------------------
    '''MODEL LOADING'''
    # --------------------------------------------------------------------------
    if args.pretrained_model is not None:
        MODEL = importlib.import_module(args.model)
        if 'dgcnn' in args.model:
            print('DGCNN params')
            classifier = MODEL.get_model(num_part, normal_channel=args.normal, k=args.dgcnn_k).cuda()
        else:
            classifier = MODEL.get_model(num_part, normal_channel=args.normal, reconstruct=args.reconstruct).cuda()

        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            classifier = nn.DataParallel(classifier)
        
        log_string('Loading pretrained model from %s' % args.pretrained_model)
        ckpt = torch.load(args.pretrained_model)
        classifier.load_state_dict(ckpt['model_state_dict'])

    chamfer_loss_list = []

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        total_seen_class_2, total_correct_class_2 = np.zeros(num_part), np.zeros(num_part)
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        shape_ious2 = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat           

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9, desc='Evaluation'):
            batch_index = 0
            cur_batch_size, NUM_POINT, _ = points.size()
            class_list = [classes[label[idx, :].data] for idx in range(label.size()[0])]

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            if args.category:
                category_label = to_categorical(label, num_classes).contiguous()
            else:
                category_label = torch.zeros([label.shape[0], 1, num_classes]).cuda()
            
            classifier = classifier.eval()
            #quantile=args.quantile, msc_iterations=args.msc_iterations, max_num_clusters=args.max_num_clusters, alpha=args.alpha, beta=args.beta, batch_id=i, epoch=epoch, class_list=class_list, evaluation=False 
            with torch.no_grad():
                seg_pred, _, feat, _, chamfer_loss = classifier(points, to_categorical(label, num_classes), include_convex_loss=False, visualize=False, if_cuboid=args.if_cuboid, quantile=args.quantile, msc_iterations=args.msc_iterations, max_num_clusters=args.max_num_clusters, alpha=args.alpha, beta=args.beta, batch_id=batch_id, seed=seed, class_list=class_list, evaluation=True, embed=args.embed)
           
            chamfer_loss_list.append(np.mean(chamfer_loss.cpu().data.numpy()))            
            cur_pred_val_logits = seg_pred.cpu().data.numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            points = points.transpose(2, 1)
            feat = feat.transpose(2, 1)
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]  # can be made faster..................
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                visualize = False
                if visualize:
                    directory_1 = 'k50_s-204/inputs/'
                    directory_2 = 'k50_s-204/embeddings/'
                    if not os.path.exists(directory_1):
                        os.makedirs(directory_1)
                    if not os.path.exists(directory_2):
                        os.makedirs(directory_2)

                    path1 = os.path.join(directory_1, 'batch_{}_{}_{}.xyz'.format(batch_id, i, class_list[i]))
                    path2 = os.path.join(directory_2, 'batch_{}_{}_{}.xyz'.format(batch_id, i, class_list[i]))
                    np.savetxt(path1, points[i, :, :].data.cpu().numpy())
                    np.savetxt(path2, feat[i, :, :].data.cpu().numpy())
                   
                    #save_point_cloud(path_1, points.transpose(2, 1).cpu()[i, :, :])
                    #pcd = visualize_point_cloud_from_labels(points=points.transpose(2, 1).cpu()[i], labels=cur_pred_val[i])
                    #o3d.io.write_point_cloud(path_2, pcd)

            correct = np.sum(cur_pred_val.reshape(-1, 1) == target.reshape(-1, 1))

            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)
            new_array1, new_array2 = np.zeros(num_part), np.zeros(num_part)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))  # can be made faster...................

            # total_seen_class_2_epoch = np.array(np.unique(target, return_counts=True)[1])
            # total_seen_class2_ids_epoch = np.array(np.unique(target, return_counts=True)[0])
            # np.put(new_array1, total_seen_class2_ids_epoch, total_seen_class_2_epoch)
            # total_seen_class_2 += new_array1
            #
            # diff = np.equal((cur_pred_val + 1).reshape(-1), (target + 1).reshape(-1))
            #
            # intersection = diff * (cur_pred_val + 1).reshape(-1)
            # total_correct_class_2_epoch = np.unique(intersection, return_counts=True)[1][1:]
            # total_correct_class2_ids_epoch = np.unique(intersection, return_counts=True)[0][1:] - 1
            # np.put(new_array2, total_correct_class2_ids_epoch, total_correct_class_2_epoch)
            # total_correct_class_2 += new_array2

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))  # can be made faster..............
                shape_ious[cat].append(np.mean(part_ious))

                # diff = np.equal(segp + 1, segl + 1)
                # intersection = diff * (segp + 1)
                # intersection_ids, intersection_counts = np.unique(intersection, return_counts=True)
                # np.put(intersection_parts, intersection_ids[1:] - seg_classes[cat][0] - 1, intersection_counts[1:])
                #
                # union_ids, union_counts = np.unique(np.concatenate((segp + 1, segl + 1)), return_counts=True)
                # np.put(union_parts, union_ids - seg_classes[cat][0] - 1, union_counts)
                #
                # union_parts = union_parts - intersection_parts
                # union = np.copy(union_parts)
                # union[(union_parts == 0) & (intersection_parts == 0)] = 1
                # intersection_parts[(union_parts == 0) & (intersection_parts == 0)] = 1
                # iou = intersection_parts / union
                #
                # shape_ious2[cat].append(np.mean(iou))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:  # can be made faster.............
                all_shape_ious.append(iou)  # IOU of all the instances(shapes/objects)
            shape_ious[cat] = np.mean(shape_ious[cat])  # mIOU of separate classes.

        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['chamfer_loss'] = np.mean(chamfer_loss_list)
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious  # mean of all the separate classes IOU combined.
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)  # mean of all the instances IOU combined(shapes/objects)
    if metrics != {}:
        #if  metrics['best_chamfer_loss'] >= test_metrics['chamfer_loss']:
        #if  metrics['best_instance_avg_miou'] <= test_metrics['instance_avg_iou']:
        if  metrics['best_class_avg_miou'] <= test_metrics['class_avg_iou']:
            metrics['best_chamfer_loss'] = test_metrics['chamfer_loss']
            metrics['best_epoch'] = epoch + 1
            metrics['best_acc'] = test_metrics['accuracy']
            metrics['best_class_avg_miou'] = test_metrics['class_avg_iou']
            metrics['best_instance_avg_miou'] = test_metrics['instance_avg_iou']
        log_string('Best test Accuracy: {:6f}, Best Epoch: {},  Best Class avg mIOU: {:6f}, Best Instance avg mIOU: {:6f}, Best Loss: {:6f}'.format(metrics['best_acc'],metrics['best_epoch'], metrics['best_class_avg_miou'], metrics['best_instance_avg_miou'], metrics['best_chamfer_loss']))
        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f, Loss: %f' % (
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou'], test_metrics['chamfer_loss']))
    if metrics == {}:
        log_string('Test Accuracy: %f,  Class avg mIOU: %f,   Instance avg mIOU: %f, Loss: %f' % (test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou'], test_metrics['chamfer_loss']))

    return metrics


if __name__ == '__main__':
    args = parse_args()
    evaluation(args)
