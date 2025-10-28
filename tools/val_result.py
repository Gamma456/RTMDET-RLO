from argparse import ArgumentParser
import os

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules
import numpy as np
from shapely.geometry import Polygon, MultiPoint
import cv2
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img-dir', default='data/train_data/test/images', help='Image directory')
    parser.add_argument('--config', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\final\test8_sppwithLSKA\k_size=53_stage_234_120epoch_2\rotated_rtmdet_tiny-3x-dota.py', help='Config file')
    parser.add_argument('--checkpoint', default=r'C:\Users\Owner\Desktop\RTMDet_RLO\run\final\test8_sppwithLSKA\k_size=53_stage_234_120epoch_2\epoch_120.pth', help='Checkpoint file')
    parser.add_argument('--out-dir', default='work_dirs/final/box_1', help='Directory to save output images')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--gt-dir', default='data/train_data/test/labels', help='Directory of ground truth labels (in DOTA format)')
    parser.add_argument(
        '--iou-method', default='RBD', help='IoU calculation method to use')
    args = parser.parse_args()
    return args

def calculate_iou(boxA, boxB):
    # 確保框內有8個座標 (4個點)
    if len(boxA) < 8 or len(boxB) < 8:
        return 0.0
    polyA = Polygon([(boxA[0], boxA[1]), (boxA[2], boxA[3]), (boxA[4], boxA[5]), (boxA[6], boxA[7])])
    polyB = Polygon([(boxB[0], boxB[1]), (boxB[2], boxB[3]), (boxB[4], boxB[5]), (boxB[6], boxB[7])])
    if not polyA.is_valid or not polyB.is_valid:
        return 0.0
    inter_area = polyA.intersection(polyB).area
    union_area = polyA.union(polyB).area
    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

def calculate_boundary_distance(gt_box, pred_box):
    gt_points   = np.array(gt_box).reshape((4,2))
    pred_points = np.array(pred_box).reshape((4,2))


    sum_gt2pred   = 0.0
    for gt_pt in gt_points:
        dists = np.linalg.norm(pred_points - gt_pt, axis=1)
        sum_gt2pred += dists.min()

    sum_pred2gt   = 0.0
    for pred_pt in pred_points:
        dists = np.linalg.norm(gt_points - pred_pt, axis=1)
        sum_pred2gt += dists.min()

    # 最後如果要對稱平均，就把兩者加起來再除以 8
    total = (sum_gt2pred + sum_pred2gt) / 8
    # total = sum_pred2gt / 4
    # total = sum_gt2pred / 4
    # print(f"GT→Pred sum: {sum_gt2pred:.5f}")
    # print(f"Pred→GT sum: {sum_pred2gt:.5f}")
    # print(f"Boundary distance (avg): {total:.5f}")
    # boundary_distance = (sum_gt2pred + sum_pred2gt) / 8
    return total

def calculate_RBD(boxA, boxB, iou, lambda_=1, alpha=1):
    gt_points = np.array(boxA).reshape((4, 2))
    pred_points = np.array(boxB).reshape((4, 2))
    # 計算邊界距離
    boundary_distance = calculate_boundary_distance(boxA, boxB)
    # 合併所有點以建立 MultiPoint 物件
    all_points = np.vstack((gt_points, pred_points))
    multipoint = MultiPoint(all_points)
    # 計算最小旋轉包圍盒
    min_enclosing_box = multipoint.minimum_rotated_rectangle
    min_box_coords = np.array(min_enclosing_box.exterior.coords)
    if len(min_box_coords) > 4:
        min_box_coords = min_box_coords[:-1]
    # 取得包圍盒對角線長度
    c = np.sqrt((min_box_coords[0][0] - min_box_coords[2][0]) ** 2 + 
                (min_box_coords[0][1] - min_box_coords[2][1]) ** 2)
    # 邊界距離正規化
    # boundary_distance_norm = boundary_distance / c
    # 計算中心點距離（原始值）
    gt_center = np.mean(gt_points, axis=0)
    pred_center = np.mean(pred_points, axis=0)
    center_distance = np.sqrt((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2)
    normalized_center_distance = center_distance / c
    print(f"center: {center_distance:.5f}")
    print(f"boundary_distance: {boundary_distance:.5f}")
    # 此處的 RBD 定義為邊界距離與中心距離的和
    RBD = boundary_distance + center_distance
    return RBD, center_distance, boundary_distance

def calculate_iou_method(gt_box, pred_box, method='standard', lambda_=1, alpha=0.5):
    iou = calculate_iou(pred_box, gt_box)
    if method == 'RBD':
        # 回傳 tuple: (RBD, center_distance, boundary_distance)
        return calculate_RBD(pred_box, gt_box, iou, lambda_, alpha)
    return iou

def convert_rotated_box_to_polygon(rotated_box):
    # rotated_box 格式 [x_center, y_center, width, height, angle]
    x_center, y_center, width, height, angle = rotated_box
    angle_rad = angle  # 假設角度為弧度
    corners = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_corners = np.dot(corners, rotation_matrix)
    rotated_corners[:, 0] += x_center
    rotated_corners[:, 1] += y_center
    return rotated_corners.flatten().tolist()

def main(args):
    # 初始化 mmrotate 模組
    register_all_modules()
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    os.makedirs(args.out_dir, exist_ok=True)
    
    total_rbd = 0.0
    rbd_count = 0

    # 用來累積所有影像的 center 與 boundary 資料（選擇性）
    global_center_list = []
    global_boundary_list = []
    
    # 處理資料夾中每張影像
    for img_file in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img_file)
        if not img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            continue
        
        result = inference_detector(model, img_path)
        predicted_bboxes = result.pred_instances.bboxes.cpu().numpy()
        predicted_bboxes_converted = []
        for pred_box in predicted_bboxes:
            if len(pred_box) == 5:
                pred_box = convert_rotated_box_to_polygon(pred_box)
            if len(pred_box) == 8:
                predicted_bboxes_converted.append(pred_box)

        gt_file = os.path.join(args.gt_dir, img_file.replace('.png', '.txt'))
        if not os.path.exists(gt_file):
            print(f"Ground truth file for {img_file} not found.")
            continue
        
        ground_truth_bboxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    coords = list(map(float, parts[:8]))
                    if len(coords) == 8:
                        ground_truth_bboxes.append(coords)
                except ValueError:
                    print(f"Skipping invalid line in ground truth file {gt_file}: {line.strip()}")
                    continue
        
        # 為本張影像準備局部列表來記錄 center, boundary 與 RBD 距離
        image_centers = []
        image_boundaries = []
        image_rbd_list = []  # 新增，用以記錄每對的 RBD
        
        # 計算每個 predicted 與 GT 框對的 RBD
        for pred_box in predicted_bboxes_converted:
            for gt_box in ground_truth_bboxes:
                if len(gt_box) < 8:
                    continue
                if args.iou_method == 'RBD':
                    rbd, center_d, boundary_d = calculate_iou_method(gt_box, pred_box, method=args.iou_method)
                    print(f"RBD between predicted and ground truth for {img_file}: {rbd:.5f}")
                    total_rbd += rbd
                    rbd_count += 1
                    image_rbd_list.append(rbd)  # 記錄每個pair的 RBD
                    image_centers.append(center_d)
                    image_boundaries.append(boundary_d)
                    global_center_list.append(center_d)
                    global_boundary_list.append(boundary_d)
                else:
                    rbd = calculate_iou_method(gt_box, pred_box, method=args.iou_method)
                    total_rbd += rbd
                    rbd_count += 1
        
        # 顯示影像處理結果（繪製 GT 與預測框）
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        scale_factor = 3  # 放大倍率
        enlarged_img = cv2.resize(img, (img.shape[1] * scale_factor, img.shape[0] * scale_factor), interpolation=cv2.INTER_LINEAR)
        for gt_box in ground_truth_bboxes:
            gt_points = (np.array(gt_box).reshape((4, 2)) * scale_factor).astype(int)
            cv2.polylines(enlarged_img, [gt_points], isClosed=True, color=(0, 255, 0), thickness=8)
        for pred_box in predicted_bboxes:
            if len(pred_box) == 5:
                x_center, y_center, width, height, angle = pred_box
                rect = ((x_center * scale_factor, y_center * scale_factor), (width * scale_factor, height * scale_factor), np.degrees(angle))
                box_points = cv2.boxPoints(rect).astype(int)
                cv2.polylines(enlarged_img, [box_points], isClosed=True, color=(255, 0, 0), thickness=8)
            elif len(pred_box) == 8:
                pred_points = (np.array(pred_box).reshape((4, 2)) * scale_factor).astype(int)
                cv2.polylines(enlarged_img, [pred_points], isClosed=True, color=(255, 0, 0), thickness=8)
  
        if image_centers and image_boundaries and image_rbd_list:
            center_val = image_centers[0]
            boundary_val = image_boundaries[0]
            rbd_val = image_rbd_list[0]
            info_text = [
                f"Center: {center_val:.5f}",
                f"Boundary: {boundary_val:.5f}",
                f"RBD: {rbd_val:.5f}"
            ]
            # 依行繪製文字, y 軸間隔可依需求調整
            for i, text in enumerate(info_text):
                cv2.putText(enlarged_img, text, (30, 200 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255 ,0, 0), thickness=7)


        final_img = cv2.resize(enlarged_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        out_file = os.path.join(args.out_dir, img_file)
        cv2.imwrite(out_file, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    
    # 全資料集的平均 RBD
    if rbd_count > 0:
        avg_rbd = total_rbd / rbd_count
        if args.iou_method == 'RBD':
            print(f"Total RBD: {total_rbd:.5f}, Average RBD: {avg_rbd:.5f}")
    else:
        print("No RBD was calculated.")

    # 若需要，也可印出全資料集的平均 center 與 boundary distance
    if global_center_list:
        total_center = sum(global_center_list)
        avg_center_all = total_center / len(global_center_list)
        print(f"Average Center Distance: {avg_center_all:.5f}")
    if global_boundary_list:
        total_boundary = sum(global_boundary_list)
        avg_boundary_all = total_boundary / len(global_boundary_list)
        print(f"Average Boundary Distance: {avg_boundary_all:.5f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
