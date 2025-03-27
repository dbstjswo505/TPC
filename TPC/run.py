#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import copy

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pdb

# Detectron2/DensePose
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)

# MMPose
from mmpose.apis import MMPoseInferencer

# Segment Anything
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="DensePose + Segment-Anything + MMPose pipeline")

    # --- ref & pose input ---
    parser.add_argument("--ref", type=str, default="sample1",
        help="Reference ID (e.g., '3_people' -> 'input/ref_img/3_people.PNG')"
    )
    parser.add_argument("--pose", type=str, default="sample1",
        help="Pose (driving) video ID (e.g., '3_people' -> 'input/driving_video/3_people.mp4')"
    )

    # --- output base dir ---
    parser.add_argument("--output_dir", type=str, default="./results",
        help="Output directory to save results"
    )

    # --- DensePose setting ---
    parser.add_argument("--model_name", type=str, default="densepose_rcnn_R_101_FPN_DL_s1x",
        help="DensePose model base name (yaml/pkl). ex) densepose_rcnn_R_101_FPN_DL_s1x"
    )
    parser.add_argument("--min_score_thresh", type=float, default=0.5,
        help="Detection threshold for DensePose ROI Heads"
    )

    # --- Segment Anything setting ---
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint"
    )
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
        help="Type of the SAM model, e.g., vit_b / vit_h / vit_l"
    )

    # --- MMPose setting ---
    parser.add_argument("--mmpose_model_alias", type=str, default="human",
        help="MMPose model alias (default: 'human')"
    )

    # --- hyperparameter ---
    parser.add_argument("--max_human_num", type=int, default=3,
        help="Maximum number of humans to handle in a frame"
    )
    parser.add_argument("--area_thresh", type=float, default=10000,
        help="Threshold for bounding box area to filter out small detection"
    )
    parser.add_argument("--keypoint_score_thresh", type=float, default=0.3,
        help="Threshold for valid keypoints (face alignment, etc.)"
    )

    # --- range ---
    parser.add_argument("--dstart_frame", type=int, default=0,
        help="Start frame index for driving video"
    )
    parser.add_argument("--dend_frame", type=int, default=-1,
        help="End frame index for driving video (-1 means process all frames)"
    )

    return parser.parse_args()


def init_densepose(args):
    """
    DensePose config and predictor initialization.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg_path = f"densepose_configs/{args.model_name}.yaml"
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.min_score_thresh
    cfg.MODEL.WEIGHTS = f"{args.model_name}.pkl"

    predictor = DefaultPredictor(cfg)

    # Visualizers
    VISUALIZERS = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }
    vis_specs = ["dp_segm"]  # 
    visualizers = []
    extractors = []
    for vis_spec in vis_specs:
        vis = VISUALIZERS[vis_spec]()
        visualizers.append(vis)
        extractor_ = create_extractor(vis)
        extractors.append(extractor_)

    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)

    return predictor, visualizer, extractor


def densepose_predict(predictor, visualizer, extractor, img, area_thresh=10000):
    """
    DensePose inference and mask extraction
    Returns:
      - pose_vid: visualized pose images
      - person_mask: per person mask images (list of np.array)
      - person_bbx: per person bbox [x,y,w,h] (list)
    """
    outputs = predictor(img)['instances']
    #  purple background
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = np.tile(image_gray[:, :, np.newaxis], [1, 1, 3])
    image_gray[:, :, 0] = 84
    image_gray[:, :, 1] = 0
    image_gray[:, :, 2] = 68

    data = extractor(outputs)
    pose_vid = visualizer.visualize(image_gray, data)

    person_mask = []
    person_bbx = []
    
    for inst in range(len(outputs)):
        inst_data = extractor(outputs[inst])
        # bbox [x, y, w, h]
        bbx = inst_data[0][1][0]

        # outlier: small detection filtering
        if bbx[2] * bbx[3] < area_thresh:
            continue
        person_bbx.append(bbx)

        # black color mask
        black_bg = np.zeros_like(image_gray)
        black_viz = visualizer.visualize(black_bg, inst_data)
        mask_vid = (black_viz > 0.1).astype(np.uint8) * 255
        person_mask.append(mask_vid)

    return pose_vid, person_mask, person_bbx


def procrustes_analysis(S1, S2):
    """
    Procrustes Analysis S1 -> S2 alignment (translation, rotation, scale).
    """
    transposed = False
    if S1.shape[0] not in (2, 3):
        S1 = S1.T
        S2 = S2.T
        transposed = True

    assert S2.shape[1] == S1.shape[1], "Input shapes do not match."

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2)
    K = X1.mm(X2.T)

    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    R = V.mm(Z.mm(U.T))

    scale = torch.trace(R.mm(K)) / var1
    t = mu2 - scale * (R.mm(mu1))
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat, R, t, scale


def prepare_sam(args, device='cuda'):
    """
    Segment Anything Model(SAM) loading and Predictor initialization.
    """
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def run_mmpose_inference(mmpose_model_alias, inputs):
    """
    MMPoseInferencer-based keypoint extraction.
    """
    inferencer = MMPoseInferencer(mmpose_model_alias)
    result_generator = inferencer(inputs, show=False)
    return result_generator


def resize_and_save_image(src_path, dst_path, size=(512, 512)):
    """
    resize
    """
    img = Image.open(src_path)
    w, h = img.size
    if (w, h) != size:
        img = img.resize(size)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img.save(dst_path)


def video_to_frames(video_path, out_dir, resize_hw=(512, 512)):
    """
    frame save
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if resize_hw is not None:
            image = cv2.resize(image, resize_hw)
        out_path = os.path.join(out_dir, f"{count:03d}.jpg")
        cv2.imwrite(out_path, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"[video_to_frames] Finished converting {video_path} to frames in {out_dir}.")


def main():
    args = parse_args()

    # --ref, --pose path
    args.img_ref = f"input/ref_img/{args.ref}.PNG"             # ex: "input/ref_img/3_people.PNG"
    args.driving_video = f"input/driving_video/{args.pose}.mp4" # ex: "input/driving_video/3_people.mp4"
    args.img_poses = f"{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/pose_video/{args.pose}.mp4"

    print("[Info]")
    print(f" ref = {args.ref}, pose = {args.pose}")
    print(f" img_ref = {args.img_ref}")
    print(f" driving_video = {args.driving_video}")
    print(f" img_poses (output) = {args.img_poses}")
    print("======================================")

    # 1) DensePose initialization
    densepose_predictor, vis, extr = init_densepose(args)

    # 2) SAM initialization
    sam_predictor = prepare_sam(args, device='cuda')

    # 3) reference image resize (512x512) -> output_dir 
    ref_img_resized = os.path.join(
        args.output_dir,
        "cal_image",
        f"pose_{args.pose}_ref_{args.ref}",
        "ref_img",
        os.path.basename(args.img_ref)
    )
    resize_and_save_image(args.img_ref, ref_img_resized, size=(512, 512))

    # 4) driving_video -> pose_video path
    os.makedirs(os.path.dirname(args.img_poses), exist_ok=True)
    dcaptura = cv2.VideoCapture(args.driving_video)

    dtotal_frame = int(dcaptura.get(cv2.CAP_PROP_FRAME_COUNT))
    dwidth = int(dcaptura.get(cv2.CAP_PROP_FRAME_WIDTH))
    dheight = int(dcaptura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dfps = int(dcaptura.get(cv2.CAP_PROP_FPS))
    dfourcc = cv2.VideoWriter_fourcc(*'mp4v')

    dout_pose = cv2.VideoWriter(args.img_poses, dfourcc, dfps, (dwidth, dheight))

    dcnt = 0
    pose_mask_all = []
    pose_bbx_all = []
    prev_pose_bbx = []

    # 5) driving video -> DensePose inference & pose_video generation
    while dcaptura.isOpened():
        ret, frame = dcaptura.read()
        if not ret:
            break

        dcnt += 1
        if dcnt <= args.dstart_frame:
            continue
        if dcnt == args.dend_frame:
            break

        # DensePose process
        pose_vid, pose_mask, pose_bbx = densepose_predict(
            densepose_predictor, vis, extr,
            frame,
            area_thresh=args.area_thresh
        )
        dout_pose.write(pose_vid)

        pose_mask_all.append(pose_mask)
        pose_bbx_all.append(pose_bbx)
        prev_pose_bbx = copy.deepcopy(pose_bbx)

        print(f"[DensePose] Processed frame {dcnt}/{dtotal_frame}")

    dcaptura.release()
    dout_pose.release()

    # 6) pose_video -> frame extraction
    tmp_pose_frames = os.path.join(args.output_dir, "tmp_pose")
    shutil.rmtree(tmp_pose_frames)
    os.makedirs(tmp_pose_frames)
    video_to_frames(args.img_poses, tmp_pose_frames, resize_hw=(512, 512))

    # 원본 driving video -> frame extraction
    tmp_drive_frames = os.path.join(args.output_dir, "tmp_drive")
    shutil.rmtree(tmp_drive_frames)
    os.makedirs(tmp_drive_frames)
    video_to_frames(args.driving_video, tmp_drive_frames, resize_hw=(512, 512))

    # 7) MMPose keypoint extraction
    print("[MMPose] Inference on reference image...")
    ref_result_list = run_mmpose_inference(args.mmpose_model_alias, ref_img_resized)
    tgt_result = next(ref_result_list)

    print("[MMPose] Inference on driving frames...")
    drive_results = run_mmpose_inference(args.mmpose_model_alias, tmp_drive_frames)
    results = [result for result in drive_results]
    print("[MMPose] Inference done.")

    once_flags = [True] * len(tgt_result['predictions'][0])

    # get the saved images
    flst = os.listdir(tmp_pose_frames)
    flst.sort()

    pose_sum = np.zeros((512,512,3), dtype=bool)

    ref_image = cv2.imread(ref_img_resized)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    for i in range(len(flst)):
        
        src_path = os.path.join(tmp_pose_frames,flst[i])
        src_result = results[i]

        s_im = Image.open(src_path)
        
        # make pose image and pose mask
        sW, sH = s_im.size
        print('[Procrustes Warping] Image %d/%d'%(i+1, len(flst)))

        # preprocessing tgt prediction
        tgt_result_tmp = []
        src_result_tmp = []
        for yy in range(len(tgt_result['predictions'][0])):
            if tgt_result['predictions'][0][yy]['bbox_score'] > 0.4:
                tgt_result_tmp = tgt_result_tmp + [tgt_result['predictions'][0][yy]]
        
        for yy in range(len(src_result['predictions'][0])):
            if src_result['predictions'][0][yy]['bbox_score'] > 0.4:
                src_result_tmp = src_result_tmp + [src_result['predictions'][0][yy]]

        tgt_result['predictions'][0] = tgt_result_tmp
        src_result['predictions'][0] = src_result_tmp

        shn = len(src_result['predictions'][0])
        thn = len(tgt_result['predictions'][0])
        phn = len(pose_mask_all[i])
        if shn != thn or thn != phn or phn != shn:
            print('Error: number of human is not same')
            print(f'source {shn}, target {thn}, pose{phn}')
            print('Copy previous step')
            for h in range(len(tgt_result['predictions'][0])):
                out_dir1 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{h}-person_cal_ref_img'
                cpy_path1 = os.path.join(out_dir1, '%03d.png'%(i))
                prv_path1 = os.path.join(out_dir1, '%03d.png'%(i-1))
                shutil.copy(prv_path1, cpy_path1)

                out_dir4 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{h}-person_cal_ref_mask_img'
                cpy_path4 = os.path.join(out_dir4, '%03d.png'%(i))
                prv_path4 = os.path.join(out_dir4, '%03d.png'%(i-1))
                shutil.copy(prv_path4, cpy_path4)
                
                out_dir0 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{h}-person_pose_mask_img'
                cpy_path0 = os.path.join(out_dir0, '%03d.png'%(i))
                prv_path0 = os.path.join(out_dir0, '%03d.png'%(i-1))
                shutil.copy(prv_path0, cpy_path0)

            continue
        # Multi people, nearest neighbor ref-to-pose matching
        if len(src_result['predictions'][0]) > 1:
            # src re-sort
            changer = []
            mapper = []
            cand = torch.arange(len(src_result['predictions'][0])).tolist()
            for aa in range(len(tgt_result['predictions'][0])):
                shp = src_result['predictions'][0][aa]['bbox'][0][:2]
                cpr = 10000000.0
                for cd in cand:
                    thp = tgt_result['predictions'][0][cd]['bbox'][0][:2]
                    sc = (shp[0] - thp[0])**2 + (shp[1] - thp[1])**2
                    if sc < cpr:
                        poptgt = cd
                        cpr = sc
                cand.remove(poptgt)
                mapper.append(poptgt)
            for bb in range(len(mapper)):
                changer.append(src_result['predictions'][0][mapper[bb]])
            src_result['predictions'][0] = changer
            # pose re-sort
            changer = []
            changer2 = []
            mapper = []
            cand = torch.arange(len(pose_bbx_all[i])).tolist()
            for aa in range(len(tgt_result['predictions'][0])):
                php = pose_bbx_all[i][aa][:2]
                cpr = 10000000.0
                for cd in cand:
                    thp = tgt_result['predictions'][0][cd]['bbox'][0][:2]
                    sc = (php[0] - thp[0])**2 + (php[1] - thp[1])**2
                    if sc < cpr:
                        poptgt = cd
                        cpr = sc
                cand.remove(poptgt)
                mapper.append(poptgt)
            for bb in range(len(mapper)):
                changer.append(pose_bbx_all[i][mapper[bb]])
                changer2.append(pose_mask_all[i][mapper[bb]])
            pose_bbx_all[i] = changer
            pose_mask_all[i] = changer2
        
        for j in range(len(tgt_result['predictions'][0])):

            pout = pose_mask_all[i][j] 
            pose_out = Image.fromarray(pout, 'RGB')
            pose_crt = pose_mask_all[i][j] > 0
            pose_sum = np.logical_or(pose_sum, pose_crt)

            out_dir0 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{j}-person_pose_mask_img'
            os.makedirs(out_dir0, exist_ok=True)
            out_path0 = os.path.join(out_dir0, '%03d.png'%(i))
            pose_out.save(out_path0)
            
            # stage 1
            point = time.time()
            # weighting on face alignment
            Src_pts = src_result['predictions'][0][j]['keypoints'][:5]
            Tgt_pts = tgt_result['predictions'][0][j]['keypoints'][:5]
            Src_score = src_result['predictions'][0][j]['keypoint_scores'][:5]
            src_key = torch.tensor(Src_score) > args.keypoint_score_thresh

            src_bbx = src_result['predictions'][0][j]['bbox']
            tgt_bbx = tgt_result['predictions'][0][j]['bbox']

            src_pts = torch.tensor(Src_pts) 
            tgt_pts = torch.tensor(Tgt_pts)

            src_pts = src_pts[src_key]
            tgt_pts = tgt_pts[src_key]

            # stage 2
            cal_tgt_pts, R, t, scale = procrustes_analysis(tgt_pts, src_pts)

            # stage 3
            # PA visualizer: i don't know it is used or not...
            x = np.array(cal_tgt_pts[:,0])
            y = np.array(cal_tgt_pts[:,1])

            plt.xlim(0, 512.0)
            plt.ylim(0, 512.0)

            edges = np.array([[0,1],[3,4],[3,2],[2,4]])

            #procrutes analysis based transform
            matrix = np.zeros((3,3), dtype=np.float64)
            matrix[2,2] = 1.0
            #
            matrix[0,2] = t[0,0]
            matrix[1,2] = t[1,0]
            # 
            matrix[:2,:2] = R[:2,:2]*scale

            warped_image = cv2.warpPerspective(ref_image, matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)
            
            tmp = np.zeros((512,512,3), dtype=np.int8)
            
            cbx = src_bbx[0]
            tmp = warped_image
            # calirbrated image save
            tmp_cal = tmp.astype('uint8')
            cal_out = Image.fromarray(tmp_cal, 'RGB')

            out_dir1 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{j}-person_cal_ref_img'
            os.makedirs(out_dir1, exist_ok=True)
            out_path1 = os.path.join(out_dir1, '%03d.png'%(i))
            cal_out.save(out_path1)
            
            keypts = torch.round(cal_tgt_pts).to(torch.int)
            org_keypts = torch.round(tgt_pts).to(torch.int)

            # save calibrated image mask

            if once_flags[j]:
                first_Tgt_pts = tgt_result['predictions'][0][j]['keypoints'][:]
                first_Tgt_score = tgt_result['predictions'][0][j]['keypoint_scores'][:]

                first_tgt_key = torch.tensor(first_Tgt_score) > args.keypoint_score_thresh
                first_tgt_pts = torch.tensor(first_Tgt_pts)
                first_tgt_pts = first_tgt_pts[first_tgt_key]

                first_tgt_keypts = torch.round(first_tgt_pts).to(torch.int)
                #input_point = np.load(out_path3)
                input_point = np.array(first_tgt_keypts)
                input_label = np.array([1]*input_point.shape[0])

                sam_predictor.set_image(ref_image)
                
                masks, scores, logits = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                out = masks[0]*255
                out = np.expand_dims(out, axis=-1)
                out = np.repeat(out, 3, axis=-1)
                imout = Image.fromarray(out.astype('uint8'), 'RGB')
                out_dir5 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{j}-person_ref_mask_img'
                os.makedirs(out_dir5, exist_ok=True)
                out_path5 = os.path.join(out_dir5, 'ref.png')
                imout.save(out_path5)

            once_flags[j] = False
            #
            mask_ref_image = cv2.imread(out_path5)
            mask_ref_image = cv2.cvtColor(mask_ref_image, cv2.COLOR_BGR2RGB)
            mask_warped_image = cv2.warpPerspective(mask_ref_image, matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)

            tmp = mask_warped_image
            tmp_cal_mask_ref = tmp.astype('uint8')
            imout = Image.fromarray(tmp_cal_mask_ref, 'RGB')

            out_dir4 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/{j}-person_cal_ref_mask_img'
            os.makedirs(out_dir4, exist_ok=True)
            out_path4 = os.path.join(out_dir4, '%03d.png'%(i))
            imout.save(out_path4)
            
        
    # pose_sum = (1 - pose_sum.astype('uint8'))*255
    # pose_sum_out = Image.fromarray(pose_sum, 'RGB')
    # out_dir6 = f'{args.output_dir}/cal_image/pose_{args.pose}_ref_{args.ref}/pose_pad_mask_img'
    # os.makedirs(out_dir6, exist_ok=True)
    # out_path6 = os.path.join(out_dir6, 'pad.png')
    # imout.save(out_path6)

    dcaptura.release()
    dout_pose.release()


if __name__ == "__main__":
    main()
