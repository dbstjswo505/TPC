from mmpose.apis import MMPoseInferencer
from segment_anything import SamPredictor, sam_model_registry
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
import pdb
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import shutil
import copy
import time


def Procrustes_Analysis(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat, R, t, scale

def prediction_optimizer(key_pts, out_path, predictor):
    input_point = np.array(key_pts)
    input_label = np.array([1]*input_point.shape[0])

    x_min = input_point[:,0].min()
    x_max = input_point[:,0].max()
    
    y_min = input_point[:,1].min()
    y_max = input_point[:,1].max()
    
    cal_ref_image = cv2.imread(out_path)
    cal_ref_image = cv2.cvtColor(cal_ref_image, cv2.COLOR_BGR2RGB)
    pad = np.zeros((3,512,512), dtype=bool)
    
    
    input_patch = cal_ref_image[x_min:x_max, y_min:y_max, :]
    re_point = np.zeros_like(input_point)
    re_point[:,0] = input_point[:,0] - x_min
    re_point[:,1] = input_point[:,1] - y_min

    predictor.set_image(input_patch)
    
    masks, scores, logits = predictor.predict(
        point_coords=re_point,
        point_labels=input_label,
        multimask_output=True,
    )
    pad[:, x_min:x_max, y_min:y_max] = masks[:,:,:]

    return pad

ref='S_12'
pose='S_12'
have_pose_video = False
#ref='my_multiperson'
#pose='multi_dancing'
img_ref = f'input/ref_img/{ref}.PNG'   # replace this with your own image path
driving_video = f'input/driving_video/{pose}.mp4'

if have_pose_video:
    img_poses = f'input/pose_img/{pose}.mp4'
    driving_video = None
else:
    print('make pose video and pose mask')
    img_poses = f'./results/cal_image/pose_{pose}_ref_{ref}/pose_video/{pose}.mp4'

# DensePose load
###########################################################################################
# Please remember to download the model from densepose github
# model_name = 'densepose_rcnn_R_50_FPN_s1x'
model_name = 'densepose_rcnn_R_101_FPN_DL_s1x'
dstart_frame = 0 # 0 for all frames
dend_frame = -1 # -1 for all frames
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(f"densepose_configs/{model_name}.yaml")
cfg.MODEL.DEVICE = "cuda"
#cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = f"{model_name}.pkl"
densepose_predictor = DefaultPredictor(cfg)
VISUALIZERS = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer, # I've changed this class
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
}
vis_specs = ['dp_segm']
visualizers = []
extractors = []
for vis_spec in vis_specs:
    vis = VISUALIZERS[vis_spec]()
    visualizers.append(vis)
    extractor = create_extractor(vis)
    extractors.append(extractor)
visualizer = CompoundVisualizer(visualizers)
extractor = CompoundExtractor(extractors)
context = {
    "extractor": extractor,
    "visualizer": visualizer
    }

def predict(img):
    outputs = densepose_predictor(img)['instances']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    # Fixed value for purple background
    image = np.ones_like(image)
    image[:,:,0] *= 84 #81
    image[:,:,1] *= 0  # 0
    image[:,:,2] *= 68 #65
    pose_vid = visualizer.visualize(image, data)
    
    person_mask = []
    person_bbx = []

    for xx in range(len(outputs)):
        data_person = extractor(outputs[xx])
        person_bbx = person_bbx + [data_person[0][1][0]]
        #data1 = extractor(outputs[0])
        #data2 = extractor(outputs[1])
        # Fixed value for black background
        mimage = np.ones_like(image)
        mimage[:,:,0] *= 0 #81
        mimage[:,:,1] *= 0  # 0
        mimage[:,:,2] *= 0 #65
        mask_vid = visualizer.visualize(mimage, data_person)
        mask_vid = mask_vid > 0.1
        mask_vid = mask_vid.astype(np.uint8)*255
        person_mask = person_mask + [mask_vid]
    return pose_vid, person_mask, person_bbx

output_pose_path = f'./results/cal_image/pose_{pose}_ref_{ref}/pose_video'
#output_mask_path = f'./results/cal_image/pose_{pose}_ref_{ref}/pose_mask_img'
os.makedirs(output_pose_path, exist_ok=True)
#os.makedirs(output_mask_path, exist_ok=True)

if driving_video is not None:
    dcaptura = cv2.VideoCapture(driving_video)
    
    dtotal_frame = int(dcaptura.get(cv2.CAP_PROP_FRAME_COUNT))
    dwidth = int(dcaptura.get(cv2.CAP_PROP_FRAME_WIDTH))
    dheight = int(dcaptura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dfps = int(dcaptura.get(cv2.CAP_PROP_FPS))
    dfourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_pose_video = os.path.join(output_pose_path, f'{pose}.mp4')
    #output_mask_video = os.path.join(output_mask_path, f'{pose}.mp4')
    
    dout_pose = cv2.VideoWriter(output_pose_video, dfourcc, dfps, (dwidth, dheight))
    #dout_mask = cv2.VideoWriter(output_mask_video, dfourcc, dfps, (dwidth, dheight))
    dcnt = 0
    pose_mask_all = []
    pose_bbx_all = []
    fix_pose_num = 1
    prev_pose_bbx = []
    while dcaptura.isOpened():
        print('Progress: %d / %d' % (dcnt, dtotal_frame))
        dcnt += 1
        ret, frame = dcaptura.read()
        if not ret:
            break
        # Your own frame range
        if dcnt <= dstart_frame:
            continue
        if dcnt == dend_frame:
            break
        pose_vid, pose_mask, pose_bbx = predict(frame)
        #dout_pose.write(pose_vid)
        #dout_mask.write(mask_vid)
        pose_mask_tmp = []
        pose_bbx_tmp = []
        # remove outlier by size        
        for ee in range(len(pose_bbx)):
            if pose_bbx[ee][2] * pose_bbx[ee][3] > 20000:
                pose_bbx_tmp = pose_bbx_tmp + [pose_bbx[ee]]
                pose_mask_tmp = pose_mask_tmp + [pose_mask[ee]]
    
        pose_mask = pose_mask_tmp
        pose_bbx = pose_bbx_tmp
        # remove outlier previous step comparing
        outlier_remove_mask_tmp = []
        outlier_remove_bbx_tmp = []
        if len(pose_bbx) > fix_pose_num:
            for uu in range(fix_pose_num):
                ppb = prev_pose_bbx[uu] 
                tmpt = []
                for kk in range(len(pose_bbx)):
                    aaa = ppb - pose_bbx[kk]
                    tmpt.append((aaa * aaa).sum())
                tid = torch.tensor(tmpt).argmin()
                outlier_remove_mask_tmp = outlier_remove_mask_tmp + [pose_mask[tid]]
                outlier_remove_bbx_tmp = outlier_remove_bbx_tmp + [pose_bbx[tid]]
                pose_mask.pop(tid)
                pose_bbx.pop(tid)
            pose_mask = outlier_remove_mask_tmp
            pose_bbx = outlier_remove_bbx_tmp
    
        # remove outlier in the pose video
        if fix_pose_num == 1:
            for uuu in range(len(pose_bbx_tmp)):
                x = int(pose_bbx_tmp[uuu][0])
                y = int(pose_bbx_tmp[uuu][1])
                w = int(pose_bbx_tmp[uuu][2])
                h = int(pose_bbx_tmp[uuu][3])
    
                pose_vid[:y,:,0] = 84
                pose_vid[:y,:,1] = 0
                pose_vid[:y,:,2] = 68
                
                pose_vid[:,:x,0] = 84
                pose_vid[:,:x,1] = 0
                pose_vid[:,:x,2] = 68
                
                pose_vid[y+h:,:,0] = 84
                pose_vid[y+h:,:,1] = 0
                pose_vid[y+h:,:,2] = 68
                
                pose_vid[:,x+w:,0] = 84
                pose_vid[:,x+w:,1] = 0
                pose_vid[:,x+w:,2] = 68
        dout_pose.write(pose_vid)
    
        prev_pose_bbx = copy.deepcopy(pose_bbx_tmp)
        pose_mask_all = pose_mask_all + [pose_mask_tmp]
        pose_bbx_all = pose_bbx_all + [pose_bbx_tmp]
        
    dcaptura.release()
    dout_pose.release()
    #dout_mask.release()
else:
    output_pose_video = os.path.join(output_pose_path, f'{pose}.mp4')
    shutil.copy(img_poses, output_pose_video)

###########################################################################################

# get reference image
cim = Image.open(img_ref)
w,h = cim.size
if w != 512:
    cimr = cim.resize((512,512))
    cimr.save(img_ref)

cim_ref = f'./results/cal_image/pose_{pose}_ref_{ref}/ref_img'
os.makedirs(cim_ref, exist_ok=True)
cim_path = os.path.join(cim_ref, f'{ref}.PNG')
cim.save(cim_path)

# get target video and temporally save into images
tmp_path = './input/pose_img/tmp'
shutil.rmtree(tmp_path)
os.makedirs(tmp_path)

vidcap = cv2.VideoCapture(img_poses)
success,image = vidcap.read()
count = 0
while success:
    s_p = os.path.join(tmp_path,'%03d.jpg' % count)
    w,h,_ = image.shape
    if w != 512:
        image = cv2.resize(image, (512, 512))
    cv2.imwrite(s_p, image)     # save frame as JPEG file
    success,image = vidcap.read()
    count += 1
print("finish! convert pose video to frame")
#############################################################

tmp_path2 = './input/pose_img/tmp2'
shutil.rmtree(tmp_path2)
os.makedirs(tmp_path2)

vidcap = cv2.VideoCapture(driving_video)
success,image = vidcap.read()
count = 0
while success:
    s_p = os.path.join(tmp_path2,'%03d.jpg' % count)
    w,h,_ = image.shape
    if w != 512:
        image = cv2.resize(image, (512, 512))
    cv2.imwrite(s_p, image)     # save frame as JPEG file
    success,image = vidcap.read()
    count += 1
print("finish! convert video to frame")
############################################################

# get the saved images
flst = os.listdir(tmp_path)
flst.sort()

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# debug
#img_path = 'input/11.jpg'
#result_generator = inferencer(img_path, show=False, out_dir='output', black_background=True, radius=4, thickness=2)
#src_result = next(result_generator)
#pdb.set_trace()

# ref_img pose estimation
result_generator = inferencer(img_ref, show=False)
tgt_result = next(result_generator)
ref_image = cv2.imread(img_ref)
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

max_human = 3
out = np.zeros((len(flst),max_human,17,4,))

# SAM load
# largest model 2.4 GB
#sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# smallest model 300 MB
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam = sam.to(device='cuda')
predictor = SamPredictor(sam)
thr = 0.3

once_flags = [True] * len(tgt_result['predictions'][0])

#
#use driving video
result_generator = inferencer(tmp_path2, show=False)

# save openpose
#openpose_result_generator = inferencer(tmp_path2, show=False, skeleton_style='openpose', out_dir='output', black_background=True, radius=6, thickness=4)
#src_result = next(openpose_result_generator)
#while src_result:
#    src_result = next(openpose_result_generator)
#pdb.set_trace()

#use pose video
#result_generator = inferencer(tmp_path, show=False)

results = [result for result in result_generator]

pose_sum = np.zeros((512,512,3), dtype=bool)
for i in range(len(flst)):
    
    src_path = os.path.join(tmp_path,flst[i])
    #result_generator = inferencer(src_path, show=False)
    #src_result = next(result_generator)
    src_result = results[i]

    s_im = Image.open(src_path)
    
    # make pose image and pose mask
    sW, sH = s_im.size
    print('Image %03d is working'%(i))

    # preprocessing tgt prediction
    tgt_result_tmp = []
    src_result_tmp = []
    for yy in range(len(tgt_result['predictions'][0])):
        if tgt_result['predictions'][0][yy]['bbox_score'] > 0.5:
            tgt_result_tmp = tgt_result_tmp + [tgt_result['predictions'][0][yy]]
    
    for yy in range(len(src_result['predictions'][0])):
        if src_result['predictions'][0][yy]['bbox_score'] > 0.5:
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
            out_dir1 = f'./results/cal_image/pose_{pose}_ref_{ref}/{h}-person_cal_ref_img'
            cpy_path1 = os.path.join(out_dir1, '%03d.png'%(i))
            prv_path1 = os.path.join(out_dir1, '%03d.png'%(i-1))
            shutil.copy(prv_path1, cpy_path1)
            
            #out_dir2 = f'./results/cal_image/pose_{pose}_ref_{ref}/{h}-person_cal_ref_key'
            #cpy_path2 = os.path.join(out_dir2, '%03d.npy'%(i))
            #prv_path2 = os.path.join(out_dir2, '%03d.npy'%(i-1))
            #shutil.copy(prv_path2, cpy_path2)

            out_dir4 = f'./results/cal_image/pose_{pose}_ref_{ref}/{h}-person_cal_ref_mask_img'
            cpy_path4 = os.path.join(out_dir4, '%03d.png'%(i))
            prv_path4 = os.path.join(out_dir4, '%03d.png'%(i-1))
            shutil.copy(prv_path4, cpy_path4)
            
            out_dir0 = f'./results/cal_image/pose_{pose}_ref_{ref}/{h}-person_pose_mask_img'
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

        out_dir0 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_pose_mask_img'
        os.makedirs(out_dir0, exist_ok=True)
        out_path0 = os.path.join(out_dir0, '%03d.png'%(i))
        pose_out.save(out_path0)
        
        # stage 1
        point = time.time()
        # weighting on face alignment
        Src_pts = src_result['predictions'][0][j]['keypoints'][:5]
        Tgt_pts = tgt_result['predictions'][0][j]['keypoints'][:5]
        Src_score = src_result['predictions'][0][j]['keypoint_scores'][:5]
        src_key = torch.tensor(Src_score) > thr

        src_bbx = src_result['predictions'][0][j]['bbox']
        tgt_bbx = tgt_result['predictions'][0][j]['bbox']

        src_pts = torch.tensor(Src_pts) 
        tgt_pts = torch.tensor(Tgt_pts)

        src_pts = src_pts[src_key]
        tgt_pts = tgt_pts[src_key]

        # stage 2
        cal_tgt_pts, R, t, scale = Procrustes_Analysis(tgt_pts, src_pts)

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
        # scale fix
        #matrix[:2,:2] = R[:2,:2]

        # affine transform
        #X1 = src_pts[0].tolist()
        #X2 = src_pts[5].tolist()
        #X3 = src_pts[6].tolist()
        #Y1 = tgt_pts[0].tolist()
        #Y2 = tgt_pts[5].tolist()
        #Y3 = tgt_pts[6].tolist()

        #pts1 = np.array([X1, X2, X3], dtype=np.float32)
        #pts2 = np.array([Y1, Y2, Y3], dtype=np.float32)

        #aff_matrix = cv2.getAffineTransform(pts2, pts1)

        warped_image = cv2.warpPerspective(ref_image, matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)
        #warped_image = cv2.warpAffine(ref_image, aff_matrix, (ref_image.shape[1], ref_image.shape[0]))
        
        tmp = np.zeros((512,512,3), dtype=np.int8)
        
        cbx = src_bbx[0]

        tmp = warped_image
        # calirbrated image save
        tmp_cal = tmp.astype('uint8')
        cal_out = Image.fromarray(tmp_cal, 'RGB')

        out_dir1 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_cal_ref_img'
        os.makedirs(out_dir1, exist_ok=True)
        out_path1 = os.path.join(out_dir1, '%03d.png'%(i))
        cal_out.save(out_path1)
        
        # calibrated_keypoints save (option)
        #out_dir2 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_cal_ref_key'
        #os.makedirs(out_dir2, exist_ok=True)
        keypts = torch.round(cal_tgt_pts).to(torch.int)
        #out_path2 = os.path.join(out_dir2, '%03d.npy'%(i))
        #np.save(out_path2, keypts)
        
        # reference_keypoints save (option)
        #out_dir3 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_ref_key'
        #os.makedirs(out_dir3, exist_ok=True)
        #out_path3 = os.path.join(out_dir3, 'ref.npy')
        org_keypts = torch.round(tgt_pts).to(torch.int)
        #if once_flags[j]:
        #    np.save(out_path3, org_keypts)
        
        
        # save calibrated image mask

        #input_point = np.load(out_path2)
        #masks = prediction_optimizer(keypts, out_path1, predictor)
        if once_flags[j]:
            first_Tgt_pts = tgt_result['predictions'][0][j]['keypoints'][:]
            first_Tgt_score = tgt_result['predictions'][0][j]['keypoint_scores'][:]

            first_tgt_key = torch.tensor(first_Tgt_score) > thr
            first_tgt_pts = torch.tensor(first_Tgt_pts)
            first_tgt_pts = first_tgt_pts[first_tgt_key]

            first_tgt_keypts = torch.round(first_tgt_pts).to(torch.int)
            #input_point = np.load(out_path3)
            input_point = np.array(first_tgt_keypts)
            input_label = np.array([1]*input_point.shape[0])

            predictor.set_image(ref_image)
            
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            out = masks[0]*255
            out = np.expand_dims(out, axis=-1)
            out = np.repeat(out, 3, axis=-1)
            imout = Image.fromarray(out.astype('uint8'), 'RGB')
            out_dir5 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_ref_mask_img'
            os.makedirs(out_dir5, exist_ok=True)
            out_path5 = os.path.join(out_dir5, 'ref.png')
            imout.save(out_path5)

        once_flags[j] = False

        #input_point = np.array(first_tgt_keypts)
        #input_label = np.array([1]*input_point.shape[0])

        #
        mask_ref_image = cv2.imread(out_path5)
        mask_ref_image = cv2.cvtColor(mask_ref_image, cv2.COLOR_BGR2RGB)
        mask_warped_image = cv2.warpPerspective(mask_ref_image, matrix, (ref_image.shape[1], ref_image.shape[0]), flags=cv2.INTER_CUBIC)
        #predictor.set_image(cal_ref_image)
        
        #masks, scores, logits = predictor.predict(
        #    point_coords=input_point,
        #    point_labels=input_label,
        #    multimask_output=True,
        #)
        tmp = mask_warped_image
        tmp_cal_mask_ref = tmp.astype('uint8')
        imout = Image.fromarray(tmp_cal_mask_ref, 'RGB')
        
        #out = masks[0]*255
        #out = np.expand_dims(out, axis=-1)
        #crout = np.repeat(out, 3, axis=-1)
        #imout = Image.fromarray(crout.astype('uint8'), 'RGB')
        #imout = Image.fromarray(cal_mask_ref_out)

        out_dir4 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_cal_ref_mask_img'
        os.makedirs(out_dir4, exist_ok=True)
        out_path4 = os.path.join(out_dir4, '%03d.png'%(i))
        imout.save(out_path4)
       
        # union cr mas and p mask
        #out_dir6 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_pose_mask_img'
        #os.makedirs(out_dir6, exist_ok=True)
        #out_path6 = os.path.join(out_dir6, '%03d.png'%(i))
        #pcrout = pout + crout
        #pcrout = np.clip(pcrout, 0, 255)
        #pcr_out = Image.fromarray(pcrout.astype('uint8'), 'RGB')
        #pcr_out.save(out_path6)

        # update calibrated reference using mask (only for the qualitative results)
        #mask_cal = masks[2]
        #mask_cal = np.expand_dims(mask_cal, axis=-1)
        #tmp_cal = tmp_cal * mask_cal
        #cal_out = Image.fromarray(tmp_cal, 'RGB') 

        #out_dir1 = f'./results/cal_image/pose_{pose}_ref_{ref}/{j}-person_cal_ref_img'
        #os.makedirs(out_dir1, exist_ok=True)
        #out_path1 = os.path.join(out_dir1, '%03d.png'%(i))
        #cal_out.save(out_path1)
        
    
pose_sum = (1 - pose_sum.astype('uint8'))*255
pose_sum_out = Image.fromarray(pose_sum, 'RGB')
out_dir6 = f'./results/cal_image/pose_{pose}_ref_{ref}/pose_pad_mask_img'
os.makedirs(out_dir6, exist_ok=True)
out_path6 = os.path.join(out_dir6, 'pad.png')
imout.save(out_path6)


    
dcaptura.release()
dout_pose.release()
