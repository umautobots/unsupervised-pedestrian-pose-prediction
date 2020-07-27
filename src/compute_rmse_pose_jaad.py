#! /usr/bin/python3

'''
Code for computing RMSE results for pedestrian poses (OpenPose detection results), after filtering out errors detected by Mask R-CNN
save RMSE_x, RMSE_y, RMSE_xy for openpose skeletons between X_test and X_hat, in test data

Latest revision by X. Du 2020/01

'''

import os
import numpy as np
import json
import pickle as pkl
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
import math
from jaad_settings import *

def intersection_over_union(box_a, box_b):
    # Determine the coordinates of each of the two boxes
    xA = max(box_a['left'], box_b['left'])
    yA = max(box_a['top'], box_b['top'])
    xB = min(box_a['left'] + box_a['width'], box_b['left'] + box_b['width'])
    yB = min(box_a['top'] + box_a['height'], box_b['top'] + box_b['height'])

    # Calculate the area of the intersection area
    area_of_intersection = (xB - xA + 1) * (yB - yA + 1)

    # Calculate the area of both rectangles
    box_a_area = (box_a['width'] + 1) * (box_a['height'] + 1)
    box_b_area = (box_b['width'] + 1) * (
                box_b['height'] + 1)  # Calculate the area of intersection divided by the area of union
    # Area of union = sum both areas less the area of intersection
    iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)

    # Return the score
    return iou


# ### the list of all hkl data filenames
hkl_filename_all = []
for fileList_test_idx in range(len(fileList_test)):  
    fileList_test_name = fileList_test[fileList_test_idx]
    seq_id = fileList_test_name[-10:-4]  #example: '000434', indicate which saved hkl X sequence is for test
    hkl_filename = X_seq_filename_dir_python3 + 'X_seq_filename_' + seq_id + '.pkl'
    hkl_file = pkl.load(open(hkl_filename,'rb'))
    hkl_filename_all = hkl_filename_all + hkl_file   # list of shape [NumSequencesVideos][nt], which matches the indices of 'X_hat_images_openpose_json' folder


# ### compute RMSE 
Seq_nt_framenames_all = []
RMSE_x=np.array([]).reshape(0,)
RMSE_y=np.array([]).reshape(0,)
RMSE_xy=np.array([]).reshape(0,)
RMSE_joints = np.array([]).reshape(0,25)
pose_keypoints_75_x_X_test_orig_all = np.array([]).reshape(0,25)
pose_keypoints_75_y_X_test_orig_all = np.array([]).reshape(0,25)
pose_keypoints_75_x_X_hat_orig_all = np.array([]).reshape(0,25)
pose_keypoints_75_y_X_hat_orig_all = np.array([]).reshape(0,25)

r_rois_area = np.array([]).reshape(0,)
r_rois_locations = np.array([]).reshape(0,4)
video_idx_all = np.array([]).reshape(0,)
frame_idx_all = np.array([]).reshape(0,)
RMSE_counter = 0
occluded_count = 0
for fileList_test_idx in range(len(hkl_filename_all)):#range(len(hkl_filename_all)):  # range(len(hkl_filename_all)):
    for j in range(1, nt):  # range(1,nt):  nt=30, the first frame X_hat is all black (spatially uniform due to PredNet initial layer setting)
        print('----------Processing Seq #' + str(fileList_test_idx) + ' out of ' + str(len(hkl_filename_all)) + ', nt=' + str(j) + '----------------')
        openpose_filename_X_test = X_test_images_openpose_json_dir + 'Seq{:05d}_nt{:02d}_keypoints.json'.format(fileList_test_idx, j)
        data_op_json_X_test = json.load(open(openpose_filename_X_test))
        op_NumPeds_X_test = len(data_op_json_X_test['people'])
        pose_keypoints_75_xy_medians_X_test = np.empty((op_NumPeds_X_test, 2))
        for kk in range(op_NumPeds_X_test):
            pose_keypoints_75_X_test = np.array(data_op_json_X_test['people'][kk]['pose_keypoints_2d'])
            pose_keypoints_75_x_X_test = pose_keypoints_75_X_test[::3].astype(int)
            pose_keypoints_75_y_X_test = pose_keypoints_75_X_test[1::3].astype(int)
            pose_keypoints_75_xy_median_X_test = np.array([np.median(pose_keypoints_75_x_X_test), np.median(pose_keypoints_75_y_X_test)])
            pose_keypoints_75_xy_medians_X_test[kk, :] = pose_keypoints_75_xy_median_X_test.reshape((1,2))

        openpose_filename_X_hat = X_hat_images_openpose_json_dir + 'Seq{:05d}_nt{:02d}_keypoints.json'.format(
            fileList_test_idx, j)
        data_op_json_X_hat = json.load(open(openpose_filename_X_hat))
        op_NumPeds_X_hat = len(data_op_json_X_hat['people'])
        pose_keypoints_75_xy_medians_X_hat = np.empty((op_NumPeds_X_hat, 2))
        for kk in range(op_NumPeds_X_hat):
            pose_keypoints_75_X_hat = np.array(data_op_json_X_hat['people'][kk]['pose_keypoints_2d'])
            pose_keypoints_75_x_X_hat = pose_keypoints_75_X_hat[::3].astype(int)
            pose_keypoints_75_y_X_hat = pose_keypoints_75_X_hat[1::3].astype(int)
            pose_keypoints_75_xy_median_X_hat = np.array(
                [np.median(pose_keypoints_75_x_X_hat), np.median(pose_keypoints_75_y_X_hat)])
            pose_keypoints_75_xy_medians_X_hat[kk, :] = pose_keypoints_75_xy_median_X_hat.reshape((1, 2))

        hkl_filename_each = hkl_filename_all[fileList_test_idx][j]
        video_idx = int(hkl_filename_each[-14:-10]) - 1  ##'.../video_0006/00121.png'.Note by adding"-1" the video_idx will be 0-based.
        frame_idx = int(hkl_filename_each[-9:-4])
        # vbb_json_filename = vbb_data_path + 'video_{:04d}/annotations/I{:05d}.json'.format(video_idx + 1, frame_idx)
        # vbb_json_data_orig = json.load(open(vbb_json_filename))
        # NumPeds_vbb = len(vbb_json_data_orig)
        # for ped_idx in range(NumPeds_vbb):
        #     ped_vbb_pos = np.array(vbb_json_data_orig[ped_idx]['pos'])

        ##### compare to the mask RCNN bounding boxes extracted on X_test image
        save_pkl_name_python3 = os.path.join(X_test_images_MaskRCNN_py3_dir,  'Seq{:05d}_nt{:02d}_r.pkl'.format(fileList_test_idx, j))
        maskrcnn_data = pkl.load(open(save_pkl_name_python3, "rb"))
        maskrcnn_NumPed = len(maskrcnn_data['rois'])
        r_rois_centers = np.empty((maskrcnn_NumPed,2))
        r_rois_widths = np.empty((maskrcnn_NumPed,))
        r_rois_heights = np.empty((maskrcnn_NumPed,))
        for jj in range(maskrcnn_NumPed):
            ###r_rois = maskrcnn_data['rois'][jj]  # correct, un-needed, (NumPersons, 4)
            y1, x1, y2, x2 = maskrcnn_data['rois'][jj]
            r_rois_width = x2 - x1
            r_rois_height = y2 - y1
            r_rois_center = np.array([int((x1+x2)/2),int((y1+y2)/2)])
            r_rois_centers[jj,:] = r_rois_center.reshape((1,2))
            r_rois_widths[jj] = r_rois_width
            r_rois_heights[jj] = r_rois_height

            ##print(np.linalg.norm(pose_keypoints_75_xy_median - r_rois_center))

        # ##### plot mask r-cnn bounding boxes over X_test_images, correct (optional)
        # import skimage.io
        # import matplotlib
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # fig = plt.figure()
        # ax3 = fig.add_subplot(111)
        # image = skimage.io.imread(X_test_images_dir + 'Seq{:05d}_nt{:02d}.png'.format(fileList_test_idx, j))
        # ax3.imshow(image.astype(np.uint8))
        # for ii in range(maskrcnn_NumPed):
        #     y1, x1, y2, x2 = maskrcnn_data['rois'][ii]
        #     p = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
        #                           alpha=0.7, edgecolor='r', facecolor='none')  #linestyle="dashed"
        #     ax3.add_patch(p)
        # plt.show()
        

        if len(r_rois_centers) > 0 and len(pose_keypoints_75_xy_medians_X_test) > 0 and len(pose_keypoints_75_xy_medians_X_hat) > 0:
            r_rois_centers_distances_X_test = euclidean_distances(r_rois_centers, pose_keypoints_75_xy_medians_X_test)  # pair-wiseEuclidean distance, shape (maskrcnn_NumPed,op_NumPeds)
            r_distances_min_X_test = np.min(r_rois_centers_distances_X_test, axis=0)  #shape(op_NumPeds_X_test,)
            r_distances_indices_X_test = np.argmin(r_rois_centers_distances_X_test, axis=0) #shape(op_NumPeds_X_test,)

            r_rois_centers_distances_X_hat = euclidean_distances(r_rois_centers, pose_keypoints_75_xy_medians_X_hat)  # pair-wiseEuclidean distance, shape (maskrcnn_NumPed,op_NumPeds)
            r_distances_min_X_hat = np.min(r_rois_centers_distances_X_hat, axis=0)  # shape(op_NumPeds_X_hat,)
            r_distances_indices_X_hat = np.argmin(r_rois_centers_distances_X_hat, axis=0)  # shape(op_NumPeds_X_hat,)

            r_distances_indices_X_intersect = np.intersect1d(r_distances_indices_X_hat, r_distances_indices_X_test)  #corresponds to the same mask rcnn bounding box--assumed to be the same pedestrian

            if len(r_distances_indices_X_intersect) > 0:
                for r1 in range(len(r_distances_indices_X_intersect)):  #number of maskrcnn bounding box that has pedestrians detected
                    r_distances_indices_X_intersect_idx = r_distances_indices_X_intersect[r1]  # corresponds to mask rcnn bounding box indices
                    k1 = np.where(r_distances_indices_X_test == r_distances_indices_X_intersect_idx)[0] # index in op_NumPeds_X_test. non-empty
                    k2 = np.where(r_distances_indices_X_hat == r_distances_indices_X_intersect_idx)[0] # index in op_NumPeds_X_hat. non-empty
                    if len(k1) > 1:  # If more than one openpose skeleton corresponds to the same bounding box, pick the one that's closest to mask rcnn bounding box center
                        tmp_min_idx = np.argmin(r_distances_min_X_test[k1])
                        k1 = k1[tmp_min_idx]
                    if len(k2) > 1:
                        tmp_min_idx = np.argmin(r_distances_min_X_hat[k2])
                        k2 = k2[tmp_min_idx]
                    
                    k1 = int(k1) # make integer index. Currently, both k1 and k2 will be forced to have len(k1)==len(k2)==1.
                    k2 = int(k2)

                    if r_distances_min_X_test[k1] <= np.min((r_rois_widths[r_distances_indices_X_intersect_idx],r_rois_heights[r_distances_indices_X_intersect_idx]))\
                        and r_distances_min_X_hat[k2] <= np.min((r_rois_widths[r_distances_indices_X_intersect_idx],r_rois_heights[r_distances_indices_X_intersect_idx])):  # if openpose skeleton is within Mask R-CNN ROI
                        ##### extra filter: if they are within mask rcnn bounding box

                        pose_keypoints_75_X_test = np.array(data_op_json_X_test['people'][k1]['pose_keypoints_2d']) #(75,), original
                        pose_keypoints_75_x_X_test_orig = pose_keypoints_75_X_test[::3].astype(int)  #shape (25,),original
                        pose_keypoints_75_y_X_test_orig = pose_keypoints_75_X_test[1::3].astype(int) #shape (25,), original
                        pose_keypoints_75_x_X_test = np.copy(pose_keypoints_75_x_X_test_orig)#shape (25,) -->will be reduced to occluded
                        pose_keypoints_75_y_X_test = np.copy(pose_keypoints_75_y_X_test_orig) #shape (25,) -->will be reduced to occluded
                        pose_keypoints_75_xy_X_test = np.copy(pose_keypoints_75_X_test).astype(int)
                        pose_keypoints_75_xy_X_test = np.delete(pose_keypoints_75_xy_X_test, np.arange(2, pose_keypoints_75_xy_X_test.size, 3)) #shape(50,) -->will be reduced to occluded
                        pose_keypoints_75_X_hat = np.array(data_op_json_X_hat['people'][k2]['pose_keypoints_2d'])
                        pose_keypoints_75_x_X_hat_orig = pose_keypoints_75_X_hat[::3].astype(int)
                        pose_keypoints_75_y_X_hat_orig = pose_keypoints_75_X_hat[1::3].astype(int)
                        pose_keypoints_75_x_X_hat = np.copy(pose_keypoints_75_x_X_hat_orig)
                        pose_keypoints_75_y_X_hat = np.copy(pose_keypoints_75_y_X_hat_orig)
                        pose_keypoints_75_xy_X_hat = np.copy(pose_keypoints_75_X_hat).astype(int)
                        pose_keypoints_75_xy_X_hat = np.delete(pose_keypoints_75_xy_X_hat, np.arange(2, pose_keypoints_75_xy_X_hat.size, 3))

                        ############# exclude occluded joints #######
                        occluded_joint_idx_x_test = np.where(pose_keypoints_75_x_X_test_orig == 0)[0]
                        occluded_joint_idx_x_hat = np.where(pose_keypoints_75_x_X_hat_orig == 0)[0]
                        occluded_joint_idx_x = np.union1d(occluded_joint_idx_x_test, occluded_joint_idx_x_hat)
                        pose_keypoints_75_x_X_test = np.delete(pose_keypoints_75_x_X_test, occluded_joint_idx_x)
                        pose_keypoints_75_x_X_hat = np.delete(pose_keypoints_75_x_X_hat, occluded_joint_idx_x)
                        RMSE_x_each = math.sqrt(mean_squared_error(pose_keypoints_75_x_X_test, pose_keypoints_75_x_X_hat))

                        if len(occluded_joint_idx_x_hat) > len(occluded_joint_idx_x_test):
                            occluded_count += 1
                            print('----X_hat more occluded in ' + 'Seq{:05d}_nt{:02d}'.format(fileList_test_idx, j) + '------')

                        occluded_joint_idx_y_test = np.where(pose_keypoints_75_y_X_test_orig == 0)[0]
                        occluded_joint_idx_y_hat = np.where(pose_keypoints_75_y_X_hat_orig == 0)[0]
                        occluded_joint_idx_y = np.union1d(occluded_joint_idx_y_test, occluded_joint_idx_y_hat)
                        pose_keypoints_75_y_X_test = np.delete(pose_keypoints_75_y_X_test, occluded_joint_idx_y)
                        pose_keypoints_75_y_X_hat = np.delete(pose_keypoints_75_y_X_hat, occluded_joint_idx_y)
                        RMSE_y_each = math.sqrt(mean_squared_error(pose_keypoints_75_y_X_test, pose_keypoints_75_y_X_hat))

                        occluded_joint_idx_xy_test = np.where(pose_keypoints_75_xy_X_test == 0)[0]
                        occluded_joint_idx_xy_hat= np.where(pose_keypoints_75_xy_X_hat == 0)[0]
                        occluded_joint_idx_xy = np.union1d(occluded_joint_idx_xy_test, occluded_joint_idx_xy_hat)
                        pose_keypoints_75_xy_X_test = np.delete(pose_keypoints_75_xy_X_test, occluded_joint_idx_xy)
                        pose_keypoints_75_xy_X_hat = np.delete(pose_keypoints_75_xy_X_hat, occluded_joint_idx_xy)
                        RMSE_xy_each = math.sqrt(mean_squared_error(pose_keypoints_75_xy_X_test, pose_keypoints_75_xy_X_hat))

                        pose_keypoints_75_xy_X_test_orig = np.vstack(([pose_keypoints_75_x_X_test_orig, pose_keypoints_75_y_X_test_orig]))  #shape(2,25)
                        pose_keypoints_75_xy_X_hat_orig = np.vstack(([pose_keypoints_75_x_X_hat_orig, pose_keypoints_75_y_X_hat_orig]))  #shape(2,25)
                        rmse_joints_each = np.empty((1,25))
                        for p in range(25):  # error for each joint
                            if p not in occluded_joint_idx_x or p not in occluded_joint_idx_y:
                                rmse_joints_each[0, p] = math.sqrt(mean_squared_error(pose_keypoints_75_xy_X_test_orig[:, p], pose_keypoints_75_xy_X_hat_orig[:, p]))
                            else:
                                rmse_joints_each[0, p] = np.nan

                        RMSE_x = np.concatenate((RMSE_x, np.array([RMSE_x_each])),axis=0)
                        RMSE_y = np.concatenate((RMSE_y, np.array([RMSE_y_each])), axis=0)
                        RMSE_xy = np.concatenate((RMSE_xy, np.array([RMSE_xy_each])), axis=0)
                        RMSE_joints = np.concatenate((RMSE_joints, rmse_joints_each), axis=0)

                        pose_keypoints_75_x_X_test_orig_all = np.concatenate((pose_keypoints_75_x_X_test_orig_all,pose_keypoints_75_x_X_test_orig.reshape(1,25)), axis=0)
                        pose_keypoints_75_y_X_test_orig_all = np.concatenate((pose_keypoints_75_y_X_test_orig_all,pose_keypoints_75_y_X_test_orig.reshape(1,25)), axis=0)
                        pose_keypoints_75_x_X_hat_orig_all = np.concatenate((pose_keypoints_75_x_X_hat_orig_all,pose_keypoints_75_x_X_hat_orig.reshape(1,25)), axis=0)
                        pose_keypoints_75_y_X_hat_orig_all = np.concatenate((pose_keypoints_75_y_X_hat_orig_all,pose_keypoints_75_y_X_hat_orig.reshape(1,25)), axis=0)
                        Seq_nt_framenames_all+=[openpose_filename_X_test]

                        RMSE_counter = RMSE_counter + 1
                        print('RMSE_counter=' + str(RMSE_counter))


                        ##### bounding box size
                        r_rois_width_tmp = r_rois_widths[r_distances_indices_X_intersect_idx]
                        r_rois_height_tmp = r_rois_heights[r_distances_indices_X_intersect_idx]
                        r_rois_area_each = r_rois_width_tmp * r_rois_height_tmp
                        r_rois_area = np.concatenate((r_rois_area, np.array([r_rois_area_each])), axis=0)
                        r_rois_location_each = maskrcnn_data['rois'][r_distances_indices_X_intersect_idx].reshape((1,4))
                        r_rois_locations = np.concatenate((r_rois_locations, r_rois_location_each), axis=0)

                        ###### save video_idx and frame_idx
                        video_idx_all = np.concatenate((video_idx_all, np.array([video_idx])), axis=0)
                        frame_idx_all = np.concatenate((frame_idx_all, np.array([frame_idx])), axis=0)

print('RMSE_x: {:.03f}({:.03f})'.format(np.mean(RMSE_x), np.std(RMSE_x)))
print('RMSE_y: {:.03f}({:.03f})'.format(np.mean(RMSE_y), np.std(RMSE_y)))
print('RMSE_xy: {:.03f}({:.03f})'.format(np.mean(RMSE_xy), np.std(RMSE_xy)))
for p in range(25):
    print('joint #' + str(p) + ': RMSE: {:.03f}({:.03f})'.format(np.nanmean(RMSE_joints[:,p]), np.nanstd(RMSE_joints[:,p])))


print("that's all folks!")


### Note: One can compute RMSE for each timestep by switching the two lines of code in Line #70 and 71 (the two for-loops).