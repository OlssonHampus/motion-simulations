# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:23:43 2023

@author: nolsson
"""

import numpy as np
import os
import torchio as tio
import shutil

"""Read data, define directories and create dataset"""
dataset_dir = os.path.join(r'\\mdc-berlin.net', 'fs', 'RD_MRT_Nako', 'MR-ART')
base_output_dir = os.path.join(r'\\mdc-berlin.net', 'fs', 'RD_MRT_Nako',
                               'MR-ART_TorchIO')
content = os.listdir(dataset_dir)
del content[0:6]
nsubs = len(content)
# nsubs = 1  # For testing
subjects = []
for sub_ix in range(nsubs):
    sub_name = content[sub_ix]
    nii_name = sub_name + '_acq-standard_T1w.nii.gz'
    nii_dir = os.path.join(dataset_dir, sub_name, 'anat', nii_name)
    subject = tio.Subject(mprage=tio.ScalarImage(nii_dir))
    subjects.append(subject)
original_dataset = tio.SubjectsDataset(subjects)

"""Define motion paradigm"""
pitch: int() = 15  # Pitch of nod [°]
# t_dur: int() = 5  # Duration of nod [s] (original data with wrong rotation)
t_dur: int() = 2.5  # Duration of nod [s]
# t_dur: int() = 100  # Duration of nod [s] Only for descriptive figure
num_nods_list: int() = np.array([5, 10])  # Number of nods during acquisition
# num_nods_list: int() = np.array([10])  # Only for descriptive figure
t_acq: int() = 316  # Acquisition duration of standard MPRAGE [s]
# t_acq: int() = 228  # Active acquisition duration of standard MPRAGE [s]

"""Parameters that follow from the user set parameters above"""
t_dur_norm: float() = t_dur / t_acq
t_int_arr: float() = [t_acq / x for x in num_nods_list]  # T between nods [s]
t_int_norm_arr: float() = [x / t_acq for x in t_int_arr]  # Norm to t_acq

"""Run modified TorchIO and save output as NIfTIs"""
print('TorchIO version:', tio.__version__)
# Unsafe workaround to multiple libiomp5md.dll files
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

for sub_ix in range(nsubs):
    progress_track = 'Subject: ' + str(sub_ix+1) + '/' + str(nsubs)
    print(progress_track)
    for num_nods_ix in range(len(num_nods_list)):
        num_nods = int(num_nods_list[num_nods_ix])
        num_trans: int() = num_nods*4  # Each nod = 4 discrete transforms
        t_int = t_int_norm_arr[num_nods_ix]
        t_start = np.linspace(0, 1-t_int, num=num_nods)
        t_start = np.reshape(t_start, [num_nods, 1])  # To fit times_arr
        # Timings for each rigid transform
        times_arr = np.zeros([num_trans, 1])
        times_arr[::4] = t_start  # Start -> Halfway up
        times_arr[1::4] = t_start+t_dur_norm/3  # Halfway up -> Up
        times_arr[2::4] = t_start+2*t_dur_norm/3  # Up -> Halfway down
        times_arr[3::4] = t_start+t_dur_norm  # Halfway down -> Down
        # Check if timings are monotonically increasing
        monot_incr_arr = np.all(times_arr[1:, :] > times_arr[:-1, :], axis=1)
        monot_incr_any = all(monot_incr_arr)
        if monot_incr_any is False:
            print('Warning: Timings in motion paradigm are nonsensical!')
        # Define rotation matrix to match times_arr (only pitch)
        # degrees_sub_array1 = np.array([0, pitch/2, 0])  # Hlfway up/Hlfway down
        # degrees_sub_array2 = np.array([0, pitch, 0])  # All the way up
        # 25.07.2023 All rotations were along the wrong axis
        degrees_sub_array1 = np.array([pitch/2, 0, 0])  # Hlfway up/Hlfway down
        degrees_sub_array2 = np.array([pitch, 0, 0])  # All the way up
        degrees_sub_array3 = np.array([0, 0, 0])  # All the way down
        degrees_arr = np.zeros([num_trans, 3])
        degrees_arr[::4, :] = degrees_sub_array1  # Halfway up
        degrees_arr[1::4, :] = degrees_sub_array2  # All the way up
        degrees_arr[2::4, :] = degrees_sub_array1  # Halfway down
        degrees_arr[3::4, :] = degrees_sub_array3  # All the way down
        # Define translation matrix to match times_arr (empty)
        translation_sub_array = np.array([0, 0, 0])
        translation_arr = np.tile(translation_sub_array, (num_trans, 1))
        # Tuple conversion of numpy arrays needed for TorchIO
        times_tuple = tuple(times_arr)
        degrees_tuple = tuple(degrees_arr)
        translation_tuple = tuple(translation_arr)
        #  Create TorchIO motion augmentation transform from the 3 tuples
        mot_aug = tio.my_NewMotion(degrees=degrees_tuple,
                                   translation=translation_tuple,
                                   times=times_tuple,
                                   image_interpolation='linear')
        # Create default random TorchIO motion augmentation as comparison
        # rand_mot_trans = tio.RandomMotion(degrees=(0, 15),
        #                                   translation=(0, 0),
        #                                   num_transforms=num_nods,
        #                                   image_interpolation='linear')
        augmented_dataset = tio.SubjectsDataset(subjects,
                                                transform=mot_aug)
        augmented_subject = augmented_dataset[sub_ix]  # Computations done here
        """Save augmented data and original as NIfTI files"""
        sub_name = content[sub_ix]
        sub_out_dir = os.path.join(base_output_dir, sub_name)
        sub_out_img_type_dir = os.path.join(sub_out_dir, 'anat')
        isExist = os.path.exists(sub_out_dir)
        if not isExist:
            os.makedirs(sub_out_dir)
            os.makedirs(sub_out_img_type_dir)
        acq_name = ('acq-pitch' + str(pitch) + 'dur' + str(t_dur) +
                    'nnods' + str(num_nods)) # Make sure filename doesn't have dots!!
        # Change the name for the random augmentation here
        # acq_name = ('acq-rot0to15nnods' + str(num_nods))
        nii_name = sub_name + '_' + acq_name + '_T1w.nii.gz'
        nii_out_dir = os.path.join(sub_out_img_type_dir, nii_name)
        augmented_subject.mprage.save(nii_out_dir)
        """Copy over accompanying json sidecars"""
        json_new_name = sub_name + '_' + acq_name + '_T1w.json'
        json_out_dir = os.path.join(sub_out_img_type_dir, json_new_name)
        input_dir = os.path.join(dataset_dir, sub_name, 'anat')
        if num_nods_ix == 0:
            json_in_dir = os.path.join(input_dir, sub_name +
                                       '_acq-headmotion1_T1w.json')
        if num_nods_ix == 1:
            json_in_dir = os.path.join(input_dir, sub_name +
                                       '_acq-headmotion2_T1w.json')
        isExist = os.path.exists(json_in_dir)
        if isExist:
            shutil.copyfile(json_in_dir, json_out_dir)
        # Save orignal image once per subject
        if num_nods_ix == 0:
            original_nii_name = sub_name + '_acq-standard_T1w.nii.gz'
            original_nii_out_dir = os.path.join(sub_out_img_type_dir,
                                                original_nii_name)
            original_subject = original_dataset[sub_ix]
            original_subject.mprage.save(original_nii_out_dir)
            nii_input_dir = os.path.join(dataset_dir, sub_name, 'anat',
                                         original_nii_name)
            json_standard_in_dir = os.path.join(input_dir, sub_name
                                                + '_acq-standard_T1w.json')
            json_standard_out_dir = os.path.join(
                sub_out_img_type_dir, sub_name + '_acq-standard_T1w.json')
            shutil.copyfile(json_standard_in_dir, json_standard_out_dir)
