import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from mouse_gait_analysis.utils import calculate_distance, rotate_points, create_rotation_affine

def visualize_distances(
        keypoints, 
        bodyparts,
        start,
        duration,
        figsize=(5,2)):

    plt.figure(figsize=figsize)
    end = start + duration
    for bodypart in bodyparts:
        bodypart_keypoints = keypoints.xs(
            bodypart, level='bodyparts', axis=1).droplevel('scorer', axis=1)
        distance = calculate_distance(bodypart_keypoints)
        distance = distance[start:end]
        plt.plot(distance, label=bodypart)

    plt.legend()
    plt.xlabel("[frame]")
    plt.ylabel("[mm]")


def visualize_step_over_time(video,
        keypoints,
        bodyparts,
        start, 
        duration, 
        padding=50, 
        width_height_ratio=2.0,
        subplot_size=2,
        marker_size=20,
        threshold=2):
    
    end = start + duration
    cmap = plt.cm.get_cmap('rainbow')
    cmap_linspace = cmap(np.linspace(0, 1, duration))
    
    # Set padding
    if isinstance(padding, list) or isinstance(padding, tuple):
        padx, pady = padding
    else:
        padx = pady = padding

    fig_height = subplot_size
    fig_width = fig_height * width_height_ratio
    plt.figure(figsize=(fig_width, fig_height*duration))
    gs = plt.GridSpec(duration, 1)
    gs.update(wspace=0.025, hspace=0.05)

    # Find camera angle
    start_position = keypoints.xs('tailbase', level='bodyparts', axis=1).droplevel('scorer', axis=1).loc[start]
    end_position = keypoints.xs('nose', level='bodyparts', axis=1).droplevel('scorer', axis=1).loc[end]
    delta = end_position - start_position
    rad = np.arctan2(delta.y, delta.x)

    # Get the cropping area in the original image
    cropx = [start_position.x - padx, start_position.x + padx, end_position.x - padx, end_position.x + padx]
    cropy = [start_position.y - pady, start_position.y + pady, end_position.y - pady, end_position.y + pady]

    # Find the offset caused by rotating the frame
    frame = video[0]
    w, h, c = frame.shape
    corners = np.array([(0,0), (w,0), (w,h), (0,h)])
    affine = create_rotation_affine(-rad)
    tcorners = rotate_points(corners, affine)
    rotation_offset = tcorners.min(axis=0)

    # Find the offset caused by 
    crop_points = np.c_[cropx, cropy]
    crop_points = rotate_points(crop_points, affine)
    crop_points = crop_points - rotation_offset

    # Rotate and crop frames
    left, top = crop_points.min(axis=0)
    right, bottom = crop_points.max(axis=0)

    # Make same ratio
    width = right - left
    height = bottom - top
    if width > height:
        height = width / width_height_ratio
        center_y = (top + bottom) / 2
        top = center_y - height / 2
        bottom = center_y + height / 2
    else:
        width = height * 2
        center_x = (left + right) / 2
        top = center_x - width / 2
        bottom = center_x + width / 2

    crop_offset = np.array([left, top])
    for i in range(duration):
        plt.subplot(gs[i])
        plt.ylabel(f'Frame {start+i:05}\n[mm]')
        frame = video[start + i]
        image = Image.fromarray(frame)
        image = image.rotate(np.rad2deg(rad), resample=Image.Resampling.BICUBIC, expand=1)
        plt.gca().get_xaxis().set_visible(False)
        plt.imshow(image.crop(box=(left, top, right, bottom)))

        # Plot the movement of previous parts
        for bodypart in bodyparts:
            bodypart_keypoints = keypoints.xs(bodypart, level='bodyparts', axis=1).droplevel('scorer', axis=1)
            bodypart_keypoints = bodypart_keypoints.loc[start:start+i]
            points = bodypart_keypoints[['x', 'y']].values.copy()
            points = rotate_points(points, affine)
            points = points - rotation_offset - crop_offset
            x, y = points.T
            for j in range(i+1):
                plt.scatter(x[j], y[j], s=marker_size, c=cmap_linspace[duration-(i-j)-1].reshape(-1,4))

    plt.gca().get_xaxis().set_visible(True)
    plt.xlabel('[mm]')


def visualize_step_by_distance(
        video,
        keypoints,
        bodyparts,
        start, 
        duration, 
        padding=50, 
        width_height_ratio=2.0,
        figsize=(5,5),
        marker_size=20,
        threshold=2):
    
    end = start + duration
    frame = video[end]
    
    # Set padding
    if isinstance(padding, list) or isinstance(padding, tuple):
        padx, pady = padding
    else:
        padx = pady = padding

    plt.figure(figsize=figsize)
    gs = plt.GridSpec(4, 1)
    gs.update(wspace=0.025, hspace=0.05)

    # Find camera angle
    start_position = keypoints.xs('tailbase', level='bodyparts', axis=1).droplevel('scorer', axis=1).loc[start]
    end_position = keypoints.xs('nose', level='bodyparts', axis=1).droplevel('scorer', axis=1).loc[end]
    delta = end_position - start_position
    rad = np.arctan2(delta.y, delta.x)

    # Get the cropping area in the original image
    cropx = [start_position.x - padx, start_position.x + padx, end_position.x - padx, end_position.x + padx]
    cropy = [start_position.y - pady, start_position.y + pady, end_position.y - pady, end_position.y + pady]

    # Find the offset caused by rotating the frame
    w, h, c = frame.shape
    corners = np.array([(0,0), (w,0), (w,h), (0,h)])
    affine = create_rotation_affine(-rad)
    tcorners = rotate_points(corners, affine)
    rotation_offset = tcorners.min(axis=0)

    # Find the offset caused by 
    crop_points = np.c_[cropx, cropy]
    crop_points = rotate_points(crop_points, affine)
    crop_points = crop_points - rotation_offset
    crop_offset = crop_points.min(axis=0) 

    # Rotate and crop frames
    image = Image.fromarray(frame)
    image = image.rotate(np.rad2deg(rad), resample=Image.Resampling.BICUBIC, expand=1)
    left, top = crop_points.min(axis=0)
    right, bottom = crop_points.max(axis=0)

    # Make same ratio
    width = right - left
    height = bottom - top
    if width > height:
        height = width / width_height_ratio
        center_y = (top + bottom) / 2
        top = center_y - height / 2
        bottom = center_y + height / 2
    else:
        width = height * 2
        center_x = (left + right) / 2
        top = center_x - width / 2
        bottom = center_x + width / 2

    crop_offset = np.array([left, top])
    ax = plt.subplot(gs[:-2])
    plt.gca().get_xaxis().set_visible(False)
    plt.imshow(image.crop(box=(left, top, right, bottom)))

    # Plot bodypart keypoints
    for bodypart in bodyparts:
        bodypart_keypoints = keypoints.xs(bodypart, level='bodyparts', axis=1).droplevel('scorer', axis=1)
        distance = calculate_distance(bodypart_keypoints).loc[start:end].values

        bodypart_keypoints = bodypart_keypoints.loc[start:end]
        points = bodypart_keypoints[['x', 'y']].values.copy()
        points = rotate_points(points, affine)
        points = points - rotation_offset - crop_offset
        x, y = points.T

        plt.subplot(gs[:-2])
        plt.scatter(x, y, c=np.arange(len(x)), s=marker_size)
        plt.plot(x, y, label=bodypart)

        plt.subplot(gs[-2], sharex=ax)
        plt.plot(x, distance, label=bodypart)

        plt.subplot(gs[-1], sharex=ax)
        plt.plot(x, distance > threshold, label=bodypart)  

    # Plots
    plt.subplot(gs[:-2])
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel("[mm]")
    plt.legend()

    plt.subplot(gs[-2])
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel("[mm/frame]")

    plt.subplot(gs[-1])
    plt.yticks([0, 1], ['stance', 'swing'])
    plt.ylim([-0.2, 1.2])
    plt.xlabel("[mm]")

