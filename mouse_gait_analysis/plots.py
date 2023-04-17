import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from skimage import draw
from PIL import Image
from mouse_gait_analysis.utils import calculate_distance, rotate_points, create_rotation_affine
from mouse_gait_analysis.io import VideoReader, VideoWriter

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


def plot_steps(
        video, 
        keypoints, 
        start, 
        duration,
        stance_threshold=1.4, 
        swing_threshold=2.0,
        rolling_window=5,
        display_touch_positions=True):
    
    end = start + duration

    stance_threshold = 1.4
    swing_threshold = 2.0

    plt.figure(figsize=(12,12))
    frame = video[end]
    plt.imshow(frame)

    for bodypart in ['left_back_paw', 'right_back_paw']:
        bodypart_keypoints = keypoints.xs(bodypart, level='bodyparts', axis=1).droplevel(0, axis=1)
        bodypart_keypoints = bodypart_keypoints.rolling(rolling_window, center=True).mean()
        deltas = bodypart_keypoints.diff()
        distances = np.sqrt(deltas.x**2 + deltas.y**2)
        distances.loc[start:end]

        steps = distances.copy()
        steps[:] = np.nan
        step = 0
        stance_start = -1
        stance_end = -1
        is_bodypart_moving = True 
        for idx in tqdm(distances.index):
            d = distances.loc[idx]

            # Stance phase logic
            if is_bodypart_moving and d < stance_threshold:
                is_bodypart_moving = False
                stance_start = idx
            elif not is_bodypart_moving and d > stance_threshold:
                stance_end = idx

            # Don't start counting steps until a stance phase has been entered once
            if stance_start == -1:
                continue

            # Trigger based on movement threshold
            if not is_bodypart_moving and d > swing_threshold:
                is_bodypart_moving = True
                steps.loc[stance_start:stance_end] = step
                step += 1

        plt.plot(*bodypart_keypoints.loc[start:end, ['x','y']].values.T)
        for step in steps.loc[start:end].dropna().unique():
            step_points = bodypart_keypoints.loc[steps == step]
            # plt.scatter(step_points.x, step_points.y, s=20, color='red')

            center = bodypart_keypoints.loc[steps == step].mean()
            circle = plt.Circle((center.x, center.y), radius=10, facecolor=[0,0,0,0], linewidth=2, edgecolor=[1.,0,0,0.5])
            plt.gca().add_patch(circle)

            def plot_step(point):
                x = point.x
                y = point.y
                s = 3
                circle = plt.Polygon([[x-s,y-s], [x+s,y-s], [x,y+s]], color=[0,1.0,0])
                plt.gca().add_patch(circle)
                
            if display_touch_positions:
                plot_step(bodypart_keypoints.loc[steps == step].iloc[0])
                plot_step(bodypart_keypoints.loc[steps == step].iloc[-1])


def create_step_video(
        video, 
        keypoints,
        output_path, 
        start, 
        duration, 
        stance_threshold=1.4, 
        swing_threshold=2.0,
        rolling_window=5,
        display_touch_positions=True,
        output_fps=5,
        track_color=[0,0,255],
        step_color=[255,0,0],
        touch_color=[0,255,0]):
    
    end = start + duration
    bodyparts = None

    frame = np.zeros_like(video[0])
    w, h, c = frame.shape

    if bodyparts is None:
        bodyparts = ['left_back_paw', 'right_back_paw']

    for bodypart in bodyparts:
        bodypart_keypoints = keypoints.xs(bodypart, level='bodyparts', axis=1).droplevel(0, axis=1)
        bodypart_keypoints = bodypart_keypoints.rolling(rolling_window, center=True).mean()

        # Calculate distance
        deltas = bodypart_keypoints.diff()
        distances = np.sqrt(deltas.x**2 + deltas.y**2)
        distances.loc[start:end]

        # Extract steps based on distance
        steps = distances.copy()
        steps[:] = np.nan
        step = 0
        is_bodypart_moving = True
        stance_start = -1
        stance_end = -1
        for idx in tqdm(distances.index, desc=f"Finding stationary positions of {bodypart}"):
            d = distances.loc[idx]

            # Stance phase logic
            if is_bodypart_moving and d < stance_threshold:
                is_bodypart_moving = False
                stance_start = idx
            elif not is_bodypart_moving and d > stance_threshold:
                stance_end = idx

            # Don't start counting steps until a stance phase has been entered once
            if stance_start == -1:
                continue

            # Trigger based on movement threshold
            if not is_bodypart_moving and d > swing_threshold:
                is_bodypart_moving = True
                steps.loc[stance_start:stance_end] = step
                step += 1

        # Draw tracks
        points = bodypart_keypoints.loc[start:end, ["x", "y"]].values.astype(int)
        x, y = points.T
        for i in range(len(x)-1):
            rr, cc = draw.line(y[i], x[i], y[i+1], x[i+1])
            frame[rr, cc] = track_color

        # Draw average step position, start, and exit
        for step in steps.loc[start:end].dropna().unique():
            step_points = bodypart_keypoints.loc[steps == step]

            # Center position
            center = step_points.mean()
            orr, occ = draw.ellipse(center.y, center.x, r_radius=12, c_radius=12, shape=(h,w))
            irr, icc = draw.ellipse(center.y, center.x, r_radius=10, c_radius=10, shape=(h,w))
            mask = np.zeros(shape=(h, w, 3), dtype=float)
            mask[orr, occ] = 1
            mask[irr, icc] = 0
            rr, cc = np.argwhere((mask > 0).any(axis=-1)).T 
            frame[rr, cc] = step_color

            # Plot strike/leave positions
            if display_touch_positions:
                def draw_foot_position(row):
                    x = row.x
                    y = row.y
                    s = 3
                    points = np.array([[x-s,y-s], [x+s, y-s], [x, y+s]]).astype(int)
                    x, y = points.T
                    rr, cc = draw.polygon(y, x, shape=(h, w))
                    frame[rr, cc] = touch_color
                    
                draw_foot_position(step_points.iloc[0])
                draw_foot_position(step_points.iloc[-1])

    background = frame 

    # Overlay frames
    with VideoWriter(video=output_path, fps=output_fps, width=w, height=h) as writer:
        for i in tqdm(range(start, end+1), desc="Writing video"):
            frame = video[i]

            rr, cc = np.argwhere((background > 0).any(axis=-1)).T 
            frame[rr, cc] = background[rr, cc]
            writer.write(frame)