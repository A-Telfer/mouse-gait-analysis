import numpy as np
import pandas as pd
import cv2

from typing import Union

class PerspectiveTransformer:
    """
    Usage
    ---
    ```
    transformer = PerspectiveTransformer()
    transformer.register(points, target)
    transformed_points = transformer.apply()
    ```
    """
    def register(self, points, target):
        self.M = get_homography(points, target)
        xmax, ymax = target.max(axis=0)
        self.output_shape = (xmax, ymax)

    def apply(self, x: Union[np.ndarray, pd.DataFrame]):
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                return cv2.warpPerspective(x, self.M, self.output_shape)
            elif len(x.shape) == 2 and x.shape[-1] == 2:
                return transform_array_to_perspective(x, self.M)
            else: 
                raise Exception(f"Unrecognized shape {x.shape}. Must be image or x,y column vectors")
        elif isinstance(x, pd.DataFrame):
            return transform_dataframe_to_perspective(x, self.M)
        else:
            raise Exception(f"Unrecognized type. Must be ndarray or dataframe, found {type(x)}")

def transform_array_to_perspective(arr, T):
    """Move into the box's frame of reference"""
    x, y = arr.T
    tx, ty, v = T @ np.c_[x, y, np.ones_like(x)].T
    return np.c_[tx / v, ty / v]

def transform_dataframe_to_perspective(df, T):
    """Transform the coordinate dataframes to be in the box's frame of reference""" 
    na_values = df.isna()
    df = df.copy().fillna(0)
    idx = pd.IndexSlice
     
    x = df.loc[:, idx[:, :, "x"]]
    y = df.loc[:, idx[:, :, "y"]]
    x = x.stack(dropna=False).stack(dropna=False)
    y = y.stack(dropna=False).stack(dropna=False)

    tx, ty, v = T @ np.c_[x, y, np.ones_like(x)].T
    tx = tx / v
    ty = ty / v

    tx = pd.DataFrame(tx, index=x.index, columns=x.columns).unstack().unstack()
    ty = pd.DataFrame(ty, index=y.index, columns=y.columns).unstack().unstack()

    # Update multi index columns to match
    df.loc[:, pd.IndexSlice[:, :, "x"]] = tx
    df.loc[:, pd.IndexSlice[:, :, "y"]] = ty
    df[na_values] = np.NaN
    return df

def calculate_distance(data: Union[np.ndarray, pd.DataFrame], smoothing_window=5):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        # data = data.rolling(smoothing_window, center=True).mean()
        delta = data.diff()
        distance = np.linalg.norm(delta.values, axis=1)
    elif isinstance(data, pd.DataFrame):
        # data = data.rolling(smoothing_window, center=True).mean()
        delta = data.diff()
        distance = np.sqrt(delta.x**2 + delta.y**2)
    else:
        raise Exception("Type not recognized")
    
    distance = pd.Series(distance).rolling(smoothing_window, center=True).mean()
    return distance
    

def create_rotation_affine(rad):
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0,0,1]])

def rotate_points(points, affine):
    points = np.c_[points, np.ones_like(points[:,0])]
    points = affine @ points.T
    return points.T[:,:2]

def get_homography(points, target):
    points = points.astype(np.float32)
    target = target.astype(np.float32)
    T, res = cv2.findHomography(
        points, target, cv2.RANSAC, ransacReprojThreshold=32)
    return T

def filter_likelihood(keypoints, pcutoff):
    likelihood = keypoints.xs('likelihood', level='coords', axis=1)
    return keypoints.where(likelihood > pcutoff).interpolate()

def filter_distance_traveled(keypoints, threshold, iterations=1):
    for i in range(iterations):
        x = keypoints.xs('x', level='coords', axis=1).diff()
        y = keypoints.xs('y', level='coords', axis=1).diff()
        distance = np.sqrt(x**2 + y**2)
        keypoints = keypoints.where(distance < threshold).interpolate()

    return keypoints

class VideoAnalysis:
    def __init__(self, video, dlc_file):
        self.video = video
        self.dlc_file = dlc_file

        # Load keypoints
        self.keypoints = pd.read_hdf(self.dlc_file)   

    @property
    def bodyparts(self):
        return self.keypoints.columns.get_level_values('bodyparts').unique()
    
    @property
    def cap(self):
        return cv2.VideoCapture(str(self.video))
    
