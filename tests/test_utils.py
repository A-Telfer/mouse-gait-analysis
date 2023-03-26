from mouse_gait_analysis import utils
from pathlib import Path

class TestAnalysis:
    def test_bodyparts(self):
        test_data = Path(__file__).parent / 'test_data'
        video_analysis = utils.VideoAnalysis(
            test_data / 'test_video.mp4', 
            test_data / 'test_keypoints.h5')

        assert set(video_analysis.bodyparts) == {
            'nose', 'chest', 'tailbase', 'left_front_paw', 'right_front_paw', 
            'left_toes', 'right_toes', 'left_back_paw', 'right_back_paw'}
