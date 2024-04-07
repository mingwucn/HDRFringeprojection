from os import cpu_count
from lights import *
from tqdm import tqdm

calibration = False
gamma_calibration = False
exp_val = -2

PS = False
HDR = True

generate=True
steps = 2

capture = False
decode = False

exposure_times = [-1,-2,-3,-4,-5,-6]

if __name__ == "__main__":
    C = Capture(
        wait_time=1,
        cam_show=False,
        cam_id=1,
        exp_val=exp_val,
        fps=20,
        capture_width=4024,
        capture_height=3036,
        im_width=1280,
        im_height=800,
        )

    # Calibration
    if calibration==True:
        ## Calibrate camera
        calibrate_cam(
            capture=False,
            exp_val = exp_val,fps=30,
            cam_id = 1,
            capture_width=4024,
            capture_height=3036,
            hori_corner = 12,
            vertical_corner = 11,
            chess_block_size = 4,
            show_ratio = 0.2
        )
    
    if gamma_calibration == True:
        ## calibrate the projector gamma and generate calibrated camera pattern
        C.calibrate_gamma(test=False)
        C.decode_gamma(test=False)

    # Phase shifting
    if PS == True:
        C.PS(generate=generate,capture=capture,decode=decode,steps=steps)

    # HDR
    if HDR == True:
        C.HDR_capture(capture=True,exposure_times=exposure_times)
        # post HDR
        HDR_processing()
        C.decode_HDR()

