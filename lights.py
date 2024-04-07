from genericpath import exists
import os,warnings,cv2,imageio,numpy,time
warnings.filterwarnings("ignore")
import structuredlight as sl
import numpy as np
from tqdm import tqdm
import sys
if sys.version_info[0] == 2:  # the tkinter library changed it's name from Python 2 to 3.
    import Tkinter
    tkinter = Tkinter #I decided to use a library reference to avoid potential naming conflicts with people's programs.
else:
    import tkinter
from PIL import Image, ImageTk


import sys,os,re,argparse,cv2,glob,os.path
import plotly.offline as po
import plotly.graph_objs as go
import numpy as np
from scipy.optimize import fmin, brent

# pip install git+https://github.com/elerac/structuredlight

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

class Pattern():
    """"
    Binary
    Gray
    XOR
    Ramp
    PhaseShifting
    Stripe
    """
    def __init__(self,width  = 848,height = 480,fps=1) -> None:
        self.width = width
        self.height =height
        self.fps=fps

    def Grey(self):
        gray = sl.Gray()
        imlist_pattern = gray.generate((self.width, self.height))
        self.imlist_pattern = imlist_pattern
        self.save_gif()
        return imlist_pattern
    
    def Binary(self):
        binary = sl.Binary()
        imlist = binary.generate((self.width, self.height))
        img_index = binary.decode(imlist, thresh=127)
        self.imlist_pattern = imlist
        self.save_gif()
        return imlist
    
    def Xor(self):
        xor = sl.XOR(index_last=-1)
        imlist = xor.generate((self.width, self.height))
        img_index = xor.decode(imlist, thresh=127)
        self.imlist_pattern = imlist
        self.save_gif()
        return imlist
    
    def Ramp(self):
        ramp = sl.Ramp()
        imlist = ramp.generate((self.width, self.height))
        img_index = ramp.decode(imlist)
        self.imlist_pattern = imlist
        self.save_gif()
        return imlist
    
    def PhaseShifting(self):
        phaseshifting = sl.PhaseShifting(num=3)
        imlist = phaseshifting.generate((self.width, self.height))
        img_index = phaseshifting.decode(imlist)
        self.imlist_pattern = imlist
        self.save_gif()
        return imlist
    
    def Stripe(self):
        stripe = sl.Stripe()
        imlist = stripe.generate((self.width, self.height))
        img_index = stripe.decode(imlist)
        self.imlist_pattern = imlist
        self.save_gif()
        return imlist
    
    def save_gif(self):
        imageio.mimsave(os.path.join("light_source","project.gif"),self.imlist_pattern,fps = self.fps)


class Capture():
    def __init__(
        self,
        wait_time=1,
        cam_show=False,
        cam_id=1,
        exp_val=-4,
        fps=20,
        capture_width=4024,
        capture_height=3036,
        im_width=1280,
        im_height=800):
        # initialization
        self.wait_time=wait_time
        self.cam_show=cam_show
        self.cam_id=cam_id
        self.exp_val=exp_val
        self.fps=fps
        self.capture_width=capture_width
        self.capture_height=capture_height
        self.im_width = im_width
        self.im_height = im_height

    def web_start(self):
        from selenium import webdriver
        from webdriver_manager.firefox import GeckoDriverManager
        import bs4
        self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        webpath = os.path.join(os.getcwd(),'calibrate.html')
        self.driver.get(webpath)
        # load the file
        with open("calibrate.html") as inf:
            txt = inf.read()
            self.soup = bs4.BeautifulSoup(txt)
        #     self.soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat00.png"
        # # save the file again
        # with open("calibrate.html", "w") as outf:
            # outf.write(str(self.soup))
        self.driver.refresh()
        return self.driver

    def cam_set(self):
        camera = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        codec = 0x47504A4D # MJPG
        # print("Setting camera mode")
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        camera.set(cv2.CAP_PROP_FPS, self.fps)
        camera.set(cv2.CAP_PROP_FOURCC, codec)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, self.exp_val)
        return camera

    def capture_loop(self,code_dir,capture_dir,capture_name = "pat", exposure_time=None):
        # create capture_dir folder
        os.makedirs(capture_dir,exist_ok=True)
        # get code imgs list
        code_list = [i for i in os.listdir(code_dir) if os.path.splitext(i)[1]==".png"]
        for img_counter,code_img in enumerate(tqdm(code_list)):
            src_path = os.path.join(code_dir,code_img)
            # replace img src
            relpath = os.path.relpath(src_path,os.getcwd())
            self.soup.find('img')['src'] = ".\\"+relpath
            # save the file again
            with open("calibrate.html", "w") as outf:
                outf.write(str(self.soup))     
            self.driver.refresh()
            if img_counter==0:
                camera = self.cam_set()
                del camera
            time.sleep(self.wait_time) 
            # capture and save
            img_name = capture_name+"{}.png".format(str(img_counter).zfill(2))
            img_name = os.path.join(capture_dir,img_name)
            if exposure_time is None:
                exposure_time = self.exp_val
            else:
                self.exp_val = exposure_time
            # if exposure_time in [1,0,-1]:
            #     time.sleep(10)
            # elif exposure_time in [-2,-3,-4]:
            #     time.sleep(5)
            # else:
            #     pass
            # print (self.exp_val)
            camera = self.cam_set()
            ret,frame = camera.read()
            if self.cam_show == True:
                cv2.imshow("frame",frame)
            self.driver.refresh()
            cv2.imwrite(img_name,frame)
            camera.release()
            del ret,frame,camera

    def calibrate_gamma(self,test=False,steps=2):
        """"
        return: "./calibrate/gamma/uncorrection_captures/gamma.txt" with gamma value
        Test = True: generate phase shifting pattern and capture

        """
        # start the web server
        self.web_start()
        if input("Do You Want To GO ? [y/n]") == "y":
            pass
        else:
            return
        # get the code dir
        code_dir = os.path.join(os.getcwd(),"calibrate","gamma","uncorrection_patterns")
        capture_dir = os.path.join(os.getcwd(),"calibrate","gamma","uncorrection_captures")
        os.makedirs(code_dir,exist_ok=True)
        # generate code img
        GC = Gamma_Correction()
        print("Generating gamma un-correct pattren...")
        GC.generate(
            width=self.im_width,
            height=self.im_height,
            code_dir=code_dir)
        # start the capture loop, assert code_dir, capture_dir
        print("Capturing the un-corrections...")
        self.capture_loop(code_dir,capture_dir)
        GC.decode(code_dir=code_dir,capture_dir=capture_dir)
        # generate os.path.join(capture_dir,'gamma.txt') with gamma value

        if test==True:
            code_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns")
            capture_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_captures")
            PS = Phase_Shifting()
            print("Generating gamma corrected pattren...")
            PS.generate(
                code_dir,
                width=self.im_width,
                height=self.im_height,
                step=steps)
            # capture again
            print("Capturing the corrections")
            self.capture_loop(code_dir,capture_dir)
            # decode
            # PS.decode(capture_dir=capture_dir,code_dir=code_dir)

    def decode_gamma(self,test=False,steps=2):
        """
        Test = True: generate phase shifting pattern and capture
        """
        # start the web server
        self.web_start()
        if input("Do You Want To GO ? [y/n]") == "y":
            pass
        else:
            return
        # get the code dir
        code_dir = os.path.join(os.getcwd(),"calibrate","gamma","uncorrection_patterns")
        capture_dir = os.path.join(os.getcwd(),"calibrate","gamma","uncorrection_captures")
        os.makedirs(code_dir,exist_ok=True)
        # generate code img
        GC = Gamma_Correction()
        GC.decode(code_dir=code_dir,capture_dir=capture_dir)
        if test==True:
            code_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns")
            capture_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_captures")
            PS = Phase_Shifting()
            print("Generating gamma corrected pattren...")
            PS.generate(
                code_dir,
                width=self.im_width,
                height=self.im_height,
                step=steps)
            # capture again
            print("Capturing the corrections")
            self.capture_loop(code_dir,capture_dir)
            # decode
            # PS.decode(capture_dir=capture_dir,code_dir=code_dir)

    def PS(self,generate=True,capture=True,decode=True,steps=2,exposure_time=-4):
        code_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns")
        capture_dir = os.path.join(
            os.getcwd(),
            "Phase_shift",
            "captures",
            )
        PS = Phase_Shifting()

        if generate==True:
            PS.generate(
                code_dir,
                width=self.im_width,
                height=self.im_height,
                step=steps)

        if capture==True:
            # start the web server
            self.web_start()
            if input("Do You Want To GO ? [y/n]") == "y":
                pass
            else:
                return
            self.capture_loop(code_dir,capture_dir,exposure_time=exposure_time)

        if decode == True:
            # decode
            black_threshold = [20,10,5,0,0,0]
            white_threshold = [4,2,1,1,1,1]
            PS.decode(
                capture_dir=capture_dir,
                code_dir=code_dir,
                black_thr=black_threshold[0],
                white_thr=white_threshold[0])
                
    # TODO HDR
    def HDR_capture(self,exposure_times = [-1,-2,-3,-4,-5,-6],capture=True):
        """"
        CAP_PROP_EXPOSURE           Actual exposure time
        -1	                         500ms
        -2	                         250ms
        -3	                         125ms
        -4	                         62.5ms
        -5	                         31.3ms
        -6	                         15.6ms
        -7                           7.8ms
        """
        if capture==True:
        # start the web server
            self.web_start()
            if input("Do You Want To GO ? [y/n]") == "y":
                pass
            else:
                return
        for _,exposure_time in enumerate(exposure_times):
            print("Current exposure time:{}".format(exposure_time))
            # if _<3:
            #     continue
            # else:
            #     pass
            # print(_)
            code_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns")
            capture_dir = os.path.join(
                os.getcwd(),
                "HDR","Phase_shift",
                "captures",
                "exposure_{}".format(exposure_time)
                )
            PS = Phase_Shifting()
            # PS.generate(
            #     code_dir,
            #     width=self.im_width,
            #     height=self.im_height,
            #     STEP=2)
            # capture again
            if capture==True:
                self.capture_loop(code_dir,capture_dir,exposure_time=exposure_time)
            else:
                pass
            # decode
            black_threshold = [20,10,5,0,0,0]
            white_threshold = [4,2,1,1,1,1]
            PS.decode(
                capture_dir=capture_dir,
                code_dir=code_dir,
                black_thr=black_threshold[_],
                white_thr=white_threshold[_])

    def decode_HDR(self,file_type=".jpg"):
        code_dir = os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns")
        capture_dir = os.path.join(os.getcwd(),"HDR","Phase_shift","Generate")
        PS = Phase_Shifting()
        # decode
        PS.decode(capture_dir=capture_dir,code_dir=code_dir,file_type=file_type,black_thr=1,white_thr=1)

def HDR_processing():
    # load img
    img_num = len(glob.glob(os.path.join(os.getcwd(),"calibrate","phase_shifting","correction_patterns","*.png")))
    os.makedirs(os.path.join(os.getcwd(),"HDR","Phase_Shift","Generate"),exist_ok=True)
    for i in tqdm(range(img_num)):
        img_list = glob.glob(os.path.join(os.getcwd(),"HDR","Phase_Shift","captures","*\\pat{}.png".format(str(i).zfill(2))))
        images = list(map(cv2.imread,img_list))
        times = np.array([1/2,1/4,1/8,1/16,1/32,1/64],dtype=np.float32)
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(images, times)
        res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint16')
        cv2.imwrite(
            os.path.join(
                os.getcwd(),
                "HDR",
                "Phase_Shift",
                "Generate",
                "pat{}.jpg".format(str(i).zfill(2))),
                res_mertens_8bit)
    pass

class Phase_Shifting():
    def __init__(self):
        pass
        # self.width = WIDTH
        # self.height = HEIGHT
        # self.step = STEP
        # self.GC_STEP = int(STEP/2)

    def generate(self,code_dir,width,height,step,GAMMA=None):
        WIDTH = width
        HEIGHT = height
        STEP = step
        if GAMMA==None:
            with open(os.path.join(os.getcwd(),"calibrate",'gamma.txt')) as f:
                lines = f.readlines()
            GAMMA = float(lines[0])
        self.gamma = GAMMA
        GC_STEP = int(STEP/2)
        OUTPUTDIR = code_dir
        print (OUTPUTDIR)
        os.makedirs(OUTPUTDIR,exist_ok=True)
        # if not os.path.exists(OUTPUTDIR):
        #     os.mkdir(OUTPUTDIR)

        imgs = []

        print('Generating sinusoidal patterns ...')
        angle_vel = np.array((6, 4))*np.pi/STEP
        xs = np.array(range(WIDTH))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5*(np.cos(xs*angle_vel[i-1] + np.pi*(phs-2)*2/3)+1)
                vec = 255*(vec**GAMMA)
                vec = np.round(vec)
                img = np.zeros((HEIGHT, WIDTH), np.uint8)
                for y in range(HEIGHT):
                    img[y, :] = vec
                imgs.append(img)

        ys = np.array(range(HEIGHT))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5*(np.cos(ys*angle_vel[i-1] + np.pi*(phs-2)*2/3)+1)
                vec = 255*(vec**GAMMA)
                vec = np.round(vec)
                img = np.zeros((HEIGHT, WIDTH), np.uint8)
                for x in range(WIDTH):
                    img[:, x] = vec
                imgs.append(img)

        print('Generating graycode patterns ...')
        gc_height = int((HEIGHT-1)/GC_STEP)+1
        gc_width = int((WIDTH-1)/GC_STEP)+1

        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
        for pat in patterns:
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    img[y, x] = pat[int(y/GC_STEP), int(x/GC_STEP)]
            imgs.append(img)
        imgs.append(255*np.ones((HEIGHT, WIDTH), np.uint8))  # white
        imgs.append(np.zeros((HEIGHT, WIDTH), np.uint8))     # black

        for i, img in enumerate(imgs):
            cv2.imwrite(OUTPUTDIR+'/pat'+str(i).zfill(2)+'.png', img)

        print('Saving config file ...')
        fs = cv2.FileStorage(os.path.join(OUTPUTDIR,'config.xml'), cv2.FILE_STORAGE_WRITE)
        fs.write('disp_width', WIDTH)
        fs.write('disp_height', HEIGHT)
        fs.write('step', STEP)
        fs.release()

        print('Done')

    def decode(
        self,capture_dir,
        code_dir,black_thr=20,
        white_thr=4,filter_size=1,
        file_type = ".png",
        output_dir=None,*kargs):
        BLACKTHR = black_thr
        WHITETHR = white_thr
        INPUTPRE = capture_dir
        FILTER = filter_size
        configfile = os.path.join(code_dir,'config.xml')
        if output_dir ==None:
            output_dir = capture_dir
        OUTPUTDIR = output_dir
        fs = cv2.FileStorage(configfile, cv2.FILE_STORAGE_READ)
        DISP_WIDTH = int(fs.getNode('disp_width').real())
        DISP_HEIGHT = int(fs.getNode('disp_height').real())
        STEP = int(fs.getNode('step').real())
        GC_STEP = int(STEP/2)
        fs.release()

        gc_width = int((DISP_WIDTH-1)/GC_STEP)+1
        gc_height = int((DISP_HEIGHT-1)/GC_STEP)+1
        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        graycode.setBlackThreshold(BLACKTHR)
        graycode.setWhiteThreshold(WHITETHR)

        print('Loading images ...')
        re_num = re.compile(r'(\d+)')

        def numerical_sort(text):
            return int(re_num.split(text)[-2])

        filenames = sorted(
            glob.glob(os.path.join(INPUTPRE,'*{}'.format(file_type))), key=numerical_sort)
        if len(filenames) != graycode.getNumberOfPatternImages() + 14:
            print('Number of images is not right (right number is ' +
                str(graycode.getNumberOfPatternImages() + 14) + ')')
            print("Now is {}".format(len(filenames)))
            print("Path is {}".format(os.path.join(INPUTPRE,'*{}'.format(file_type))))
            # print(filenames)
            return

        imgs = []
        for f in filenames:
            imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        ps_imgs = imgs[0:12]
        gc_imgs = imgs[12:]
        black = gc_imgs.pop()
        white = gc_imgs.pop()
        CAM_WIDTH = white.shape[1]
        CAM_HEIGHT = white.shape[0]

        print('Decoding images ...')

        def decode_ps(pimgs):
            pimg1 = pimgs[0].astype(np.float32)
            pimg2 = pimgs[1].astype(np.float32)
            pimg3 = pimgs[2].astype(np.float32)
            return np.arctan2(
                np.sqrt(3)*(pimg1-pimg3), 2*pimg2-pimg1-pimg3)

        ps_map_x1 = decode_ps(ps_imgs[0:3])
        ps_map_x2 = decode_ps(ps_imgs[3:6])
        ps_map_y1 = decode_ps(ps_imgs[6:9])
        ps_map_y2 = decode_ps(ps_imgs[9:12])

        gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
        mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:
                    continue
                err, proj_pix = graycode.getProjPixel(gc_imgs, x, y)
                if not err:
                    gc_map[y, x, :] = np.array(proj_pix)
                    mask[y, x] = 255

        if FILTER != 0:
            print('Applying smoothing filter ...')
            ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                        np.ones((FILTER*2+1, FILTER*2+1)))
            for y in range(CAM_HEIGHT):
                for x in range(CAM_WIDTH):
                    if mask[y, x] == 0 and ext_mask[y, x] != 0:
                        sum_x = 0
                        sum_y = 0
                        cnt = 0
                        for dy in range(-FILTER, FILTER+1):
                            for dx in range(-FILTER, FILTER+1):
                                ty = y + dy
                                tx = x + dx
                                if ((dy != 0 or dx != 0)
                                        and ty >= 0 and ty < CAM_HEIGHT
                                        and tx >= 0 and tx < CAM_WIDTH
                                        and mask[ty, tx] != 0):
                                    sum_x += gc_map[ty, tx, 0]
                                    sum_y += gc_map[ty, tx, 1]
                                    cnt += 1
                        if cnt != 0:
                            gc_map[y, x, 0] = np.round(sum_x/cnt)
                            gc_map[y, x, 1] = np.round(sum_y/cnt)

            mask = ext_mask

        def decode_pixel(gc, ps1, ps2):
            dif = None
            if ps1 > ps2:
                if ps1-ps2 > np.pi*4/3:
                    dif = (ps2-ps1)+2*np.pi
                else:
                    dif = ps1-ps2
            else:
                if ps2-ps1 > np.pi*4/3:
                    dif = (ps1-ps2)+2*np.pi
                else:
                    dif = ps2-ps1

            p = None
            if gc % 2 == 0:
                p = ps1
                if dif > np.pi/6 and p < 0:
                    p = p + 2*np.pi
                if dif > np.pi/2 and p < np.pi*7/6:
                    p = p + 2*np.pi
            else:
                p = ps1
                if dif > np.pi*5/6 and p > 0:
                    p = p - 2*np.pi
                if dif < np.pi/2 and p < np.pi/6:
                    p = p + 2*np.pi
                p = p + np.pi
            return gc*GC_STEP + STEP*p/3/2/np.pi

        print('Decoding each pixels ...')
        viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
        res_list = []
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if mask[y, x] != 0:
                    est_x = decode_pixel(
                        gc_map[y, x, 0], ps_map_x1[y, x], ps_map_x2[y, x])
                    est_y = decode_pixel(
                        gc_map[y, x, 1], ps_map_y1[y, x], ps_map_y2[y, x])
                    viz[y, x, :] = (est_x, est_y, 128)
                    res_list.append((y, x, est_y, est_x))

        print('Exporting result ...')
        if not os.path.exists(OUTPUTDIR):
            os.mkdir(OUTPUTDIR)
        cv2.imwrite(OUTPUTDIR+'/vizualized.jpg', viz)
        with open(OUTPUTDIR+'/camera2display.csv', mode='w') as f:
            f.write('camera_y, camera_x, display_y, display_x\n')
            for (cam_y, cam_x, disp_y, disp_x) in res_list:
                f.write(str(cam_y) + ', ' + str(cam_x) +
                        ', ' + str(disp_y) + ', ' + str(disp_x) + '\n')

        print('Done')

class Gamma_Correction():

    def __init__(self):
        pass
        # self.width = WIDTH
        # self.height = HEIGHT
        # self.step = STEP
        # # self.gamma = GAMMA
        # self.GC_STEP = int(STEP/2)
        # self.output_dir =OUTPUTDIR
    def generate(self,width,height,code_dir,step=1,gamma_p1=0.75,gamma_p2=1.25,*kargs):
        WIDTH = width
        HEIGHT = height
        GAMMA_P1 = gamma_p1
        GAMMA_P2 = gamma_p2
        STEP = step
        PHSSTEP = int(WIDTH/8)
        OUTPUTDIR = code_dir

        if not os.path.exists(OUTPUTDIR):
            os.mkdir(OUTPUTDIR)

        imgs = []

        print('Generating sinusoidal patterns ...')
        angle_vel = 2*np.pi/PHSSTEP
        gamma = [1/GAMMA_P1, 1/GAMMA_P2]
        xs = np.array(range(WIDTH))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5*(np.cos(xs*angle_vel + np.pi*(phs-2)*2/3)+1)
                vec = 255*(vec**gamma[i-1])
                vec = np.round(vec)
                img = np.zeros((HEIGHT, WIDTH), np.uint8)
                for y in range(HEIGHT):
                    img[y, :] = vec
                imgs.append(img)

        ys = np.array(range(HEIGHT))
        for i in range(1, 3):
            for phs in range(1, 4):
                vec = 0.5*(np.cos(ys*angle_vel + np.pi*(phs-2)*2/3)+1)
                vec = 255*(vec**gamma[i-1])
                img = np.zeros((HEIGHT, WIDTH), np.uint8)
                for x in range(WIDTH):
                    img[:, x] = vec
                imgs.append(img)

        print('Generating graycode patterns ...')
        gc_height = int((HEIGHT-1)/STEP)+1
        gc_width = int((WIDTH-1)/STEP)+1

        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
        for pat in patterns:
            if STEP == 1:
                img = pat
            else:
                img = np.zeros((HEIGHT, WIDTH), np.uint8)
                for y in range(HEIGHT):
                    for x in range(WIDTH):
                        img[y, x] = pat[int(y/STEP), int(x/STEP)]
            imgs.append(img)
        imgs.append(255*np.ones((HEIGHT, WIDTH), np.uint8))  # white
        imgs.append(np.zeros((HEIGHT, WIDTH), np.uint8))     # black

        for i, img in enumerate(imgs):
            cv2.imwrite(OUTPUTDIR+'/pat'+str(i).zfill(2)+'.png', img)

        print('Saving config file ...')
        fs = cv2.FileStorage(os.path.join(OUTPUTDIR,'config.xml'), cv2.FILE_STORAGE_WRITE)
        fs.write('disp_width', WIDTH)
        fs.write('disp_height', HEIGHT)
        fs.write('gamma_p1', GAMMA_P1)
        fs.write('gamma_p2', GAMMA_P2)
        fs.write('step', STEP)
        fs.release()

        print('Done')

    def decode(self,code_dir,capture_dir,config_file=None,black_thr=30,white_thr=4):
        BLACKTHR = black_thr
        WHITETHR = white_thr
        INPUTPRE = capture_dir
        if config_file==None:
            config_file = os.path.join(code_dir,'/config.xml')
        fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
        DISP_WIDTH = int(fs.getNode('disp_width').real())
        DISP_HEIGHT = int(fs.getNode('disp_height').real())
        GAMMA_P1 = fs.getNode('gamma_p1').real()
        GAMMA_P2 = fs.getNode('gamma_p2').real()
        STEP = int(fs.getNode('step').real())
        PHSSTEP = int(DISP_WIDTH/8)
        fs.release()

        gc_width = int((DISP_WIDTH-1)/STEP)+1
        gc_height = int((DISP_HEIGHT-1)/STEP)+1
        graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
        graycode.setBlackThreshold(BLACKTHR)
        graycode.setWhiteThreshold(WHITETHR)

        print('Loading images ...')
        re_num = re.compile(r'(\d+)')

        def numerical_sort(text):
            return int(re_num.split(text)[-2])

        filenames = sorted(
            glob.glob(os.path.join(INPUTPRE,'*.png')), key=numerical_sort)
        if len(filenames) != graycode.getNumberOfPatternImages() + 14:
            print('Number of images is not right (right number is ' +
                str(graycode.getNumberOfPatternImages() + 14) + ')')
            print(len(filenames))
            return

        imgs = []
        for f in filenames:
            imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        ps_imgs = imgs[0:12]
        gc_imgs = imgs[12:]
        black = gc_imgs.pop()
        white = gc_imgs.pop()
        CAM_WIDTH = white.shape[1]
        CAM_HEIGHT = white.shape[0]

        print('Decoding graycode ...')
        gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
        viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
        mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
        target_map_x = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float32)
        target_map_y = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float32)
        angle_vel = 2*np.pi/PHSSTEP
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:
                    continue
                err, proj_pix = graycode.getProjPixel(gc_imgs, x, y)
                if not err:
                    pos = STEP*np.array(proj_pix)
                    gc_map[y, x, :] = pos
                    target_map_x[y, x] = angle_vel*pos[0]
                    target_map_y[y, x] = angle_vel*pos[1]
                    viz[y, x, 0] = pos[0]
                    viz[y, x, 1] = pos[1]
                    viz[y, x, 2] = 128
                    mask[y, x] = 1

        # cv2.imwrite('viz.png', viz)

        def decode_ps(pimgs, gamma=1.0):
            pimg1 = (pimgs[0].astype(np.float32)/255)**gamma
            pimg2 = (pimgs[1].astype(np.float32)/255)**gamma
            pimg3 = (pimgs[2].astype(np.float32)/255)**gamma
            return np.arctan2(
                np.sqrt(3)*(pimg1-pimg3), 2*pimg2-pimg1-pimg3)

        def res_func(xs, tx, ty, imgsx, imgsy, mask):
            dx = decode_ps(imgsx, xs)*mask
            dy = decode_ps(imgsy, xs)*mask
            dif = (dx-tx+np.pi) % (2*np.pi) - np.pi
            dif += (dy-ty+np.pi) % (2*np.pi) - np.pi
            res = np.sum(dif**2)
            return res

        print('Estimating gamma1-dash ...')
        gamma1d = brent(res_func, brack=(0, 3), args=(
            target_map_x, target_map_y, ps_imgs[0:3], ps_imgs[6:9], mask))
        print(' ', gamma1d)

        print('Estimating gamma2-dash ...')
        gamma2d = brent(res_func, brack=(0, 3), args=(
            target_map_x, target_map_y, ps_imgs[3:6], ps_imgs[9:12], mask))
        print(' ', gamma2d)

        gamma_a = (GAMMA_P1 - GAMMA_P2)/(gamma1d - gamma2d)
        gamma_b = (GAMMA_P1*gamma2d - gamma1d*GAMMA_P2)/(GAMMA_P1 - GAMMA_P2)
        gamma_p = (1 - gamma_b)*gamma_a
        print('  gamma a :', gamma_a)
        print('  gamma b :', gamma_b)

        print('Result')
        print('  gamma p :', gamma_p)
        with open(os.path.join(os.getcwd(),"calibrate",'gamma.txt'), 'w') as f:
            f.write(str(gamma_p))
        print('Done')

def calculate_SD(img_path):
    img = cv2.imread(img_path)

def calibrate_cam(
    capture=True,
    exp_val = -4,fps=30,
    cam_id = 1,
    capture_width=4024,
    capture_height=3036,
    hori_corner = 12,
    vertical_corner = 11,
    chess_block_size = 4,
    show_ratio = 0.4
    ):

    if capture==True:
        # ajust exposure time
        print("Initializing camera")
        camera = cv2.VideoCapture(cam_id)
        print("Setting camera mode")
        codec = 0x47504A4D # MJPG
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FOURCC, codec)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, exp_val)

        print("Start capture: \n Press 'c' to capture,\n Press 'q' to quit")
        # for i in range(1,60*fps):
        captured_num = 0

        show_width = int(capture_width*show_ratio)
        show_height = int(capture_height*show_ratio)
        while (True):
            # camera.grab()
            # retval, im = camera.retrieve(0)
            ret,im = camera.read()
            im_show = cv2.resize(im,(show_width,show_height))
            cv2.imshow("image", im_show)

            k = cv2.waitKey(1) & 0xff
            if  k == ord('q'):
                print("exit")
                break
            # if i%(fps*5) == 0:
            if  k == ord('c'):
                captured_num += 1
                cv2.imwrite('./cam_calibration/'+str(captured_num)+'.png',im)
                print("Move the chessboard: %d"% captured_num )
                # print("Move the chessboard: %d/%d"%(i//(fps*5)))
        camera.release()
        cv2.destroyAllWindows()
    else:
        pass
    
    print("Start calibration")
    import glob
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((vertical_corner*hori_corner,3), np.float32)
    objp[:,:2] = chess_block_size*np.mgrid[0:hori_corner,0:vertical_corner].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./cam_calibration/*.png')
    # images = img_list
    for fname in tqdm(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners

        ret, corners = cv2.findChessboardCorners(gray, (hori_corner,vertical_corner), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (hori_corner,vertical_corner), corners2, ret)
            cv2.imwrite(os.path.join(".\\cam_calibrated\\chessboard"+"_".join(fname.split("\\")[-2:])),img)
            # cv2.imshow('img', img)
            cv2.waitKey(500)
        elif not ret:
            print("No Cheesboard was found in: {}".format(fname))
    # cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # img = cv2.imread('left12.jpg')
    # h,  w = img.shape[:2]
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    np.save("./camera_params/ret", ret)
    np.save("./camera_params/K", mtx)
    np.save("./camera_params/dist", dist)
    np.save("./camera_params/rvecs", rvecs)
    np.save("./camera_params/tvecs", tvecs)

    print ("ret:",ret)
    print ("mtx:\n",mtx)        # 内参数矩阵
    print ("dist:\n",dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print ("rvecs:\n",rvecs)    # 旋转向量  # 外参数
    print ("tvecs:\n",tvecs)   # 平移向量  # 外参数

def pro_cam_calibrate(wait_time=1,cam_show=False,cam_id=1,exp_val=-4,fps=30,capture_width=4024,capture_height=3036):
    import os,cv2,bs4
    current_dir = os.getcwd()
    greycode_dir = os.path.join(current_dir,"test","Projector-Camera-Calibration","graycode_pattern")
    capture_dir = os.path.join(current_dir,"test","Projector-Camera-Calibration","capture")
    greycode_list = [i for i in os.listdir(greycode_dir) if os.path.splitext(i)[1]==".png"]
    # load the file
    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\Projector-Camera-Calibration\\graycode_pattern\\pattern_09.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))

    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\Projector-Camera-Calibration\\graycode_pattern\\pattern_40.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))
    capture_num = 1

    from selenium import webdriver
    from webdriver_manager.firefox import GeckoDriverManager

    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    webpath = os.path.join(os.getcwd(),'calibrate.html')
    driver.get(webpath)
      
    # print("Initializing camera")
    # camera = VideoCapture(cam_id=cam_id,exp_val=exp_val,fps=fps,capture_width=capture_width,capture_height=capture_height)
    def capture():
        camera = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        codec = 0x47504A4D # MJPG
        # print("Setting camera mode")
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FOURCC, codec)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        return camera
    # print("Setting camera mode")


    while input("Do You Want To Calibrate NO. {}? [y/n]".format(capture_num)) == "y":
        os.makedirs(os.path.join(capture_dir,"output_%d"%capture_num),exist_ok=True)
        for img_counter,grey_img in enumerate(tqdm(greycode_list)):
            src_path = os.path.join(greycode_dir,grey_img)
            # replace img src
            relpath = os.path.relpath(src_path,os.getcwd())
            soup.find('img')['src'] = ".\\"+relpath
            # save the file again
            with open("calibrate.html", "w") as outf:
                outf.write(str(soup))     
            driver.refresh()
            # capture and save
            img_name = "capture_{}.png".format(str(img_counter).zfill(2))
    
            img_name = os.path.join(capture_dir,"output_{}".format(capture_num),img_name)
            camera = capture()
            ret,frame = camera.read()
            if cam_show == True:
                cv2.imshow("frame",frame)
            driver.refresh()
            time.sleep(wait_time) 
            cv2.imwrite(img_name,frame)
            camera.release()
            del ret,frame,camera
        capture_num += 1
        soup.find('img')['src'] = ".\\test\\Projector-Camera-Calibration\\graycode_pattern\\pattern_40.png"
        # save the file again
        with open("calibrate.html", "w") as outf:
            outf.write(str(soup))
        driver.refresh()
    cv2.destroyAllWindows()

def calibrate_gamma(wait_time=1,cam_show=False,cam_id=1,exp_val=-4,fps=30,capture_width=4024,capture_height=3036):
    import os,cv2,bs4
    current_dir = os.getcwd()
    greycode_dir = os.path.join(current_dir,"test","phase-shifting","gamma_correction_patterns")
    capture_dir = os.path.join(current_dir,"test","phase-shifting","gamma_correction")
    greycode_list = [i for i in os.listdir(greycode_dir) if os.path.splitext(i)[1]==".png"]
    # load the file
    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat05.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))

    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat51.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))
    capture_num = 1

    from selenium import webdriver
    from webdriver_manager.firefox import GeckoDriverManager

    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    webpath = os.path.join(os.getcwd(),'calibrate.html')
    driver.get(webpath)
      
    # print("Initializing camera")
    # camera = VideoCapture(cam_id=cam_id,exp_val=exp_val,fps=fps,capture_width=capture_width,capture_height=capture_height)
    def capture():
        camera = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        codec = 0x47504A4D # MJPG
        # print("Setting camera mode")
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FOURCC, codec)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        return camera
    # print("Setting camera mode")


    while input("Do You Want To Calibrate NO. {}? [y/n]".format(capture_num)) == "y":
        os.makedirs(os.path.join(capture_dir),exist_ok=True)
        for img_counter,grey_img in enumerate(tqdm(greycode_list)):
            src_path = os.path.join(greycode_dir,grey_img)
            # replace img src
            relpath = os.path.relpath(src_path,os.getcwd())
            soup.find('img')['src'] = ".\\"+relpath
            # save the file again
            with open("calibrate.html", "w") as outf:
                outf.write(str(soup))     
            driver.refresh()
            # capture and save
            img_name = "pat{}.png".format(str(img_counter).zfill(2))
    
            img_name = os.path.join(capture_dir,img_name)
            camera = capture()
            ret,frame = camera.read()
            if cam_show == True:
                cv2.imshow("frame",frame)
            driver.refresh()
            time.sleep(wait_time) 
            cv2.imwrite(img_name,frame)
            camera.release()
            del ret,frame,camera
        capture_num += 1
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat51.png"
        # save the file again
        with open("calibrate.html", "w") as outf:
            outf.write(str(soup))
        driver.refresh()
    cv2.destroyAllWindows()

def web_host_start():
    from selenium import webdriver
    from webdriver_manager.firefox import GeckoDriverManager

    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    webpath = os.path.join(os.getcwd(),'calibrate.html')
    driver.get(webpath)
    return driver

def capture_gamma(start_flag = 0, output_dir='captures',wait_time=1,cam_show=False,cam_id=1,exp_val=-4,fps=30,capture_width=4024,capture_height=3036):
    import os,cv2,bs4
    current_dir = os.getcwd()
    greycode_dir = os.path.join(current_dir,"test","phase-shifting","gamma_correction_patterns")
    capture_dir = os.path.join(current_dir,"test","phase-shifting",output_dir)
    greycode_list = [i for i in os.listdir(greycode_dir) if os.path.splitext(i)[1]==".png"]
    # load the file
    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat05.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))

    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat51.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))
    capture_num = 1

      
    # print("Initializing camera")
    # camera = VideoCapture(cam_id=cam_id,exp_val=exp_val,fps=fps,capture_width=capture_width,capture_height=capture_height)
    def capture(exp_val0=exp_val):
        camera = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        codec = 0x47504A4D # MJPG
        # print("Setting camera mode")
        camera.set(cv2.CAP_PROP_FPS, fps)
        camera.set(cv2.CAP_PROP_FOURCC, codec)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        camera.set(cv2.CAP_PROP_EXPOSURE, exp_val0)
        return camera
    # print("Setting camera mode")

    def cap_write(capture_num = capture_num,capture_dir=capture_dir):
        os.makedirs(os.path.join(capture_dir),exist_ok=True)
        for img_counter,grey_img in enumerate(tqdm(greycode_list)):
            src_path = os.path.join(greycode_dir,grey_img)
            # replace img src
            relpath = os.path.relpath(src_path,os.getcwd())
            soup.find('img')['src'] = ".\\"+relpath
            # save the file again
            with open("calibrate.html", "w") as outf:
                outf.write(str(soup))     
            driver.refresh()
            # capture and save
            img_name = "pat{}.png".format(str(img_counter).zfill(2))

            img_name = os.path.join(capture_dir,img_name)
            camera = capture(exp_val0=exposure_time)
            ret,frame = camera.read()
            if cam_show == True:
                cv2.imshow("frame",frame)
            driver.refresh()
            time.sleep(wait_time) 
            cv2.imwrite(img_name,frame)
            camera.release()
            del ret,frame,camera
        capture_num += 1
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat00.png"
        # save the file again
        with open("calibrate.html", "w") as outf:
            outf.write(str(soup))
        driver.refresh()
        cv2.destroyAllWindows()



    driver = web_host_start()
    if input("Do You Want To Continue? [y/n]") == "n":
        return
    else:
        pass
    if isinstance(exp_val,list):
        for exposure_time in exp_val:
            exposure_time = int(exposure_time)
            if exposure_time in [0,1]:
                wait_time = 10
            elif exposure_time in [-3,-2,-1]:
                wait_time = 5
            else:
                wait_time = 0
            capture_num = 1
            output_dir = 'capture_exposure{}'.format(exposure_time)
            capture_dir = os.path.join(current_dir,"test","phase-shifting",output_dir)
            cap_write(capture_dir=capture_dir)
    
    else:
        exposure_time = exp_val
        cap_write(capture_dir=capture_dir)


def HDR_gamma(wait_time=1,cam_show=False,cam_id=1,exp_val=-4,fps=30,capture_width=4024,capture_height=3036):
    import os,cv2,bs4
    current_dir = os.getcwd()
    greycode_dir = os.path.join(current_dir,"test","phase-shifting","gamma_correction_patterns")
    
    greycode_list = [i for i in os.listdir(greycode_dir) if os.path.splitext(i)[1]==".png"]
    # load the file
    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat00.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))

    with open("calibrate.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt)
        soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat00.png"
    # save the file again
    with open("calibrate.html", "w") as outf:
        outf.write(str(soup))
    capture_num = 1

    from selenium import webdriver
    from webdriver_manager.firefox import GeckoDriverManager

    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    webpath = os.path.join(os.getcwd(),'calibrate.html')
    driver.get(webpath)
      
    # print("Initializing camera")
    # camera = VideoCapture(cam_id=cam_id,exp_val=exp_val,fps=fps,capture_width=capture_width,capture_height=capture_height)

    # print("Setting camera mode")


    # while input("Do You Want To GO? [y/n]") == "y":
    #     for exposure_time in exp_val:
    #         capture_dir = os.path.join(current_dir,"test","phase-shifting","captures_exposure%d"%(exposure_time))
    #         os.makedirs(os.path.join(capture_dir),exist_ok=True)
    #         if exposure_time in [0,1]:
    #             wait_time = 10
    #         elif exposure_time in [-3,-2,-1]:
    #             wait_time = 5
    #         else:
    #             wait_time = 0
            
    #         capture_core(greycode_list,capture_dir)

    #         capture_num += 1
    #         soup.find('img')['src'] = ".\\test\\phase-shifting\\gamma_correction_patterns\\pat00.png"
    #         # save the file again
    #         with open("calibrate.html", "w") as outf:
    #             outf.write(str(soup))
    #         driver.refresh()
    # cv2.destroyAllWindows()
       

import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, cam_id=1,exp_val=-4,fps=30,capture_width=4024,capture_height=3036):
    self.cap = cv2.VideoCapture(cam_id)
    self.cap.set(cv2.CAP_PROP_FPS, fps)
    codec = 0x47504A4D # MJPG
    self.cap.set(cv2.CAP_PROP_FOURCC, codec)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


def showPIL(pilImage):
    root = tkinter.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (w, h))
    root.focus_set()    
    root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
    canvas = tkinter.Canvas(root,width=w,height=h)
    canvas.pack()
    canvas.configure(background='black')
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(pilImage)
    imagesprite = canvas.create_image(w/2,h/2,image=image)
    root.mainloop()

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def comparision(f1,f2):
    f1 = cv2.imread(f1)
    f2 = cv2.imread(f2)
    # return np.sqrt(np.abs(f1**2-f2**2))
    return np.abs(f1-f2)