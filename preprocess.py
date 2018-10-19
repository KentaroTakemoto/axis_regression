import numpy as np
from PIL import Image

model_names = ["79a8b5fdd40e1b1d20768660cf080d12","eb2843ff62280f5320768660cf080d12"]

def select_angles(model_number, test_angles):
    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    if model_number==1:
        angles = [-a for a in angles]
        for i in range(len(test_angles)):
            test_angles[i] = str(int(test_angles[i])*-1)

    ang1 = test_angles[0]
    ang2 = test_angles[0]
    while str(ang1) in test_angles or str(ang2) in test_angles :
        ang1,ang2 = np.random.choice(angles,2,replace=False)
    return str(ang1), str(ang2)

def load_data(view_param, model_number, mode="data", test_angles=None, size=None,n_out=7, xp=np):
    def cropping(img,size):
        w,h = img.size
        if w < h:
            if w < size:
                img = img.resize((size, size*h//w))
                w, h = img.size
        else:
            if h < size:
                img = img.resize((size*w//h, size))
                w, h = img.size
        return img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))


    param = view_param.split(' ')
    azimuth_deg = float(param[0])
    elevation_deg = float(param[1])
    theta_deg = int(param[2])
    theta_deg = (-1*theta_deg)%360
    rho = float(param[3])
    filename = "a%03d_e%03d_t%03d_d%03d" % (round(azimuth_deg), round(elevation_deg), int(theta_deg), round(rho*100))
    print('filename : {}'.format(filename))

    if mode=="label":
        txt_path = "/home/mil/takemoto/other_githubs/RenderForCNN/test_results/results/02933112_test_results/annotation/{}/0_{}.txt".format(model_names[model_number], filename)
        with open(txt_path,'r') as f:
            y=xp.array(f.read().split(' '))
            if n_out==7:
                y = y[xp.array([0,1,2,3,4,6,7])]
        ############################################################
        for i in range(3):
            y[i] = y[i] * 0.5 + 0.5
        ############################################################
        print("y : {}".format(y))
        return y

    elif mode=="data":
        ang1, ang2 = select_angles(model_number, test_angles)
        image_path1 = "/home/mil/takemoto/other_githubs/RenderForCNN/test_results/results/02933112_test_results/data/{}/{}_{}.png".format(model_names[model_number], ang1, filename)
        image_path2 = "/home/mil/takemoto/other_githubs/RenderForCNN/test_results/results/02933112_test_results/data/{}/{}_{}.png".format(model_names[model_number], ang2, filename)
        print('image_path1 : {}'.format(image_path1))


        img = Image.open(image_path1)
        w,h = img.size
        if w!=size or h!= size:
            img = cropping(img,size)
        x1 = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        x1 -= 120
        print("x1.shape : ".format(x1.shape))

        img = Image.open(image_path2)
        w,h = img.size
        if w!=size or h!= size:
            img = cropping(img,size)
        x2 = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        x2 -= 120

        x = xp.concatenate([x1,x2], axis=1)
        print("x.shape : ".format(x.shape))
        return x

    else:
        print('Error!')
        exit()
