from tinygrad import Tensor, nn, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.helpers import get_child
import numpy as np
from PIL import Image
from skimage import io
import os
from model.u2net_tiny import U2NET, U2NETP
import time
import cv2
import argparse
from pathlib import Path

def normPRED(d):
    ma, mi = d.max(), d.min()
    return (d-mi)/(ma-mi)

def save_output(image_name,predict_np,d_dir):
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'/'+imidx+'_out.jpg')

def load_and_predict(file, model, net):
    image = cv2.imread(file)
    if model != "portrait":
        image = cv2.resize(image, (320,320))

    print(f"Running U^2 Net on device: {Device.DEFAULT} for file: {file}")
    start = time.perf_counter()
    pred = inference(net, image)
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    print(f"Inference time: {elapsed_ms:.3f} ms")

    save_output(file, pred, "./output")
    return pred

def inference(net, input):
    # normalize the input
    tmpImg = np.zeros((input.shape[0],input.shape[1],3))
    input = input/np.max(input)

    tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
    tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

    # convert BGR to RGB
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis,:,:,:]
    tmpTensor = Tensor(tmpImg.astype(np.float32))

    # inference
    d1,d2,d3,d4,d5,d6,d7= net(tmpTensor)

    # normalization
    pred = 1.0 - d1[:,0,:,:]
    pred = normPRED(pred)

    # convert tinygrad tensor to numpy array
    pred = pred.squeeze()
    pred = pred.numpy()

    del d1,d2,d3,d4,d5,d6,d7

    return pred

def sky_replacement(photo_np: np.array, sky_np: np.array, mask_np: np.array):
    photo_tensor = Tensor(photo_np)
    sky_tensor = Tensor(sky_np)
    mask_tensor = Tensor(mask_np)
    mask_tensor = mask_tensor.unsqueeze(-1).repeat(1,1,3).interpolate(photo_tensor.shape)
    sky_tensor = sky_tensor.interpolate(photo_tensor.shape)
    masked_out_sky = (mask_tensor*photo_tensor).cast(dtypes.uchar)
    new_sky = ((1.0-mask_tensor)*sky_tensor).cast(dtypes.uchar)
    dominant_color_rgb = (sky_tensor/255.0).mean(axis=(0,1))
    r,g,b= masked_out_sky[:,:,0], masked_out_sky[:,:,1], masked_out_sky[:,:,2]
    r_m, g_m, b_m = dominant_color_rgb
    blended = (((r/255.0)*r_m).stack((g/255.0)*g_m, (b/255.0)*b_m, dim=2)*255).cast(dtypes.uchar)
    new_composite = blended + new_sky

    return new_composite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U^2 Net on tinygrad")

    parser.add_argument(
        "-i",
        type=str,
        default="./example_data/test2.jpg",
        help="Path to the input image or folder containing images"
    )

    parser.add_argument(
        "-m",
        type=str,
        default="fg_small",
        help="Model to load"
    )

    parser.add_argument(
        "-j",
        type=bool,
        default=True,
        help="Whether to use jit"
    )

    parser.add_argument(
        "-sky",
        type=str,
        help="Path to the image of the new sky"
    )

    args = parser.parse_args()

    unet = U2NET(3,1)

    @TinyJit
    def jit_unet(x):
        return unet(x)

    if args.m == "fg_small" or args.m == "sky_small":
        unet = U2NETP(3,1)

    print("Loading weights...")

    if args.m == "fg_small":
        loaded = nn.state.torch_load("./weights/u2netp_fg.pth")
    elif args.m == "fg":
        loaded = nn.state.torch_load("./weights/u2net_fg.pth")
    elif args.m == "portrait":
        loaded = nn.state.torch_load("./weights/u2net_portrait.pth")
    elif args.m == "sky_small":
        loaded = nn.state.torch_load("./weights/u2netp_sky.pth")
    else:
        raise RuntimeError(f"Unknown model selected={args.m}")

    for k, v in loaded.items():
      get_child(unet, k).assign(v.numpy()).realize()

    os.makedirs("./output", exist_ok=True)

    if args.sky:
        input_path = Path(args.i)
        assert Path(args.sky).exists(), "The provided sky image path does not exist"
        assert Path(args.sky).is_file(), "Sky replacement only works on single image"
        assert args.m == "sky_small", "For sky replacement use sky_small model"
        sky_np = io.imread(args.sky)
        photo_np = io.imread(args.i)
        pred_np = load_and_predict(args.i, args.m, unet if not args.j else jit_unet)
        new_sky_out = sky_replacement(photo_np, sky_np, pred_np)
        output_path =f"./{input_path.stem}_new_sky{input_path.suffix}"
        io.imsave(output_path, new_sky_out.numpy())
    elif Path(args.i).is_dir():
        for file in os.listdir(args.i):
            if "_out." in file: continue
            load_and_predict(os.path.join(args.i, file), args.m, unet if not args.j else jit_unet)
    else:
        load_and_predict(args.i, args.m, unet if not args.j else jit_unet)
