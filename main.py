import os

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import math
import scipy.signal

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# Defining functions for QIM

def Open(Path):
    image = Image.open(Path)
    return np.asarray(ImageOps.grayscale(image))


def Save(Path, img):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(Path)


def ImShow(image, Title):
    cv2.imshow(Title, image)
    cv2.waitKey(0)


def ImWrite(Path, image):
    cv2.imwrite(Path, image)


def ImRead(Path):
    return cv2.imread(Path, cv2.IMREAD_GRAYSCALE)


def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


def Watermarking(original, ori_wm, key, alpha, step_size, name_of_file):
    watermarked = dct2(original)
    # Embed Watermark in DCT
    columns = np.shape(ori_wm)[1]
    lines = np.shape(ori_wm)[0]
    for k in range(lines):
        for j in range(columns):
            quantized = step_size * round(
                (watermarked[k + 100][j + 100] - step_size * ((ori_wm[k][j] / 2) + key)) / step_size) - (
                            watermarked[k + 100][j + 100] - step_size * ((ori_wm[k][j] / 2) + key))
            watermarked[k + 100][j + 100] = watermarked[k + 100][j + 100] + (alpha * quantized)

    output = idct2(watermarked)
    np.save("Watermarked/Watermarked" + str(name_of_file) + ".npy", output)
    print("Watermarked image " + str(name_of_file) + " created")



def DetectWatermark(image, key, alpha, step_size):

    bis_image = dct2(image)
    columns = 64
    lines = 64
    result = np.full((lines, columns), 255)

    for k in range(lines):
        for j in range(columns):
            decision = step_size * round((bis_image[k + 100][j + 100] - (key * step_size)) / step_size) - (
                bis_image[k + 100][j + 100]) + (key * step_size)
            if abs(decision) < ((1 - alpha) * (step_size / 2)):
                result[k][j] = 0
            else:
                result[k][j] = 255
    Save("Retrieved/Retrieved.png", result)


def CreateWatermark(created_wm):
    fnt = ImageFont.truetype('arial.ttf', 20)
    image = Image.new(mode="RGB", size=(64, 64), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), created_wm, font=fnt, fill=(0, 0, 0))
    image.save("Watermarks/Watermark" + str(created_wm) + ".png")
    print("Watermark successfully created")


def PSNR(original, watermarked):
    err = np.sum((original.astype("float") - watermarked.astype("float")) ** 2)
    err /= float(original.shape[0] * watermarked.shape[1])
    result = 10 * math.log10(255 ** 2 / err)
    return result


def MSE(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def NCC(imageA, imageB):
    return scipy.signal.correlate(imageA, imageB, "same")


def ReadWatermark(file):
    img = ImRead(file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    read_wm = pytesseract.image_to_string(img_rgb, config="--psm 11 tessedit_char_whitelist=0123456789")
    return read_wm[:3]


def randomImageDetection(num, key, alpha, step_size):
    i = 0
    occurences = open("Occurences.txt", "r")
    successes = 0
    missed = 0
    errors = 0
    lines = occurences.readlines()
    while i < num:
        name = "{:03d}".format(np.random.randint(1, 300))
        wmd = np.load("Watermarked/Watermarked" + str(name) + ".npy")
        ImWrite("temp/temp.png", wmd)
        ImShow(ImRead("temp/temp.png"), "Image " + str(name))
        DetectWatermark(wmd, key, alpha, step_size)
        retrievedwm = ReadWatermark("Retrieved/Retrieved.png")
        ImShow(ImRead("Retrieved/Retrieved.png"), "Retrieved Watermark for Image " + str(name))
        #cv2.destroyAllWindows()
        #os.remove("Retrieved/Retrieved.png")
        #os.remove("temp/temp.png")
        if str(retrievedwm).isdigit():
            print("I found the watermark " + str(retrievedwm) + " for this image.")

            if str(name) == str(retrievedwm):
                print("Successful Retrieving.")
                split = lines[int(retrievedwm) - 1].split(' : ')
                new = split[1].replace("\n", "")
                new_line = str(split[0]) + " : " + str(int(new) + 1) + "\n"
                lines[int(retrievedwm) - 1] = new_line
                occurences = open("Occurences.txt", "w")
                occurences.writelines(lines)
                occurences.close()
                successes = successes + 1
            else:
                print("Missed detection.")
                split = lines[301].split(' : ')
                new = split[1].replace("\n", "")
                new_line = str(split[0]) + " : " + str(int(new) + 1) + "\n"
                lines[301] = new_line
                occurences = open("Occurences.txt", "w")
                occurences.writelines(lines)
                occurences.close()
                missed = missed + 1

        else:
            print("Error reading Watermark.")
            split = lines[300].split(' : ')
            new = split[1].replace("\n", "")
            new_line = str(split[0]) + " : " + str(int(new) + 1) + "\n"
            lines[300] = new_line
            occurences = open("Occurences.txt", "w")
            occurences.writelines(lines)
            occurences.close()
            errors = errors + 1
        i += 1
    plt.bar(("Detection errors","Successful detections", "Missed Detections"),(errors,successes,missed), align='center', alpha = 0.5)
    plt.show()


def AddNoise(image, filename):
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    np.save(filename, noisy)

#
# def ResizeAttack(image):
#     return (0)


# for i in range(1, 301):
#     watermark = "{:03d}".format(i)
#     CreateWatermark(watermark)


#




# f = open("Occurences.txt", "w+")
# for i in range(1, 301):
#     f.write("Image %d : 0\n" % i)
# f.write("Errors : 0\n")
# f.write("Missed Detections : 0\n")
# f.close()
# Al = 0.1
# perc_succ = []
# labels = []
# while Al < 1 :
#     for i in range(1, 301):
#         filename = "{:03d}".format(i)
#         ori = ImRead("Data/t" + str(filename) + ".png")
#         wm = ImRead("Watermarks/Watermark" + str(filename) + ".png")
#         Watermarking(ori, wm, key=0.5, alpha=Al, step_size=40, name_of_file=filename)
#
#     perc_succ.append(randomImageDetection(10, key=0.5, alpha=Al, step_size=40))
#     print(Al)
#     labels.append(Al)
#     temp = Al
#     Al = temp + 0.1
#
#
# plt.plot(labels, perc_succ, 'ro')
# plt.xlabel("Value of alpha")
# plt.ylabel("Success percentage")
# plt.show()

for i in range(1, 301):
    filename = "{:03d}".format(i)
    ori = ImRead("Data/t" + str(filename) + ".png")
    wm = ImRead("Watermarks/Watermark" + str(filename) + ".png")
    Watermarking(ori, wm, key=0.7, alpha=0.9, step_size=20, name_of_file=filename)
# #
#
# for i in range(1, 301):
#     filename = "{:03d}".format(i)
#     wm = np.load("Watermarked/Watermarked" + str(filename) + ".npy")
#     AddNoise(wm, "Watermarked/Watermarked" + str(filename) + ".npy")

randomImageDetection(10,key=0.7, alpha=0.9, step_size=20)


