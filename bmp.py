import math
from colorama import *
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

def read_rows(path):
    image_file = open(path, "rb")
    # Blindly skip the BMP header.
    global header
    header = image_file.read(54)
    # image_file.seek(54)

    # We need to read pixels in as rows to later swap the order
    # since BMP stores pixels starting at the bottom left.
    rows = []
    row = []
    pixel_index = 0

    while True:
        if pixel_index == 512:
            pixel_index = 0
            rows.insert(0, row)
            if len(row) != 512 * 3:
                raise Exception("Row length is not 512*3 but " + str(len(row)) + " / 3.0 = " + str(len(row) / 3.0))
            row = []
        pixel_index += 1

        r_string = image_file.read(1)
        g_string = image_file.read(1)
        b_string = image_file.read(1)

        if len(r_string) == 0:
            # This is expected to happen when we've read everything.
            if len(rows) != 512:
                print(
                    'Warning!!! Read to the end of the file at the correct sub-pixel (red) but we\'ve not read 512 '
                    'rows!')
            break

        if len(g_string) == 0:
            print("Warning!!! Got 0 length string for green. Breaking.")
            break

        if len(b_string) == 0:
            print("Warning!!! Got 0 length string for blue. Breaking.")
            break

        r = ord(r_string)
        g = ord(g_string)
        b = ord(b_string)

        row.append(b)
        row.append(g)
        row.append(r)

    image_file.close()

    return rows


def repack_sub_pixels(rows):
    print("Repacking pixels...")
    sub_pixels = []
    for row in reversed(rows):
        for sub_pixel in row:
            sub_pixels.append(sub_pixel)

    return sub_pixels


def filterblue(s_pixels):
    blue = []
    blueret = []
    image_filee = open('asserts/newblue.bmp', "wb")
    image_filee.write(bytearray(header))
    for i in s_pixels[0::3]:
        blue.append(i)
        blue.append(0)
        blue.append(0)
    image_filee.write(bytearray(blue))
    image_filee.close()
    for i in s_pixels[0::3]:
        blueret.append(i)
    return blueret


def filtergreen(s_pixels):
    green = []
    greenret = []
    image_filee = open('asserts/newgreen.bmp', "wb")
    image_filee.write(bytearray(header))
    for i in s_pixels[1::3]:
        green.append(0)
        green.append(i)
        green.append(0)
    image_filee.write(bytearray(green))
    image_filee.close()
    for i in s_pixels[1::3]:

        greenret.append(i)

    return greenret


def filterred(s_pixels):
    red = []
    redret = []
    image_filee = open('asserts/newred.bmp', "wb")
    image_filee.write(bytearray(header))
    for i in s_pixels[2::3]:
        red.append(0)
        red.append(0)
        red.append(i)
    image_filee.write(bytearray(red))
    image_filee.close()
    for i in s_pixels[2::3]:

        redret.append(i)

    return redret


def writered(comp, name):
    image_file = open('asserts/'+name+'.bmp','wb')
    image_file.write(bytearray(header))
    res = []
    for i in comp:
        res.append(0)
        res.append(0)
        res.append(int(i))
    image_file.write(bytearray(res))
    image_file.close()


def writegreen(comp, name):
    image_file = open('asserts/' + name + '.bmp', 'wb')
    image_file.write(bytearray(header))
    res = []
    for i in comp:
        res.append(0)
        res.append(int(i))
        res.append(0)
    image_file.write(bytearray(res))
    image_file.close()


def writeblue(comp, name):
    image_file = open('asserts/' + name + '.bmp', 'wb')
    image_file.write(bytearray(header))
    res = []
    for i in comp:
        res.append(int(i))
        res.append(0)
        res.append(0)
    image_file.write(bytearray(res))
    image_file.close()


def writeycbcr(comp, name):
    image_file = open('asserts/'+name+'.bmp', 'wb')
    image_file.write(bytearray(header))
    res = []
    for i in comp:
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
    image_file.write(bytearray(res))
    image_file.close()
def writepic(red,green,blue, name):
    image_file = open('asserts/'+name+'.bmp', 'wb')
    image_file.write(bytearray(header))
    res = []
    for i in range(len(blue)):
        res.append(int(blue[i]))
        res.append(int(green[i]))
        res.append(int(red[i]))
    image_file.write(bytearray(res))
    image_file.close()

def writeycbcrrestored(comp, name):
    image_file = open('asserts/'+name+'.bmp','wb')
    image_file.write(bytearray(header))
    res = []
    num = 0
    for it, i in enumerate(comp):

        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
    print(num)
    image_file.write(bytearray(res))
    image_file.close()
    return res


def writeycbcrrestoredb(comp, name):
    image_file = open('asserts/'+name+'.bmp','wb')
    image_file.write(bytearray(header))
    res = []
    num = 0
    for it, i in enumerate(comp):

        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1
        res.append(int(i))
        res.append(int(i))
        res.append(int(i))
        num += 1

    print(num)
    image_file.write(bytearray(res))
    image_file.close()
    return res


def clipping(comp):
    ret = []
    cur = 0
    for i in range(len(comp)):
        cur = comp[i]
        if comp[i] > 255:
            cur = 255

        elif comp[i] < 0:
            cur = 0
        ret.append(cur)

    return ret


def psnr(comp, restcomp):
    #print (len(comp),len(restcomp),'PSNR INPUT')
    sum = 0
    for i in range(len(restcomp)):
        sum += (comp[i]-restcomp[i])**2
    #wh = 512*512*255**2
    wh = 0
    for j in range(len(comp)):
        wh += 255 ** 2
    #print(wh,'wh')
    psn = 10*math.log10(wh//sum)
    return psn


def mato(comp):
    tmp = 0
    for i in range(len(comp)):
        tmp += comp[i]

    res = (1/(512**2))*tmp
    print(res)
    return res


def quadr(comp):
    tmp = 0
    m = mato(comp)
    #print(m)
    for i in range(len(comp)):
        tmp += (comp[i] - m)**2

    #print(tmp)

    res = math.sqrt((1/262143)*tmp)
    return res


def cor(comp1, comp2):
    #print(len(comp1), len(comp2), 'cor INPUT')
    ma = int(mato(comp1))
    mb = int(mato(comp2))
    print(ma,mb,'MATO')
    aminma = [i - ma for i in comp1]

    bminba = [i - mb for i in comp2]

    print(len(aminma), len(bminba))
    tc = [aminma[i]*bminba[i] for i in range(len(aminma))]
    core = mato(tc) / (quadr(comp1) * quadr(comp2))
    return core
def shift(steps,stepsy,lst):
    """ Циклический кольцевой сдвиг списка до минимума"""
    lst = lst[steps:] + lst[:steps] # 1-й вариант
    if stepsy < 0:
        stepsy = abs(stepsy)

    for i in range(len(lst)):
        if i + stepsy*512 >= len(lst):
            return lst
        lst[i] = lst[i+stepsy*512]
    return lst



def supermandecimate(comp):
    todel = []
    for i in range(512):
        if i % 2 == 0:
            todel.append(i)
    c = np.array(comp)
    c = np.reshape(c, (512, 512))
    for i in range(1,len(c)-1,1):
        for j in range(1,len(c)-1,1):
            c[i][j] = (c[i-1][j]+c[i+1][j]+c[i][j-1]+c[i][j+1])/4

    c = c.reshape((262144))
    return list(c)


def cbcrdecimate(comp):
    todel = []
    for i in range(512):
        if i % 2 == 0:
            todel.append(i)
    c = np.array(comp)
    c = np.reshape(c,(512,512))
    #c = np.delete(c, (todel), axis=0)
    for i in range(1,len(c),2):
        c[i-1] = c[i]
    for i in range(len(c)):
        for j in range(1,len(c[i]),2):
            c[i][j-1] = c[i][j]
    c = c.reshape((262144))
    return list(c)








def main():
    rows = read_rows("asserts/trash.bmp")


    sub_pixels = repack_sub_pixels(rows)
    #sub_pixels = [i for i in reversed(sub_pixels)]
    blue = filterblue(sub_pixels)

    green = filtergreen(sub_pixels)

    red = filterred(sub_pixels)

    blueshaped = np.array(blue)
    blueshaped = blueshaped.reshape((512,512))
    redshaped = np.array(red)
    redshaped = redshaped.reshape((512, 512))
    greenshaped = np.array(green)
    greenshaped = greenshaped.reshape((512, 512))
    print(greenshaped.shape)

    m1 = [] #green
    m2 = []
    m3 = [] #red
    m4 = []
    m5 = [] #blue
    m6 = []
    #
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(green, shift(i, 0, green)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="green=%s" % ('0',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(green, shift(i, 5, green)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="green=%s" % ('5',))
    #
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(green, shift(i, -5, green)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="green=%s" % ('-5',))
    #
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(green, shift(i, 10, green)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="green=%s" % ('10',))
    #
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(green, shift(i, -10, green)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="green=%s" % ('-10',))
    #
    # #plt.plot(m3, m4, label="l=%s"%('red',))
    # #plt.plot(m5, m6, label="l=%s"%('blue',))
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    # plt.show()
    # print(greensmesh5,'GREENSNESH')
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(red, shift(i, 0, red)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="red=%s" % ('0',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(red, shift(i, 5, red)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="red=%s" % ('5',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(red, shift(i, -5, red)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="red=%s" % ('-5',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(red, shift(i, 10, red)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="red=%s" % ('10',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(red, shift(i, -10, red)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="red=%s" % ('-10',))
    #
    # # plt.plot(m3, m4, label="l=%s"%('red',))
    # # plt.plot(m5, m6, label="l=%s"%('blue',))
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    # plt.show()
    # print(greensmesh5, 'GREENSNESH')
    #
    #
    #
    #
    #

    #
    #
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(blue, shift(i, 0, blue)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="blue=%s" % ('0',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(blue, shift(i, 5, blue)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="blue=%s" % ('5',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(blue, shift(i, -5, blue)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="blue=%s" % ('-5',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(blue, shift(i, 10, blue)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="blue=%s" % ('10',))
    #
    # m5.clear()
    # m6.clear()
    # for i in range(0, 105, 10):
    #     greensmesh5 = math.fabs(cor(blue, shift(i, -10, blue)))
    #
    #     m5.append(i)
    #     m6.append(greensmesh5)
    # plt.plot(m5, m6, label="blue=%s" % ('-10',))
    #
    # # plt.plot(m3, m4, label="l=%s"%('red',))
    # # plt.plot(m5, m6, label="l=%s"%('blue',))
    # leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # leg.get_frame().set_alpha(0.5)
    # plt.show()
    # print(greensmesh5, 'GREENSNESH')





    corg = math.fabs(cor(green, blue))
    corb = math.fabs(cor(red, blue))
    corr = math.fabs(cor(red, green))
    corgg = math.fabs(cor(green, green))
    print(Fore.LIGHTGREEN_EX+'Cor GB', corg)
    print('Cor RB', corb)
    print('Cor RG', corr)
    print(''+Style.RESET_ALL)
    print(len(green), len(red), len(blue))
    ycomp = [int(0.299*red[i]+0.587*green[i]+0.114*blue[i]) for i in range(len(green))]
    cbcomp = [int(128 - (0.168736*red[i])-(0.331264 * green[i]) + 0.5*blue[i]) for i in range(len(green))]
    crcomp = [int(128 + (0.5 * red[i]) - (0.418688 * green[i])- (0.081213 * blue[i])) for i in range(len(green))]
    writeycbcr(ycomp, 'ycomp')
    writeycbcr(cbcomp, 'cbcomp')
    writeycbcr(crcomp, 'crcomp')
    corg = math.fabs(cor(ycomp, cbcomp))
    corb = math.fabs(cor(ycomp, crcomp))
    corr = math.fabs(cor(cbcomp, crcomp))
    corgg = math.fabs(cor(cbcomp, cbcomp))
    print(Fore.LIGHTGREEN_EX+'Cor YCB', corg)
    print('Cor YCR', corb)
    print('Cor CRCB', corr)

    grestored = [int(ycomp[i]-0.34414*(cbcomp[i]-128)-0.71414*(crcomp[i]-128))for i in range(len(green))]
    rrestored = [int(ycomp[i] + 1.402*(crcomp[i]-128))for i in range(len(red))]
    brestored = [int(ycomp[i]+1.772*(cbcomp[i]-128))for i in range(len(blue))]
    grestored = clipping(grestored)
    rrestored = clipping(rrestored)
    brestored = clipping(brestored)
    writeblue(brestored, 'bluerestored')
    writegreen(grestored, 'greenrestored')
    writered(rrestored, 'redrestored')
    print('PSNR', psnr(blue, brestored))
    print('PSNR', psnr(green, grestored))
    print('PSNR', psnr(red, rrestored))
    print('' + Style.RESET_ALL)
    # ПРАВИЛЬНО тк сравнение двух одинаковых дает divbyzero
    # cbret = []
    # cbretnorm = []
    # cbdec = np.array(cbcomp)
    # cbdec = cbdec.reshape((512, 512))
    # for i, data in enumerate(cbdec):
    #     if i % 2 == 0:
    #         continue
    #
    #     for j, data in enumerate(cbdec[i]):
    #         if j % 2 == 0:
    #             continue
    #         else:
    #             cbret.append(data)
    #             cbretnorm.append(data)
    # print(len(cbret))
    #
    # crred = []
    # crrednorm = []
    # crdec = np.array(crcomp)
    # crdec = crdec.reshape((512, 512))
    # for i, data in enumerate(crdec):
    #     if i % 2 == 0:
    #         continue
    #
    #     for j, datas in enumerate(crdec[i]):
    #         if j % 2 == 0:
    #             continue
    #         else:
    #             crred.append(datas)
    #             crrednorm.append(datas)
    # print(len(crred))
    # print(Fore.LIGHTGREEN_EX+'(A)PSNR cb restored', psnr(cbret, cbcomp))
    # print('(A)PSNR cr restored', psnr(crred, crcomp), ''+Style.RESET_ALL)
    # crred = writeycbcrrestored(crred, 'crredrest')
    # cbret = writeycbcrrestored(cbret, 'cbretrest')
    # gfycbcr = [ycomp[i] - 0.714 * (crred[i] - 128) - 0.334 * (cbret[i] - 128) for i in range(len(ycomp))]
    # rfycbcr = [ycomp[i] + 1.402 * (crred[i] - 128) for i in range(len(ycomp))]
    # bfycbcr = [ycomp[i] + 1.772 * (cbret[i] - 128) for i in range(len(ycomp))]
    # gfycbcr = clipping(gfycbcr)
    # rfycbcr = clipping(rfycbcr)
    # bfycbcr = clipping(bfycbcr)
    # print(Fore.LIGHTGREEN_EX+'(A)PSNR G', psnr(green, gfycbcr))
    # print('(A)PSNR R', psnr(red, rfycbcr))
    # print('(A)PSNR B', psnr(blue, bfycbcr), ''+Style.RESET_ALL)

    cbdecim = []
    crdecim = []

    cbvoss = []
    crvoss = []
    cbshaped = np.array(cbcomp).reshape((512,512))
    for i in range(len(cbshaped)):
        for j in range(1,len(cbshaped),2):
            cbshaped[i][j] = cbshaped[i][j-1]

    todel = []
    for i in range(512):
        if i % 2 == 0:
            todel.append(i)

    #cbshaped = np.delete(cbshaped, (todel), axis=0)
    cbshaped = np.delete(cbshaped, (todel), axis=1)
    print(cbshaped.shape)

    for i in range(len(cbshaped)):
        #cbvoss.extend(cbshaped.tolist())
        #cbvoss.extend(cbshaped.tolist())
        for j in range(len(cbshaped[i])):
            for l in range(2):
                cbvoss.append(int(cbshaped[i][j]))

    print(len(cbvoss))

    crshaped = np.array(crcomp).reshape((512, 512))

    for i in range(len(crshaped)):
        for j in range(1, len(crshaped), 2):
            crshaped[i][j] = crshaped[i][j - 1]

    todel = []
    for i in range(512):
        if i % 2 == 0:
            todel.append(i)

    #crshaped = np.delete(crshaped, (todel), axis=0)
    crshaped = np.delete(crshaped, (todel), axis=1)
    print(crshaped.shape)

    for i in range(len(crshaped)):
        for j in range(len(cbshaped[i])):
            for l in range(2):
                crvoss.append(int(cbshaped[i][j]))

    crvoss = cbcrdecimate(crcomp)
    cbvoss = cbcrdecimate(cbcomp)
    print(len(crvoss))
    g = [int(ycomp[i] - 0.7169 * (crvoss[i] - 128) - 0.3455 * (cbvoss[i] - 128)) for i in range(len(ycomp))]
    r = [int(ycomp[i] + 1.4075 * (crvoss[i] - 128)) for i in range(len(ycomp))]
    b = [int(ycomp[i] + 1.7790 * (cbvoss[i] - 128)) for i in range(len(ycomp))]
    r = clipping(r)
    g = clipping(g)
    b = clipping(b)
    writered(r,'wow')
    writeblue(b, 'wow1')
    writegreen(g, 'wow2')
    print(psnr(red,r))
    print(psnr(green, g))
    print(psnr(blue, b))
    print(psnr(cbvoss, cbcomp))
    print(psnr(crvoss, crcomp))
    print(len(cbcrdecimate(cbcomp)))
    writepic(b,g,r,'lol')


    #bbbbbbb
    crvossb = supermandecimate(crcomp)
    cbvossb = supermandecimate(cbcomp)
    print(len(crvossb))
    g = [int(ycomp[i] - 0.7169 * (crvossb[i] - 128) - 0.3455 * (cbvossb[i] - 128)) for i in range(len(ycomp))]
    r = [int(ycomp[i] + 1.4075 * (crvossb[i] - 128)) for i in range(len(ycomp))]
    b = [int(ycomp[i] + 1.7790 * (cbvossb[i] - 128)) for i in range(len(ycomp))]
    r = clipping(r)
    g = clipping(g)
    b = clipping(b)
    writered(r,'wowBB')
    writeblue(b, 'wowBB1')
    writegreen(g, 'wowBB2')
    print(psnr(red,r))
    print(psnr(green, g))
    print(psnr(blue, b))
    print(psnr(cbvossb, cbcomp))
    print(psnr(crvossb, crcomp))
    print(len(supermandecimate(cbcomp)))
    writepic(b,g,r,'lolB')



if __name__ == '__main__':
    main()
