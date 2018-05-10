import math
from colorama import *
import numpy as np
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

    sum = 0
    for i in range(len(comp)):
        sum += (comp[i]-restcomp[i])**2
    wh = (512**2)*(255**2)
    psn = 10*math.log10(wh/sum)
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
    for i in comp:
        tmp += (i - m)**2

    #print(tmp)

    res = math.sqrt(1/((512**2)-1)*tmp)
    return res


def cor(comp1, comp2):
    ma = mato(comp1)
    mb = mato(comp2)
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
def main():
    rows = read_rows("asserts/lena512color.bmp")


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
    for i in range(0,105,10):
        greensmesh5 = math.fabs(cor(green, shift(i,10,green)))
       # redsmesh5 = math.fabs(cor(red, shift(i,10,red)))
       # bluesmesh5 = math.fabs(cor(blue, shift(i,10,blue)))

        m1.append(i)
        m2.append(greensmesh5)
        #m3.append(i)
        #m4.append(redsmesh5)
        #m5.append(i)
        #m6.append(bluesmesh5)
    plt.plot(m1, m2, label="l=%s"%('green',))
    for i in range(0, 105, 10):
         greensmesh5 = math.fabs(cor(green, shift(i,5,green)))
       # redsmesh5 = math.fabs(cor(red, shift(i,10,red)))
       # bluesmesh5 = math.fabs(cor(blue, shift(i,10,blue)))

        #m1.append(i)
        #m2.append(greensmesh5)
         m3.append(i)
         m4.append(greensmesh5)
        #m5.append(i)
        #m6.append(bluesmesh5)
    plt.plot(m3, m4, label="l=%s"%('green',))


    for i in range(0, 105, 10):
         greensmesh5 = math.fabs(cor(green, shift(i,-5,green)))
       # redsmesh5 = math.fabs(cor(red, shift(i,10,red)))
       # bluesmesh5 = math.fabs(cor(blue, shift(i,10,blue)))

        #m1.append(i)
        #m2.append(greensmesh5)
         #m3.append(i)
         #m4.append(greensmesh5)
         m5.append(i)
         m6.append(greensmesh5)
    plt.plot(m5, m6, label="l=%s"%('green',))
    m5.clear()
    m6.clear()
    for i in range(0, 105, 10):
         greensmesh5 = math.fabs(cor(green, shift(i,0,green)))

       # redsmesh5 = math.fabs(cor(red, shift(i,10,red)))
       # bluesmesh5 = math.fabs(cor(blue, shift(i,10,blue)))

        #m1.append(i)
        #m2.append(greensmesh5)
         #m3.append(i)
         #m4.append(greensmesh5)
         m5.append(i)
         m6.append(greensmesh5)
    plt.plot(m5, m6, label="l=%s"%('green',))
    #plt.plot(m3, m4, label="l=%s"%('red',))
    #plt.plot(m5, m6, label="l=%s"%('blue',))
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()
    print(greensmesh5,'GREENSNESH')


    #   plt.plot(10, greensmesh5)
    #plt.show()

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
    cbcomp = [int(0.5643*(blue[i]-ycomp[i])+128) for i in range(len(green))]
    crcomp = [int(0.7132*(red[i]-ycomp[i])+128) for i in range(len(green))]
    writeycbcr(ycomp, 'ycomp')
    writeycbcr(cbcomp, 'cbcomp')
    writeycbcr(crcomp, 'crcomp')
    corg = math.fabs(cor(ycomp, cbcomp))
    corb = math.fabs(cor(ycomp, crcomp))
    corr = math.fabs(cor(crcomp, cbcomp))
    corgg = math.fabs(cor(cbcomp, cbcomp))
    print(Fore.LIGHTGREEN_EX+'Cor YCB', corg)
    print('Cor YCR', corb)
    print('Cor CRCB', corr)

    grestored = [ycomp[i]-0.714*(crcomp[i]-128)-0.334*(cbcomp[i]-128)for i in range(len(ycomp))]
    rrestored = [ycomp[i] + 1.402*(crcomp[i]-128)for i in range(len(ycomp))]
    brestored = [ycomp[i]+1.772*(cbcomp[i]-128)for i in range(len(ycomp))]
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
    cbret = []
    cbretnorm = []
    cbdec = np.array(cbcomp)
    cbdec = cbdec.reshape((512, 512))
    for i, data in enumerate(cbdec):
        if i % 2 == 0:
            continue

        for j, data in enumerate(cbdec[i]):
            if j % 2 == 0:
                continue
            else:
                cbret.append(data)
                cbretnorm.append(data)
    print(len(cbret))

    crred = []
    crrednorm = []
    crdec = np.array(crcomp)
    crdec = crdec.reshape((512, 512))
    for i, data in enumerate(crdec):
        if i % 2 == 0:
            continue

        for j, datas in enumerate(crdec[i]):
            if j % 2 == 0:
                continue
            else:
                crred.append(datas)
                crrednorm.append(datas)
    print(len(crred))
    print(Fore.LIGHTGREEN_EX+'(A)PSNR cb restored', psnr(cbret, cbcomp))
    print('(A)PSNR cr restored', psnr(crred, crcomp), ''+Style.RESET_ALL)
    crred = writeycbcrrestored(crred, 'crredrest')
    cbret = writeycbcrrestored(cbret, 'cbretrest')
    gfycbcr = [ycomp[i] - 0.714 * (crred[i] - 128) - 0.334 * (cbret[i] - 128) for i in range(len(ycomp))]
    rfycbcr = [ycomp[i] + 1.402 * (crred[i] - 128) for i in range(len(ycomp))]
    bfycbcr = [ycomp[i] + 1.772 * (cbret[i] - 128) for i in range(len(ycomp))]
    gfycbcr = clipping(gfycbcr)
    rfycbcr = clipping(rfycbcr)
    bfycbcr = clipping(bfycbcr)
    print(Fore.LIGHTGREEN_EX+'(A)PSNR G', psnr(green, gfycbcr))
    print('(A)PSNR R', psnr(red, rfycbcr))
    print('(A)PSNR B', psnr(blue, bfycbcr), ''+Style.RESET_ALL)

    bcrret = []
    bcrretnorm = []

    for i in range(0,len(crdec),2):

        for j in range(0,len(crdec[i]),2):
            if i == 0 or j == 0 or i == len(crdec)-1 or j == len(crdec[i])-1:
                bcrret.append(crdec[i][j])
                bcrretnorm.append(crdec[i][j])


            else:

                sredn = (crdec[i][j+1]+crdec[i+1][j]+crdec[i][j-1]+crdec[i-1][j])/4
                bcrret.append(int(sredn))
                bcrretnorm.append(int(sredn))


    print(len(bcrret))

    bcbret = []
    bcbnorm = []

    for i in range(0, len(crdec), 2):

        for j in range(0, len(crdec[i]), 2):
            if i == 0 or j == 0 or i == len(crdec) - 1 or j == len(crdec[i]) - 1:
                bcbret.append(crdec[i][j])
                bcbnorm.append(crdec[i][j])

            else:

                sredn = (crdec[i][j + 1] + crdec[i + 1][j] + crdec[i][j - 1] + crdec[i - 1][j]) / 4
                bcbret.append(int(sredn))
                bcbnorm.append(int(sredn))

    print(len(bcbret))
    print(Fore.LIGHTGREEN_EX+'(B)PSNR cb restored', psnr(bcbret, cbcomp))
    print('(B)PSNR cr restored', psnr(bcrret, crcomp), ''+Style.RESET_ALL)
    bcrret = writeycbcrrestored(bcrret, 'bcrret')
    bcbret = writeycbcrrestored(bcbret, 'bcbret')
    gtrash = [ycomp[i] - 0.714 * (bcrret[i] - 128) - 0.334 * (bcbret[i] - 128) for i in range(len(ycomp))]
    rtrash = [ycomp[i] + 1.402 * (bcrret[i] - 128) for i in range(len(ycomp))]
    btrash = [ycomp[i] + 1.772 * (bcbret[i] - 128) for i in range(len(ycomp))]
    gtrash = clipping(gtrash)
    rtrash = clipping(rtrash)
    btrash = clipping(btrash)
    print(Fore.LIGHTGREEN_EX+'(B)PSNR G', psnr(green, gtrash))
    print('(B)PSNR R', psnr(red, rtrash))
    print('(B)PSNR B', psnr(blue, btrash), ''+Style.RESET_ALL)


    red = red[:int(len(red)/4)]
    print(len(red))
    green = green[:int(len(green) / 4)]
    print(len(green))
    blue = blue[:int(len(blue) / 4)]
    print(len(blue))
    crcomp = crcomp[:int(len(crcomp) / 4)]
    print(len(crcomp))
    cbcomp = cbcomp[:int(len(cbcomp) / 4)]
    print(len(cbcomp))
    ycomp = ycomp[:int(len(ycomp) / 4)]
    print(len(ycomp))

    print(Fore.LIGHTGREEN_EX + '(A)PSNR cb restored WH/4', psnr(cbretnorm, cbcomp))
    print('(A)PSNR cr restored WH/4', psnr(crrednorm, crcomp), '' + Style.RESET_ALL)
    crred = writeycbcrrestored(crrednorm, 'crredrest')
    cbret = writeycbcrrestored(cbretnorm, 'cbretrest')
    gfycbcr = [ycomp[i] - 0.714 * (crrednorm[i] - 128) - 0.334 * (cbretnorm[i] - 128) for i in range(len(ycomp))]
    rfycbcr = [ycomp[i] + 1.402 * (crrednorm[i] - 128) for i in range(len(ycomp))]
    bfycbcr = [ycomp[i] + 1.772 * (cbretnorm[i] - 128) for i in range(len(ycomp))]
    gfycbcr = clipping(gfycbcr)
    rfycbcr = clipping(rfycbcr)
    bfycbcr = clipping(bfycbcr)
    print(Fore.LIGHTGREEN_EX + '(A)PSNR G WH/4', psnr(green, gfycbcr))
    print('(A)PSNR R WH/4', psnr(red, rfycbcr))
    print('(A)PSNR B WH/4', psnr(blue, bfycbcr), '' + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX+'(B)PSNR cb restored WH/4', psnr(bcbnorm, cbcomp))
    print('(B)PSNR cr restored WH/4', psnr(bcrretnorm, crcomp), ''+Style.RESET_ALL)
    bcrret = writeycbcrrestored(bcrretnorm, 'bcrret')
    bcbret = writeycbcrrestored(bcbnorm, 'bcbret')
    gtrash = [ycomp[i] - 0.714 * (bcrretnorm[i] - 128) - 0.334 * (bcbnorm[i] - 128) for i in range(len(ycomp))]
    rtrash = [ycomp[i] + 1.402 * (bcrretnorm[i] - 128) for i in range(len(ycomp))]
    btrash = [ycomp[i] + 1.772 * (bcbnorm[i] - 128) for i in range(len(ycomp))]
    gtrash = clipping(gtrash)
    rtrash = clipping(rtrash)
    btrash = clipping(btrash)
    print(Fore.LIGHTGREEN_EX+'(B)PSNR G WH/4', psnr(green, gtrash))
    print('(B)PSNR R WH/4', psnr(red, rtrash))
    print('(B)PSNR B WH/4', psnr(blue, btrash), ''+Style.RESET_ALL)
if __name__ == '__main__':
    main()
