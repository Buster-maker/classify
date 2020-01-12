from os import scandir
from PIL import Image
from PIL import ImageEnhance
import os
image_1="./IDADP-PRCV2019-training/1-600"
image_2="./IDADP-PRCV2019-training/2-600"
image_3="./IDADP-PRCV2019-training/3-600"
image_4="./IDADP-PRCV2019-training/4-400"
image_5="./IDADP-PRCV2019-training/5-400"
image_6="./IDADP-PRCV2019-training/6-400"
image="./image"
file_path=[]
for img in scandir(image_6):
    if img.name.endswith(".JPG") and img.is_file():
        file_path.append(img.path)

# for i in file_path:
image=Image.open('./a.jpg')
enh_bri = ImageEnhance.Brightness(image)
brightness = 1.5
image_brightened = enh_bri.enhance(brightness)
enh_col = ImageEnhance.Color(image) # 色度
color = 1.5
image_colored = enh_col.enhance(color)
# 对比度增强
enh_con = ImageEnhance.Contrast(image)
contrast = 1.5
image_contrasted = enh_con.enhance(contrast)
out = image.transpose(Image.FLIP_LEFT_RIGHT)
image_brightened.save(r'./au.jpg')
image_colored.save(r'./bu.jpg')
image_contrasted.save(r'./cu.jpg')
out.save(r'./du.jpg')


    # enh_sha = ImageEnhance.Sharpness(image) # 锐度增强
    # sharpness = 3.0
    # image_sharped = enh_sha.enhance(sharpness)

# out = image.transpose(Image.FLIP_LEFT_RIGHT)
# path1=(i[-12:-4] +"_1" +".JPG")
# path2 = (i[-12:-4] + "_2" + ".JPG")
# path3 = (i[-12:-4] + "_3" + ".JPG")
# path4 = (i[-12:-4] + "_4" + ".JPG")
# path5 = (i[-12:-4] + "_5" + ".JPG")
# image_brightened.save(r'./IDADP-PRCV2019sedu/6/'+path1)
# image_colored.save(r'./IDADP-PRCV2019sedu/6/'+path2)
# image_contrasted.save(r'./IDADP-PRCV2019sedu/6/'+path3)
# out.save(r'./IDADP-PRCV2019sedu/6/' + path4)













  # box_1=(1372,744,1628,1000)# 中
    # box_2=(1244,616,1500,872)# 左上
    # box_3=(1244,872,1500,1128)# 左下
    # box_4=(1500,616,1756,872)# 右上
    # box_5=(1500,872,1756,1128)# 右下
    # img_1=img.crop(box_1)
    # img_2=img.crop(box_2)
    # img_3=img.crop(box_3)
    # img_4=img.crop(box_4)
    # img_5=img.crop(box_5)