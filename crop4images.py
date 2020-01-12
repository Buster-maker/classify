import os
from PIL import Image
from PIL import ImageEnhance
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path=os.path.abspath(os.path.join(path_name,dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.JPG'):
                print(full_path)
                image=Image.open(full_path)
                '''
                im1 = image.crop([1244,744, 1756, 1256])
                im2 = image.crop([988,488 ,1500 ,1000 ])
                im3 = image.crop([ 1500 , 488, 2012,1000 ])
                im4 = image.crop([ 988, 1000,1500, 1512 ])
                im5=image.crop([1500,1000,2012,1512])
                im1.save(full_path[:-4]+'_1.JPG')
                im2.save(full_path[:-4]+'_2.JPG')
                im3.save(full_path[:-4]+'_3.JPG')
                im4.save(full_path[:-4]+'_4.JPG')
                im5.save(full_path[:-4]+'_5.JPG')
                '''
                out1 = image.transpose(Image.ROTATE_180)  # 180°顺时针翻转
                out1.save(full_path[:-4]+'_5.JPG')
                out2 = image.transpose(Image.ROTATE_270)  # 270°顺时针翻转
                out2.save(full_path[:-4]+'_6.JPG')



                # enh_bri = ImageEnhance.Brightness(image)
                # brightness = 1.5
                # image_brightened = enh_bri.enhance(brightness)
                # image_brightened.save(full_path[:-4]+'_1.JPG')  # 保存
                # 色度增强
                '''
                enh_col = ImageEnhance.Color(image)
                color = 1.5
                image_colored = enh_col.enhance(color)
                image_colored.save(full_path[:-4] + '_2.JPG')
                # 对比度增强
                enh_con = ImageEnhance.Contrast(image)
                contrast = 1.5
                image_contrasted = enh_con.enhance(contrast)
                image_contrasted.save(full_path[:-4] + '_3.JPG')

                out = image.transpose(Image.FLIP_LEFT_RIGHT)
                out.save(full_path[:-4] + '_4.JPG')
                '''
                os.remove(full_path)
read_path(path_name=r'./IDADP-PRCV20191')
print('work done !!!')