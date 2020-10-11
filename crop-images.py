from PIL import Image
import os.path

path = "/mnt/4tb/lsun_cat/imgs/"
path_cropped = "/mnt/4tb/lsun_cat/imgs_cropped/"
files = os.listdir(path)

num_cropped = 0
for file in files:
    full_path = os.path.join(path, file)
    if os.path.isfile(full_path):
        image = Image.open(full_path)
        width, height = image.size
        d, f = os.path.split(full_path)

        # resize so the smaller dimension is 256
        aspect_ratio = width / height
        if aspect_ratio < 1:
            new_size = (256, int(256 // aspect_ratio))
        else:
            new_size = (int(256 * aspect_ratio), 256)
        image = image.resize(new_size)

        width, height = new_size
        # crop to 256 x 256
        if aspect_ratio > 1:
            num_rows_to_crop = width - 256
            left = num_rows_to_crop // 2
            if width % 2 == 0:
                right = width - left
            else:
                right = width - left - 1
            crop_size = (left, 0, right, height)
        else:
            num_rows_to_crop = height - 256
            upper = num_rows_to_crop // 2
            if height % 2 == 0:
                lower = height - upper
            else:
                lower = height - upper - 1
            crop_size = (0, upper, width, lower)

        cropped_image = image.crop(crop_size)
        cropped_image.save(path_cropped + f, 'JPEG', quality=95)
        num_cropped += 1

    if num_cropped % 1000 == 0:
        print("num cropped: " + str(num_cropped))
