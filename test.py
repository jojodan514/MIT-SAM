from PIL import Image
import os
import numpy as np

# def convert_jpg_to_png(source_dir, target_dir):
#     try:
#         # 检查目标目录是否存在，如不存在则创建
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
            
#         # 遍历源目录中的所有文件
#         for filename in os.listdir(source_dir):
#             if filename.endswith(".jpg") or filename.endswith(".jpeg"):
#                 # 构建完整的源文件路径
#                 img_path = os.path.join(source_dir, filename)
#                 # 读取图像
#                 img = Image.open(img_path)
#                 # 构建目标文件路径，将后缀名改为.png
#                 target_img_path = os.path.join(target_dir, os.path.splitext(filename)[0] + ".png")
#                 # 保存图像为PNG格式
#                 img.save(target_img_path, "PNG")
#         print("所有JPG图像已成功转换为PNG格式并保存到目标文件夹。")

#     except Exception as e:
#         print(f"处理图像时发生错误：{e}")

# # 这里填写你的源目录和目标目录
# source_dir = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/COVID-SemiSeg/Dataset/TestingSet/LungInfection-Test/root/image'
# target_dir = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/COVID-SemiSeg/Dataset/TestingSet/LungInfection-Test/root/image_1'
# convert_jpg_to_png(source_dir, target_dir)


# def rotate_and_save_images(source_dir, target_dir):
#     try:
#         # 检查目标目录是否存在，如果不存在则创建
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
        
#         # 遍历源目录中的所有文件
#         for filename in os.listdir(source_dir):
#             if filename.endswith(".png"):
#                 # 构建完整的文件路径
#                 img_path = os.path.join(source_dir, filename)
#                 # 读取图像
#                 img = Image.open(img_path)
#                 # 将图像向右旋转90度
#                 rotated_img = img.rotate(270, expand=True)
#                 # 构建目标图像路径
#                 target_img_path = os.path.join(target_dir, filename)
#                 # 保存旋转后的图像
#                 rotated_img.save(target_img_path)
#         print("所有PNG图像已成功旋转并保存到目标文件夹。")

#     except Exception as e:
#         print(f"处理图像时发生错误：{e}")

# # 使用示例
# source_dir = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/ours_semiseg_figs'
# target_dir = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/ours_semiseg_figs_rotate'
# rotate_and_save_images(source_dir, target_dir)

# Define the source and destination folders
# source_folder = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/COVID-SemiSeg/Dataset/TestingSet/LungInfection-Test/root/image'
# destination_folder = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/languidemedseg/LanGuideMedSeg-MICCAI2023-main/COVID-SemiSeg/Dataset/TestingSet/LungInfection-Test/root/image_resize'

# # Ensure the destination folder exists, create it if it doesn't
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
# # Loop through all files in the source folder
# for filename in os.listdir(source_folder):
#     # Construct the full file path
#     file_path = os.path.join(source_folder, filename)
#     # Check if the file is an image (assuming JPEG for the demonstration)
#     if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
#         # Open the image
#         with Image.open(file_path) as img:
#             # Resize the image
#             img_resized = img.resize((512, 512))
#             # Construct the full path for the destination
#             destination_path = os.path.join(destination_folder, filename)
#             # Save the resized image to the destination folder
#             img_resized.save(destination_path)
# print("Image resizing completed.")


# 输入和输出文件夹路径
input_folder = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/QaTa-COV19/QaTa-COV19-v2/Test/GTs'
output_folder = '/home/biiteam/Storage-4T/YLF/LanGuideMedSeg/QaTa-COV19/QaTa-COV19-v2/Test/GTs_3'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有图像文件
image_files = os.listdir(input_folder)

# 循环处理每张图像
for img_file in image_files:
    # 读取单通道图像
    img_path = os.path.join(input_folder, img_file)
    single_channel_img = Image.open(img_path)
    
    # 将单通道图像转换为Numpy数组
    single_channel_array = np.array(single_channel_img)
    
    # 创建一个全零的数组，形状为（224，224，3），用于存储RGB图像
    rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 将单通道图像的值叠加到RGB图像的每个通道上
    for i in range(3):
        rgb_image[:, :, i] = single_channel_array  # 每个通道都使用同一个单通道图像
    
    # 创建PIL图像对象
    rgb_image_pil = Image.fromarray(rgb_image)
    
    # 构造输出文件名
    output_path = os.path.join(output_folder, img_file)
    
    # 保存RGB图像
    rgb_image_pil.save(output_path)
    
    print(f"Processed {img_file} and saved as {output_path}")

print("All images processed and saved.")

