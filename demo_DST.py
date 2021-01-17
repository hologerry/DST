import matplotlib.pyplot as plt
import numpy as np
import torch
from utils_misc import pil_loader, pil_resize_long_edge_to, pil_to_tensor

# Image will be resized to have a long side of im_side
im_size = 256

# If you choose cuda, make sure you select GPU under [Runtime]-[Change runtime type]
device = 'cuda'

# Set input image paths
content_path = 'example/content.jpg'
style_path = 'example/style.jpg'

# Load and resize input images
content_pil = pil_resize_long_edge_to(pil_loader(content_path), int(im_size))
width, height = content_pil.size
style_pil = pil_resize_long_edge_to(pil_loader(style_path), int(im_size))
content_im_orig = pil_to_tensor(content_pil).to(device)
style_im_orig = pil_to_tensor(style_pil).to(device)


# Plot images
fig = plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(content_pil)
plt.axis('off')
plt.title('Content image')
plt.subplot(1, 2, 2)
plt.imshow(style_pil)
plt.axis('off')
plt.title('Style image')


# Set where you want NBB results to be saved
pts_path = 'example/NBBresults'

# Run NBB
'''
!python NBB/main.py --results_dir ${pts_path} --imageSize ${im_size} --fast \
  --datarootA ${content_path} --datarootB ${style_path}
'''

# Set paths and other parameters for cleaning points
content_pts_path = 'example/NBBresults/correspondence_A.txt'
style_pts_path = 'example/NBBresults/correspondence_B.txt'
activation_path = 'example/NBBresults/correspondence_activation.txt'
output_path = 'example/CleanedPts'
NBB = 1
max_num_points = 80
b = 10

'''
!python cleanpoints.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path} \
  ${activation_path} ${output_path} ${im_size} ${NBB} ${max_num_points} ${b}
'''

content_marked = 'example/CleanedPts/A_selected_final.png'
style_marked = 'example/CleanedPts/B_selected_final.png'


# Load and plot images with points marked
content_marked_pil = pil_resize_long_edge_to(pil_loader(content_marked), int(im_size))
style_marked_pil = pil_resize_long_edge_to(pil_loader(style_marked), int(im_size))

fig = plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(content_marked_pil)
plt.axis('off')
plt.title('Content image with points')
plt.subplot(1, 2, 2)
plt.imshow(style_marked_pil)
plt.axis('off')
plt.title('Style image with points')


content_pts_path = 'example/CleanedPts/correspondence_A.txt'
style_pts_path = 'example/CleanedPts/correspondence_B.txt'
output_dir = 'example/DSTresults'
output_prefix = 'example'
max_iter = 250
checkpoint_iter = 50
content_weight = 8
warp_weight = 0.5
reg_weight = 50
optim = 'sgd'
lr = 0.2
verbose = 0
save_intermediate = 0
save_extra = 0


'''
!python -W ignore main.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path} \
  ${output_dir} ${output_prefix} ${im_size} ${max_iter} \
  ${checkpoint_iter} ${content_weight} ${warp_weight} ${reg_weight} ${optim} \
  ${lr} ${verbose} ${save_intermediate} ${save_extra} ${device}
'''

result_path = 'example/DSTresults/example.png'


# Load the output image
result_pil = pil_resize_long_edge_to(pil_loader(result_path), int(im_size))


# Plot input and output images
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(content_pil)
plt.axis('off')
plt.title('Content image')
plt.subplot(1, 3, 2)
plt.imshow(style_pil)
plt.axis('off')
plt.title('Style image')
plt.subplot(1, 3, 3)
plt.imshow(result_pil)
plt.axis('off')
plt.title('DST output')
