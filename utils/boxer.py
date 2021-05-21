import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import sys

plotname = "nflow_emd.png"

im = Image.open(plotname)

# Create figure and axes
fig, ax = plt.subplots()

#img = Image.open("filename.jpg")
img = im
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(r'fonts/agane_bold.ttf', 28)
draw.text((170, 520),"Electron",(0,0,0),font=font) # this will draw text with Blackcolor and 16 size
draw.text((380, 520),"Proton",(0,0,0),font=font) # this will draw text with Blackcolor and 16 size
draw.text((560, 520),"Photon 1",(0,0,0),font=font) # this will draw text with Blackcolor and 16 size
draw.text((760, 520),"Photon 2",(0,0,0),font=font) # this will draw text with Blackcolor and 16 size

image = im

ax.imshow(image)

# Create a Rectangle patch
rect = patches.Rectangle((140, 85), 170, 475, linewidth=1, edgecolor='blue', facecolor='blue',alpha=0.125)
ax.add_patch(rect)
#rect = patches.Rectangle((335, 85), 170, 475, linewidth=1, edgecolor='red', facecolor='red',alpha=0.125)
#ax.add_patch(rect)

a = 11.5
b = 189.5-85-a
b_wide = 390-335
c = 475-a-b
offset = 1.03125
offset_1 = 2
rect = patches.Rectangle((335, 85), 170, a, linewidth=1, edgecolor='red', facecolor='red',alpha=0.125)
ax.add_patch(rect)
rect = patches.Rectangle((335, 85+a+offset), b_wide, b-offset_1, linewidth=1, edgecolor='red', facecolor='red',alpha=0.125)
ax.add_patch(rect)
rect = patches.Rectangle((335, 85+a+b), 170, c, linewidth=1, edgecolor='red', facecolor='red',alpha=0.125)
ax.add_patch(rect)


a = 11.5
b = 189.5-85-a
b_wide = 635-530
c = 475-a-b
rect = patches.Rectangle((530, 85), 170, a, linewidth=1, edgecolor='green', facecolor='green',alpha=0.125)
ax.add_patch(rect)
rect = patches.Rectangle((530+b_wide, 85+a+offset), 170-b_wide, b-offset_1, linewidth=1, edgecolor='green', facecolor='green',alpha=0.125)
ax.add_patch(rect)
rect = patches.Rectangle((530, 85+a+b), 170, c, linewidth=1, edgecolor='green', facecolor='green',alpha=0.125)
ax.add_patch(rect)





#rect = patches.Rectangle((530, 85), 170, 475, linewidth=1, edgecolor='green', facecolor='green',alpha=0.125)
#ax.add_patch(rect)

rect = patches.Rectangle((725, 85), 170, 475, linewidth=1, edgecolor='darkgreen', facecolor='darkgreen',alpha=0.125)
ax.add_patch(rect)



#plt.show()
plt.savefig("nflow_emd_with_text.png")

