import cv2

# img = 'D:/anaconda3/JupyterNotebookFile/images/dogs_and_cats.jpg'
img = "/data/jetwu/code/CoF_FVG_final/temp/ivan_drawn_1.jpg"
img = cv2.imread(img)
# cv2.imshow('original', img)

# 选择ROI
# roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
# x, y, w, h = roi

# print(roi)
roi = 213, 23, 276, 120
x, y, w, h = roi
# 显示ROI并保存图片
if roi != (0, 0, 0, 0):
    crop = img[y:h, x:w]
    # cv2.imwrite('D:/anaconda3/JupyterNotebookFile/images/dogs_and_cats_crop.jpg', crop)
    cv2.imwrite("coling_crop_1.jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print('Saved!')