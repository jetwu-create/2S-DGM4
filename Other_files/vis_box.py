import cv2

    
# image_path = "/data1/wjj/jetwu/DGM4/manipulation/HFGI/114959-HFGI.jpg"
image_path = "/data/jetwu/code/CoF_FVG_final/temp/1669321-024135-simswap.jpg"
# if image_index == "DGM4/manipulation/simswap/1669321-024135-simswap.jpg" :
# image_path = os.path.join(image_sour, image_index)
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
# result = detector.detect_faces(image)
    # print(item["image"])
    
# for i,item in enumerate(result):
    # if i == 2:
bounding_box = [213, 23, 276, 120]
cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (24,226,72), 4)
# print([bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]])
cv2.imwrite("coling.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # print([bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]])
# print()
    # pdb.set_trace()
    
    