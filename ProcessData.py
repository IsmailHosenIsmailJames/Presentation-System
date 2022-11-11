import pickle
from face_recognition import face_encodings
from cv2 import imread
from  cv2 import resize
from cv2 import imshow
from cv2 import waitKey
import os

list_of_shiftAnDdeperment = os.listdir('photos/')
list_of_face_encoding = []
list_of_roll = []
list_of_name = []
unsuccessfull = []
# Make faceData folder
if not(os.path.isdir(f'faceData/')):
    os.mkdir(f'faceData/')
# Create face database
for i in list_of_shiftAnDdeperment:
    if len(i.split('_')) != 4:
        print('Warning : \n', i, ' is not in correct formet.')
        print('I am unable to do process in ', i)
        continue
    if not(os.path.isdir(f'faceData/{i}')):
        os.mkdir(f'faceData/{i}')
    info = i.split('_')
    deperment = info[0]
    semester = info[1]
    shift = info[2]
    group = info[3]
    list_of_img_name = os.listdir(f'photos/{i}')
    for img_name in list_of_img_name:
        if len(img_name.split('_')) !=2:
            print('Warning : \n', img_name, ' is not in correct formet.')
            print('I am unable to do process in ', img_name)
            continue
        info = img_name.split('_')
        name = info[0]
        roll = info[1].split('.')[0]
        img = imread(f'photos/{i}/{img_name}')
        if img.shape[0] > 2000:
            img = resize(img, (0, 0), fx = 0.25, fy = 0.25)
        elif img.shape[0] > 1000:
            img = resize(img, (0, 0), fx = 0.5, fy = 0.5)
        encoding = face_encodings(img)
        if len(encoding) == 0:
            unsuccessfull.append(img_name)
            unsuccess_img = img
        list_of_face_encoding.append(encoding)
        list_of_name.append(name)
        list_of_roll.append(roll)
    file = open(f'faceData/{i}/encode.pickle','wb+')
    pickle.dump(list_of_face_encoding, file = file)
    file.close()
    file = open(f'faceData/{i}/names.pickle', 'wb+')
    pickle.dump(list_of_name, file = file)
    file.close()
    file = open(f'faceData/{i}/roll.pickle', 'wb+')
    pickle.dump(list_of_roll, file = file)
    file.close()
    list_of_face_encoding=[]
    list_of_name=[]
    list_of_roll=[]
    if len(unsuccessfull) != 0:
        for unsuccess in unsuccessfull:
            print('Warning : \n', unsuccess , " is Unsuccessful in encoding process. \nIt's because image is maybe not good or have to0 much noise.")
            imshow(unsuccess, unsuccess_img)
            waitKey(2000)
            continue
    print('Successfull, No error Found in : ', i, ' Total Student is :' , len(list_of_img_name))
    unsuccessfull=[]
print('\nFor find all face data cheak the folder : faceData\n')