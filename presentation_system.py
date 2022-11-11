import mediapipe as mp
import cv2
import face_recognition
import pickle
import os
import datetime
import csv

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def present_operation(image):
    today = datetime.datetime.now()
    # Create today present folder.
    store_present = f'Present/{str(today.today())[:10]}'
    if not (os.path.exists(store_present)):
        os.mkdir(store_present)
    # Get the correct folder for perform operation
    room = 311
    day_name = today.ctime()[:3]
    time_now = str(today.time())[:5]
    time_now = int(time_now[:2]) * 60 + int(time_now[3:])
    subjectANDtime = os.listdir(f'311/{day_name}')
    working_dir = ''
    got_working_folder = False
    for folder in subjectANDtime:
        tem = folder[:11]
        print(folder)
        last_class_time = tem[6:].split('.')
        start_class_time = tem[:5].split('.')
        start_class_time = int(start_class_time[0]) * 60 + int(start_class_time[1])
        last_class_time = int(last_class_time[0]) * 60 + int(last_class_time[1])
        if start_class_time <= time_now <= last_class_time - 5:
            working_dir = folder
            got_working_folder = True
    if got_working_folder == False:
        print('Time is not matched\nThere have no class in this time')
        return 'Have no class'
    # Get the name of subject and subject code and store as csv file name.
    information_about_sub = working_dir.split('_')
    sub_name = information_about_sub[1]
    sub_code = information_about_sub[2]
    depermant_name = information_about_sub[3]
    semester_number = information_about_sub[4]
    shift_number = information_about_sub[5]
    group_name = information_about_sub[6]
    # load the list of encoding, names and roll.
    working_dir = f'{room}/{day_name}/{working_dir}'

    # Load CSV present file. Get whoise are taken already present. if not found create one.
    present_already_taken = []
    csv_file_name = f'{sub_name}_{sub_code}_{depermant_name}_{semester_number}_{shift_number}_{group_name}'
    if not(os.path.isfile(f'Present/{str(today.today())[:10]}/{csv_file_name}.csv')):
        _ = open(f'Present/{str(today.today())[:10]}/{csv_file_name}.csv', 'w')
        _.close()
    file_csv_list = open(f'Present/{str(today.today())[:10]}/{csv_file_name}.csv', 'r')
    csv_list = csv.reader(file_csv_list)
    for row in list(csv_list):
        present_already_taken.append(int(row[1]))

    # Load all encode, roll, names file
    file_encode = open(f'{working_dir}/encode.pickle', 'rb')
    encode = pickle.load(file_encode)
    file_encode.close()
    file_encode = open(f'{working_dir}/roll.pickle', 'rb')
    roll = pickle.load(file_encode)
    file_encode.close()
    file_encode = open(f'{working_dir}/names.pickle', 'rb')
    names = pickle.load(file_encode)
    file_encode.close()

    is_present_taken = False
    encoding_all = []
    for enc in encode:
        encoding_all.append(enc[0])
    
    image_encode = face_recognition.face_encodings(image)
    for encode in image_encode:
        compare_result = face_recognition.compare_faces(encoding_all, encode, 0.5)
    try:
      idx_student = compare_result.index(True)
    except:
      print('I am returning')
      return

    print(idx_student)
    result_true = compare_result.count(True)

    am_i_confused = 'sure'

    if(result_true >1):
        final_task = face_recognition.face_distance(encoding_all, image_encode[0])
        idx_student = list(final_task).index(min(final_task))
        print('I am little bit confused')
        am_i_confused = 'surity 60%'
    student_roll = roll[idx_student]

    if int(student_roll) not in present_already_taken:
        present_already_taken.append(int(student_roll))
        student_name = names[idx_student]
        print(student_name, student_roll, str(today.time())[:5])
        file = open(f'Present/{str(today.today())[:10]}/{csv_file_name}.csv', 'a', newline='')
        writer = csv.writer(file)
        writer.writerow([names[idx_student], roll[idx_student], depermant_name, semester_number, shift_number, group_name,str(today.time())[:5], am_i_confused])
        file.close()
        is_present_taken = True

        # Store the photos of all students those are present in the class
        if not(os.path.exists(f'present_photo/{str(today.today())[:10]}/')):
            os.mkdir(f'present_photo/{str(today.today())[:10]}/')
            print('I am here')
        cv2.imwrite(f'present_photo/{str(today.today())[:10]}/{names[idx_student]}_{roll[idx_student]}_{depermant_name}_{semester_number}_{shift_number}_{group_name}_{str(today.time())[:5]}_{am_i_confused}.jpg', image)

        return f'{roll[idx_student]}'
        
    if int(student_roll) in present_already_taken:
        print('present already taken')
        is_present_taken = True
        return f'{roll[idx_student]}'

    if is_present_taken == False:
        print("Oops... No match found")
    return 'No match'



# Let's capture all frame and do all opration
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success: continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        face_data = detection.location_data.relative_bounding_box
        img_shape = image.shape
        xmin = int(face_data.xmin * img_shape[1])
        ymin = int(face_data.ymin * img_shape[0])
        xmax = int(xmin + face_data.width * img_shape[1])
        ymax = int(ymin + face_data.height * img_shape[0])
        xmin, ymin, xmax, ymax = xmin -10, ymin - 60, xmax + 20, ymax + 20
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > img_shape[0]: xmax = img_shape[1] - 1
        if ymax > img_shape[1]: ymax = img_shape[0] - 1
        croped_image = image[ymin:ymax,xmin:xmax]
        result = present_operation(croped_image)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.rectangle(image, (xmin, ymin-30),(xmax, ymin), (0, 255, 0),  cv2.FILLED)
        cv2.putText(image, result, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    cv2.imshow('Present Automation', image)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
cap.release()
cv2.destroyAllWindows()