import os
import cv2 as cv
import numpy as np

def testing(img_input):
    sift = cv.xfeatures2d_SIFT.create()

    vocabulary = np.load('vocabulary.npy')

    # Load SVM
    svm1 = cv.ml.SVM_create()
    svm1 = svm1.load('svm1')

    svm2 = cv.ml.SVM_create()
    svm2 = svm2.load('svm2')

    svm3 = cv.ml.SVM_create()
    svm3 = svm3.load('svm3')

    svm4 = cv.ml.SVM_create()
    svm4 = svm4.load('svm4')

    svm5 = cv.ml.SVM_create()
    svm5 = svm5.load('svm5')

    # Classification
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    test_img = img_input

    img = cv.imread(test_img)
    kp = sift.detect(img)
    bow_desc = descriptor_extractor.compute(img, kp)


    response1 = svm1.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response2 = svm2.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response3 = svm3.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response4 = svm4.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
    response5 = svm5.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)

    cnt = 0
    min_dist = 0
    if response1[1] < 0:
        # print('It is a motorbike')
        cnt = 1

    if response2[1] < 0 and cnt == 1 :
        if abs(response2[1]) > abs(response1[1]):
          cnt = 2
          # print('It is a school bus')
        else:
            cnt = 1
            # print('It is a motorbike')
    elif response2[1] < 0:
        cnt = 2
        # print('It is a school bus')

    if response3[1] < 0 and cnt == 1:
        if abs(response3[1]) > abs(response1[1]):
          cnt = 3
          # print('It is a touring bike')
        else:
            cnt = 1
            # print('It is a motorbike')
    elif response3[1] < 0 and cnt == 2:
        if abs(response3[1]) > abs(response2[1]):
            cnt = 3
            # print('It is a touring bike')
        else:
            cnt = 2
        # print('It is a school bus')
    elif response3[1] < 0:
        cnt = 3
        # print('It is a touring bike')



    if response4[1] < 0 and cnt == 1:
        if abs(response4[1]) > abs(response1[1]):
            cnt = 4
            # print('It is an airplane')
        else:
            cnt = 1
            # print('It is a motorbike')
    elif response4[1] < 0 and cnt == 2:
        if abs(response4[1]) > abs(response2[1]):
            cnt = 4
            # print('It is an airplane')
        else:
            cnt = 2
        # print('It is a school bus')
    elif response4[1] < 0 and cnt == 3:
        if abs(response4[1]) > abs(response3[1]):
            cnt = 4
            # print('It is an airplane')
        else:
            cnt = 3
            # print('It is a touring bike')
    elif response4[1] < 0:
        cnt = 4
        # print('It is an airplane')

    if response5[1] < 0 and cnt == 1:
        if abs(response5[1]) > abs(response1[1]):
            cnt = 5
            # print('It is a car side')
        else:
            cnt = 1
            # print('It is a motorbike')
    elif response5[1] < 0 and cnt == 2:
        if abs(response5[1]) > abs(response2[1]):
            cnt = 5
                # print('It is a car side')
        else:
            cnt = 2
            # print('It is a school bus')
    elif response5[1] < 0 and cnt == 3:
        if abs(response5[1]) > abs(response3[1]):
            cnt = 5
            # print('It is a car side')
        else:
            cnt = 3
            # print('It is a touring bike')
    elif response5[1] < 0 and cnt == 4:
        if abs(response5[1]) > abs(response4[1]):
            cnt = 5
            # print('It is a car side')
        else:
            cnt = 4
            # print('It is an airplane')
    elif response5[1] < 0:
        cnt = 5

        # print('It is a car side')
        cnt = 5
        # print('It is a car side')

    if(response1[1] > 0 and response2[1] > 0 and response3[1] > 0 and response4[1] > 0 and response5[1] > 0):
        min_dist = min(response1[1],response2[1],response3[1],response4[1], response5[1])
        if min_dist == response1[1]:
            cnt = 1
        elif min_dist == response2[1]:
            cnt = 2
        elif min_dist == response3[1]:
            cnt = 3
        elif min_dist == response4[1]:
            cnt = 4
        else:
            cnt  = 5


    return(cnt)
test_folders =['caltech/imagedb_test/145.motorbikes-101','caltech/imagedb_test/178.school-bus','caltech/imagedb_test/224.touring-bike','caltech/imagedb_test/251.airplanes-101','caltech/imagedb_test/252.car-side-101']
success = 0
total_tries = 0

success_class_motorbikes = 0
success_class_school_bus = 0
success_class_touring_bike = 0
success_class_airplane = 0
success_class_car_side = 0

total_tries_of_motorbikes = 0
total_tries_of_school_bus = 0
total_tries_of_touring_bike = 0
total_tries_of_airplane = 0
total_tries_of_car_side = 0

for folder in test_folders:
    files = os.listdir(folder) 
    for file in files:
       test_img = folder + '/' + file
       cnt = testing(test_img)
       total_tries = total_tries + 1


       if folder == 'caltech/imagedb_test/145.motorbikes-101' and cnt == 1:
           success_class_motorbikes = success_class_motorbikes + 1
       elif folder == 'caltech/imagedb_test/178.school-bus' and cnt == 2:
           success_class_school_bus = success_class_school_bus + 1
       elif folder == 'caltech/imagedb_test/224.touring-bike' and cnt == 3:
           success_class_touring_bike = success_class_touring_bike + 1
       elif folder == 'caltech/imagedb_test/251.airplanes-101'and cnt == 4:
           success_class_airplane = success_class_airplane + 1
       elif folder == 'caltech/imagedb_test/252.car-side-101' and cnt == 5:
           success_class_car_side = success_class_car_side + 1


       if folder == 'caltech/imagedb_test/145.motorbikes-101':
            total_tries_of_motorbikes = total_tries_of_motorbikes + 1
       elif folder == 'caltech/imagedb_test/178.school-bus':
            total_tries_of_school_bus = total_tries_of_school_bus + 1
       elif folder == 'caltech/imagedb_test/224.touring-bike':
           total_tries_of_touring_bike = total_tries_of_touring_bike + 1
       elif folder == 'caltech/imagedb_test/251.airplanes-101':
           total_tries_of_airplane = total_tries_of_airplane + 1
       elif folder == 'caltech/imagedb_test/252.car-side-101':
           total_tries_of_car_side =total_tries_of_car_side + 1

success = success_class_motorbikes + success_class_school_bus + success_class_touring_bike + success_class_airplane + success_class_car_side
accuracy = (success / total_tries)* 100
accuracy_motobikes = (success_class_motorbikes / total_tries_of_motorbikes)*100
accuracy_school_bus = (success_class_school_bus / total_tries_of_school_bus)*100
accuracy_touring_bike = (success_class_touring_bike / total_tries_of_touring_bike)*100
accuracy_airplane = (success_class_airplane / total_tries_of_airplane)*100
accuracy_car_side = (success_class_car_side / total_tries_of_car_side)*100


print('Total_accuracy: ', accuracy,'%')
print('Accuracy_class_motorbikes: ', accuracy_motobikes,'%')
print('Accuracy_class_school_bus: ', accuracy_school_bus,'%')
print('Accuracy_class_touring_bike: ', accuracy_touring_bike,'%')
print('Accuracy_class_airplane: ', accuracy_airplane,'%')
print('Accuracy_class_car_side: ', accuracy_car_side,'%')
