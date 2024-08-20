import os
import cv2 as cv
import numpy as np


test_folders =['caltech/imagedb_test/145.motorbikes-101','caltech/imagedb_test/178.school-bus','caltech/imagedb_test/224.touring-bike','caltech/imagedb_test/251.airplanes-101','caltech/imagedb_test/252.car-side-101']


def most_frequent(List):
    return max(set(List), key=List.count)
def choosing_k(k):
    k=k
    def testing(img_input,k):
        sift = cv.xfeatures2d_SIFT.create()

        bow_descs = np.load('index.npy').astype(np.float32)

        img_paths = np.load('paths.npy')

        # TRAINING
        labels = []
        for p in img_paths:

            if '145.motorbikes-101' in p:
                labels.append(1)
            elif '178.school-bus' in p:
                labels.append(2)
            elif '224.touring-bike' in p:
                labels.append(3)
            elif '251.airplanes-101' in p:
                labels.append(4)
            elif '252.car-side-101' in p:
                labels.append(5)
            else:
                labels.append(0)

        labels = np.array(labels, np.int32)
        labels2 = labels.tolist()
        # function which finds the class

        def Classification(img,bow_desc,k):
            distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
            retrieved_ids = np.argsort(distances)  # shows the spot of the min dist from min to max
            # for id in retrieved_ids.tolist():
            #     result_img = cv.imread(img_paths[id])
            #     cv.imshow('results', result_img)
            #     cv.waitKey(0)
            # pass

            best_dist = np.zeros(k)
            labels_of_nearest_neighbors = []
            class_of_image =[]


            # def most_frequent(List):
            #     return max(set(List), key=List.count)

            for n in range(0,k):
                best_dist[n] = retrieved_ids[n]
                n = n + 1

            best = np.array(best_dist,np.int32)

            for n in range(0, k):
                labels_of_nearest_neighbors.append(labels2[best[n]])

            label_of_the_class_of_the_image = most_frequent(labels_of_nearest_neighbors)
            if label_of_the_class_of_the_image == 1:
                class_of_image = str("145.motorbikes-101")
            elif label_of_the_class_of_the_image == 2:
                class_of_image = str("178.school-bus")
            elif label_of_the_class_of_the_image == 3:
                class_of_image = str("224.touring-bike")
            elif label_of_the_class_of_the_image == 4:
                class_of_image = str("251.airplanes-101")
            elif label_of_the_class_of_the_image == 5:
                class_of_image = str("252.car-side-101")
            else:
                class_of_image = str("nothing")


            return(class_of_image)

        # # TESTING
        sift = cv.xfeatures2d_SIFT.create()

        vocabulary = np.load('vocabulary.npy')

        desc = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
        desc.setVocabulary(vocabulary)

        img = cv.imread(img_input)
        kp = sift.detect(img)
        bow_desc = desc.compute(img, kp)


        #class of the image
        class_of_image = Classification(img,bow_desc,k)

        if class_of_image == "145.motorbikes-101":
            results = 1
        elif class_of_image == "178.school-bus":
            results = 2
        elif class_of_image == "224.touring-bike":
            results = 3
        elif class_of_image == "251.airplanes-101":
            results = 4
        elif class_of_image == "252.car-side-101":
            results = 5
        else:
            results = 0

        cnt = 0
        if results == 1:
            # print('It is a motorbike!')
            cnt = 1
        elif results == 2:
            # print('It is a school bus!')
            cnt = 2
        elif results == 3:
            # print('It is a touring bike!')
            cnt = 3

        elif results == 4:
            # print('It is an airplane!')
             cnt = 4

        elif results == 5:
            # print('It is a car side!')
            cnt = 5

        else:
            # print('It is not a match')
             cnt = 0

        return(cnt)

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
           cnt  = testing(test_img,k)
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
    return (accuracy)

accuracy = 0
k = 0

for n in range(5, 61):
    if (n % 2 != 0):
        accuracy=choosing_k(n)
        print('Total_accuracy: ', accuracy, '%')
        print('k=',n)





