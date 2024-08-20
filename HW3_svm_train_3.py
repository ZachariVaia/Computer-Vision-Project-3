import cv2 as cv
import numpy as np

train_folders =['caltech/imagedb/145.motorbikes-101','caltech/imagedb/178.school-bus','caltech/imagedb/224.touring-bike','caltech/imagedb/251.airplanes-101','caltech/imagedb/252.car-side-101']

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

bow_descs = np.load('index.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# Train SVM
print('Training SVM...')
#motorbikes
svm1 = cv.ml.SVM_create()
svm1.setType(cv.ml.SVM_C_SVC)
svm1.setKernel(cv.ml.SVM_RBF)
svm1.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

labels1 = []
for p in img_paths:
    if '145.motorbikes-101' in p:
        labels1.append(1)
    elif '178.school-bus' in p:
        labels1.append(0)
    elif '224.touring-bike' in p:
        labels1.append(0)
    elif '251.airplanes-101' in p:
        labels1.append(0)
    elif '252.car-side-101' in p:
        labels1.append(0)

#
labels1 = np.array(labels1, np.int32)

svm1.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels1)
svm1.save('svm1')
#schoolbus
svm2 = cv.ml.SVM_create()
svm2.setType(cv.ml.SVM_C_SVC)
svm2.setKernel(cv.ml.SVM_RBF)
svm2.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
labels2 = []
for p in img_paths:
    if '145.motorbikes-101' in p:
        labels2.append(0)
    elif '178.school-bus' in p:
        labels2.append(1)
    elif '224.touring-bike' in p:
        labels2.append(0)
    elif '251.airplanes-101' in p:
        labels2.append(0)
    elif '252.car-side-101' in p:
        labels2.append(0)


labels2 = np.array(labels2, np.int32)


svm2.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels2)
svm2.save('svm2')

#turing bike
svm3 = cv.ml.SVM_create()
svm3.setType(cv.ml.SVM_C_SVC)
svm3.setKernel(cv.ml.SVM_RBF)
svm3.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
labels3 = []
for p in img_paths:
    if '145.motorbikes-101' in p:
        labels3.append(0)
    elif '178.school-bus' in p:
        labels3.append(0)
    elif '224.touring-bike' in p:
        labels3.append(1)
    elif '251.airplanes-101' in p:
        labels3.append(0)
    elif '252.car-side-101' in p:
        labels3.append(0)


labels3 = np.array(labels3, np.int32)
# labels3 = np.array(labels3, np.float32)

svm3.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels3)
svm3.save('svm3')

#airplaines
svm4 = cv.ml.SVM_create()
svm4.setType(cv.ml.SVM_C_SVC)
svm4.setKernel(cv.ml.SVM_RBF)
svm4.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
labels4 = []
for p in img_paths:
    if '145.motorbikes-101' in p:
        labels4.append(0)
    elif '178.school-bus' in p:
        labels4.append(0)
    elif '224.touring-bike' in p:
        labels4.append(0)
    elif '251.airplanes-101' in p:
        labels4.append(1)
    elif '252.car-side-101' in p:
        labels4.append(0)


labels4 = np.array(labels4, np.int32)


svm4.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels4)
svm4.save('svm4')

#car side
svm5 = cv.ml.SVM_create()
svm5.setType(cv.ml.SVM_C_SVC)
svm5.setKernel(cv.ml.SVM_RBF)
svm5.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
labels5 = []
for p in img_paths:
    if '145.motorbikes-101' in p:
        labels5.append(0)
    elif '178.school-bus' in p:
        labels5.append(0)
    elif '224.touring-bike' in p:
        labels5.append(0)
    elif '251.airplanes-101' in p:
        labels5.append(0)
    elif '252.car-side-101' in p:
        labels5.append(1)


labels5 = np.array(labels5, np.int32)


svm5.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels5)
svm5.save('svm5')