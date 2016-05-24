'''
Created on Mar 6, 2015

@author: niko
'''

import numpy as np

from common import *

def initPredictionModel(bpf, modelFile=None, pretrainedFile=None):
    a = caffe.io.caffe_pb2.BlobProto();
    binaryProtoFile = open(bpf, 'rb')
    data = binaryProtoFile.read()
    a.ParseFromString(data)
    means=a.data
    means=np.asarray(means)
    means=means.reshape(3,256,256)
    
    # Set the right path to your model definition binaryProtoFile, pretrained model weights,
    # and the image you would like to classify.
    
    caffe.set_phase_test()
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    if modelFile is None:
        modelFile = MODEL_FILE
    if pretrainedFile is None:
        pretrainedFile = PRETRAINED
    net = caffe.Classifier(modelFile, pretrainedFile,
                           mean = means,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    
    return net

def getPrediction(model, fileName, getProbabilities=True):
    input_image = caffe.io.load_image(fileName)
    prediction = model.predict([input_image])
    if getProbabilities:
        return prediction[0]
    predictedClass = prediction[0].argmax()
    return predictedClass

def getActualValues(testFile):
    dict = {}
    itemsList = getTextEntriesFromFile(testFile)
    for item in itemsList:
        name, lbl = item.split(" ")
        n = name[:-5]
        dict[n] = int(lbl)
    return dict    
    
if __name__ == "__main__":
    makePredictionsForSubmission = True
    configs = ['run-normal']
    modelDefinitions = ['/home/niko/caffe-models/diabetic-retinopathy-detection/lenet_small_kernels.prototxt']
    pretrainedModels = ['/home/niko/caffe-models/diabetic-retinopathy-detection/snapshot/run-normal_compare_strides/lenet_pca_small_kernels_iter_210000.caffemodel']
    for conf in configs:
        selectedFolder, sourceImagesFolderTrain, sourceImagesFolderTest, dataImagesTrain, dataImagesTest, dataImagesTestAugmented, trainLabelsFile, testLabelsFile, binaryProtoFile = getPathsForConfig(conf)
        for i in range(len(modelDefinitions)):
    
            #mdl = initPredictionModel(binaryProtoFile)
            mdl = initPredictionModel(binaryProtoFile, modelDefinitions[i], pretrainedModels[i])
            k = modelDefinitions[i].rfind("/")
            mdlName = modelDefinitions[i][k+1:-9]
            resultsFile = DATA_PATH + '/submission_' + mdlName + '.csv'
        
            if makePredictionsForSubmission:
                imagesList = getTextEntriesFromFile(SAMPLE_SUBMISSION_FILE)
                items = imagesList[1:]
                imagesCount = len(items)
                imagesProcessedCount = 0
                start_time = time.time()
                time_previous_iter = start_time
                processedItemsCount = 0
                processedItems = []
                try:
                    processedItemsCount, rows = getItemsFromFile(filename=resultsFile, excludeHeader=True)
                    for item in rows:
                        name, lbl = item.split(",")
                        processedItems.append(name)
                    imagesCount -= processedItemsCount
                except:
                    storeItem("image,level", resultsFile)
                    pass
                for item in items:
                    itemLabel, itemClass = item.replace(",", " ").split()
                    if itemLabel in processedItems:
                        continue
                    itemFilename = "%s.jpeg" % itemLabel
                    imageSourceFilename = "%s/%s" %(sourceImagesFolderTest, itemFilename)
                    predictedClass = getPrediction(mdl, imageSourceFilename, getProbabilities=False)
                    row = "%s,%d" % (itemLabel, predictedClass)
                    storeItem(row, resultsFile)
                    imagesProcessedCount += 1        
                    if imagesProcessedCount % 50 == 0:
                        elapsed_time = time.time() - start_time
                        elapsed_time_previous_iter = time.time() - time_previous_iter
                        time_previous_iter = time.time()
                        secondsPerImage = elapsed_time_previous_iter/50
                        itemsRemaining = imagesCount - imagesProcessedCount
                        secondsRemaining = secondsPerImage * itemsRemaining
                        m, s = divmod(secondsRemaining, 60)
                        h, m = divmod(m, 60)
                        timeRemaining = "%d:%02d:%02d" % (h, m, s)
                        print "Processed %d of %d items. Execution time: %.3f s (%f s/image; estimated remaining time: %s)" % (imagesProcessedCount, imagesCount, elapsed_time, secondsPerImage, timeRemaining)
            
            else:
                imagesProcessedCount = 0
                processedItems = []
                vals = getActualValues(testLabelsFile)
                imagesCount = len(vals.keys())
                k = modelDefinitions[i].rfind("/")
                mdlName = modelDefinitions[i][k+1:-9]
                resultsValidationFile = VALIDATION_PATH + '/' + mdlName + '.txt'
                try:
                    processedItemsCount, rows = getItemsFromFile(filename=resultsValidationFile, excludeHeader=True)
                    for item in rows:
                        name, actualLabel, predictedLabel = item.split(" ")
                        processedItems.append(name)
                    imagesCount -= processedItemsCount
                except:
                    pass
                start_time = time.time()
                for key, value in vals.iteritems():
                    if key in processedItems:
                        continue
                    itemFilename = "%s.jpeg" % key
                    imageSourceFilename = "%s/%s" %(dataImagesTest, itemFilename)
                    prediction = getPrediction(mdl, imageSourceFilename, getProbabilities=True)
                    p0, p1, p2, p3, p4 = prediction
                    predictedClass = prediction.argmax()
                    row = "%s %d %d %f %f %f %f %f" % (key, value, predictedClass, p0, p1, p2, p3, p4)
                    storeItem(row, resultsValidationFile)
                    imagesProcessedCount += 1        
                    if imagesProcessedCount % 50 == 0:
                        elapsed_time = time.time() - start_time
                        elapsed_time_previous_iter = time.time() - time_previous_iter
                        time_previous_iter = time.time()
                        secondsPerImage = elapsed_time_previous_iter/50
                        itemsRemaining = imagesCount - imagesProcessedCount
                        secondsRemaining = secondsPerImage * itemsRemaining
                        m, s = divmod(secondsRemaining, 60)
                        h, m = divmod(m, 60)
                        timeRemaining = "%d:%02d:%02d" % (h, m, s)
                        print "Processed %d of %d items. Execution time: %.3f s (%f s/image; estimated remaining time: %s)" % (imagesProcessedCount, imagesCount, elapsed_time, secondsPerImage, timeRemaining)
                
                
    
    