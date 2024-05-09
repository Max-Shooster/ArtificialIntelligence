"""
Name: Max Shooster
File: adaboost.py

Description: Implements two learning algorithms -- decision trees and Adaboost
using decision trees to classify text as one of two languages - English or Dutch. 
First, train the data using the train.dat file and save the model via serialization. 
Input: train <examples> <hypothesisOut> <learning-type>
Then, load the previous model and predict using the test.dat file. 
Input: predict <hypothesis> <file> 

Output: For each example, the predict program prints its predicted label on a newline,
either 'nl' for Dutch or 'en' for English.

"""

import numpy as np
import argparse
import os
import sys
import pickle
import re 
import math

# Constants for learning algorithms 
maxTreeDepth = 10 # best option by testing
adaboostNumStumps = 100 # best option by testing 

# Node in a decision tree
class TreeNode:
    def __init__(self, featureIdx=None, left=None, right=None, val=None):
        self.featureIdx = featureIdx
        self.left = left
        self.right = right
        self.val = val

    def isLeaf(self):
        """
        Check if it is a leaf
        """
        return self.val is not None

# decision tree
class DecisionTree:
    def __init__(self, maxDepth=6):
        self.maxDepth = maxDepth
        self.root = None

    def fit(self, features, labels, sampleWeight=None):
        """
        Fit model given the features and labels
        """
        if sampleWeight is None:
            sampleWeight = np.ones(len(labels))
        self.root = self.buildTree(features, labels, depth=0, weights=sampleWeight)

    def predict(self, features):
        """
        Predict labels based on the features
        """
        predictions = []
        for feature in features:
            prediction = self.walkTree(feature, self.root)
            predictions.append(prediction)
        return np.array(predictions)

    def buildTree(self, features, labels, depth, weights):
        """
        build the tree
        """
        numFeatures = features.shape[1]
        if depth >= self.maxDepth or np.unique(labels).size == 1:
            leafVal = self.mostCommonLabel(labels, weights)
            return TreeNode(val=leafVal)

        topFeature = None # to store feature with highest info gain
        topGain = None # # to store highest info gain
        for featureIdx in range(numFeatures):
            gain = self.informationGain(features, labels, featureIdx, weights)
            if topGain is None or gain > topGain:
                topGain = gain
                topFeature = featureIdx

        if topGain == 0 or topFeature is None:
            return TreeNode(val=self.mostCommonLabel(labels, weights))

        # idxs where feature is 0
        idxZero = np.where(features[:, topFeature] == 0)[0]
        # idxs where  feature is non 0
        idxNonZero = np.where(features[:, topFeature] != 0)[0]

        # Build subtrees
        subtreeL = self.buildTree(features[idxZero], labels[idxZero], depth+1, weights[idxZero])
        subtreeR = self.buildTree(features[idxNonZero], labels[idxNonZero], depth+1, weights[idxNonZero])

        return TreeNode(topFeature, subtreeL, subtreeR)

    def entropy(self, labels, weights):
        """
        Calculate entropy given labels and weights
        """
        labelCount = np.unique(labels)
        totalWeight = np.sum(weights)
        # Initialize weighted probabilities array
        weightedProbs = np.zeros(labelCount.shape)
        
        # Calculate weighted probability for each unique label
        for i in range(len(labelCount)):
            label = labelCount[i]
            label_weights = weights[labels == label]
            weightedProbs[i] = np.sum(label_weights) / totalWeight

        # Compute entropy
        epsilon = 0.00001
        return -np.sum(weightedProbs * np.log2(weightedProbs + epsilon))

    def remainder(self, features, labels, featureIdx, weights):
        """
        Calculate remainder
        """
        totalWeight = weights.sum()
        labelCount = np.unique(features[:, featureIdx])
        # Calculate entropy for each split
        entropySum = 0
        for val in labelCount:
            idx = features[:, featureIdx] == val
            entropy = self.entropy(labels[idx], weights[idx])
            entropySum += (weights[idx].sum() / totalWeight) * entropy
        return entropySum

    def informationGain(self, features, labels, featureIdx, weights):
        """
        Calculate information gain
        """
        # Initial entropy of the entire set
        entropyA = self.entropy(labels, weights)
        # Remainder after splitting on feature
        entropyB = self.remainder(features, labels, featureIdx, weights)
        # Gain is the reduction in entropy after split
        return entropyA - entropyB

    def mostCommonLabel(self, labels, weights):
        """
        find most common label
        """
        labelCount = np.unique(labels)
        weightedLabels = np.zeros(len(labelCount))
        for label in labelCount:
            idx = np.where(labelCount == label)[0][0]
            weightedLabels[idx] = weights[labels == label].sum()
        maxVal = weightedLabels.argmax()
        return labelCount[maxVal]

    def walkTree(self, feature, node):
        """
        walk the tree
        """
        if node.isLeaf():
            return node.val
        if feature[node.featureIdx] == 0:
            return self.walkTree(feature, node.left)
        return self.walkTree(feature, node.right)

# adaboost
class AdaBoost:
    """
    An ensemble of decision trees. Each tree is a decision stump which is a
    decision tree with a max depth of 1. 
    """
    def __init__(self, numStumps=6):
        self.numStumps = numStumps  # Number of stumps
        self.trees = [] # used to store weak classifiers
        self.treeWeights = [] # weights for each hypothesis

    def fit(self, features, labels):
        """
        Trains the program based on the train.dat file
        """
        numLines = features.shape[0]
        weights = np.full(numLines, (1 / numLines))  # Initialize sample weights
        for i in range(self.numStumps): # loop over the number of stumps
            stump = DecisionTree(maxDepth=1) # create a new decision stump
            stump.fit(features, labels, sampleWeight=weights)  # train stump with current weights
            predictions = stump.predict(features)
            # Calculate errors of stump
            mislabeled = []
            for i in range(len(labels)):
                if labels[i] != predictions[i]:
                    mislabeled.append(weights[i])
            mislabeled = np.array(mislabeled)
            err = sum(mislabeled)
            epsilon = 0.00001 # avoid division by 0
            err = min(max(err, epsilon), 1-epsilon)
            deltaW = err / (1-err)
            stumpWeight = 0.5 * np.log((1-err) / err) # set weight for hypothesis
            # Update weights
            for i in range(numLines):
                if labels[i] == predictions[i]:
                    weights[i] *= deltaW
            weights /= weights.sum() # normalize weights
            # Store the tree and its weight
            self.trees.append(stump)
            self.treeWeights.append(stumpWeight)

    # combines weak classifiers to produce the more accurate predictions
    def predict(self, features):
        """
        predicts labels based on features
        """
        numLines = features.shape[0]
        labelPrediction = np.zeros(numLines)
        # Collect predictions across all trees
        for i in range(len(self.trees)):
            currTree = self.trees[i] 
            currTreeWeight = self.treeWeights[i]
            currPredictions = currTree.predict(features)
            weightedPredictions = currTreeWeight * np.where(currPredictions == 1, 1, -1)
            labelPrediction += weightedPredictions

        predictions = np.sign(labelPrediction)
        return predictions

# Function to compute Boolean features from 15-word Block
def computeFeatures(wordBlock):
    """
    analyze features based on the test data
    """
    features = np.zeros(13, dtype=int)
    features[0] = 1 if re.search(r'\bthe\b', wordBlock, re.IGNORECASE) else 0  
    features[1] = 1 if re.search(r'\bde\b', wordBlock, re.IGNORECASE) else 0   
    features[2] = 1 if re.search(r'\band\b', wordBlock, re.IGNORECASE) else 0  
    features[3] = 1 if re.search(r'\ben\b', wordBlock, re.IGNORECASE) else 0   
    features[4] = 1 if re.search(r'\bhet\b', wordBlock, re.IGNORECASE) else 0  
    features[5] = 1 if re.search(r'\bit\b', wordBlock, re.IGNORECASE) else 0   
    # Count consecutive double letters within each word
    words = wordBlock.split()
    doubleLetterCount = 0
    for word in words:
        for i in range(len(word) - 1):
            if word[i] == word[i+1]:
                doubleLetterCount += 1
    features[6] = 1 if doubleLetterCount > 2 else 0
    # Feature based on average word length larger than 5
    sumChars = 0
    for word in words:
        sumChars += len(word)
    avgWordLength = sumChars / len(words)
    features[7] = 1 if avgWordLength > 5 else 0 
    # Check if 'e' appears more than 10 times
    features[8] = 1 if wordBlock.lower().count('e') > 10 else 0  
    # check for ing suffix, check for 'een', check for 'in', check for 'of'
    features[9] = 1 if re.search(r'ing\b', wordBlock, re.IGNORECASE) else 0
    features[10] = 1 if re.search(r'\been\b', wordBlock, re.IGNORECASE) else 0   
    features[11] = 1 if re.search(r'\bin\b', wordBlock, re.IGNORECASE) else 0    
    features[12] = 1 if re.search(r'\bof\b', wordBlock, re.IGNORECASE) else 0    
    return features

# Load training data
def loadTestData(filePath, isTraining=True):
    """
    load the test data
    """
    features = []
    labels = []
    with open(filePath, 'r') as f:
        for line in f:
            if isTraining: # if training, then gather labels and features
                label, wordBlock = line.strip().split("|", 1)
                features.append(computeFeatures(wordBlock))
                if label == 'en':   
                    labels.append(1) # 1 for English
                else:
                    labels.append(0) # 0 for Dutch
            else: # else, just compute features
                features.append(computeFeatures(line.strip()))
    featuresArr = np.array(features)
    if isTraining: # if training, then return labels and features
        labelsArr = np.array(labels)
        return featuresArr, labelsArr
    return featuresArr, None # else, just return features

# Serialize model
def saveModel(model, filePath):
    """
    save the model
    """
    with open(filePath, 'wb') as f:
        pickle.dump(model, f)
def loadModel(filePath):
    """
    load the model
    """
    with open(filePath, 'rb') as f:
        model = pickle.load(f)
    return model

def trainModel(examplesFilePath, hypothesisOut, learningType):
    """
    train the model
    """
    featureVals, labelVals = loadTestData(examplesFilePath)
    if learningType == 'dt':
        model = DecisionTree(maxDepth=maxTreeDepth)         
    elif learningType == 'ada':
        model = AdaBoost(numStumps=adaboostNumStumps)            
    model.fit(featureVals, labelVals)
    saveModel(model, hypothesisOut)

def predictModel(hypothesisFilePath, testFilePath):
    """
    predict the model
    """
    model = loadModel(hypothesisFilePath)
    featureVals, _ = loadTestData(testFilePath, isTraining=False) 
    predictions = model.predict(featureVals)
    for pred in predictions:          
        if pred == 1:
              print('en')
        else:
              print('nl')



def main():
    scanner = argparse.ArgumentParser()
    cmds = scanner.add_subparsers(dest='cmd')
    trainModelScanner = cmds.add_parser('train')
    trainModelScanner.add_argument('examples', type=str)
    trainModelScanner.add_argument('hypothesisOut', type=str)
    trainModelScanner.add_argument('learningType', type=str, choices=['dt', 'ada'])
    predictModelScanner = cmds.add_parser('predict')
    predictModelScanner.add_argument('hypothesis', type=str)
    predictModelScanner.add_argument('file', type=str)
    args = scanner.parse_args()
    if args.cmd == 'train':
        trainModel(args.examples, args.hypothesisOut, args.learningType)
    elif args.cmd == 'predict':
        predictModel(args.hypothesis, args.file) 

if __name__ == "__main__":
    main()


