Description: Implements two learning algorithms -- decision trees and Adaboost
using decision trees to classify text as one of two languages - English or Dutch. 
First, train the data using the train.dat file and save the model via serialization. 
Input: train &lt;examples&gt; &lt;hypothesisOut&gt; &lt;learning-type&gt;
Then, load the previous model and predict using the test.dat file. 
Input: predict &lt;hypothesis&gt; &lt;file&gt;

Output: For each example, the predict program prints its predicted label on a newline,
either 'nl' for Dutch or 'en' for English.
