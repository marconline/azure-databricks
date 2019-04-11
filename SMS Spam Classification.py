# Databricks notebook source
data = spark.read.format('csv').options(delimiter='\t', inferSchema='true').load('/FileStore/tables/SMSSpamCollection.csv')
data.printSchema()

# COMMAND ----------

data = data.withColumnRenamed("_c0", "ham_spam").withColumnRenamed("_c1", "text")
data.printSchema()

# COMMAND ----------

#transforming string feature "ham_spam" into a "label" one
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="ham_spam", outputCol="label")
data = stringIndexer.fit(data).transform(data)
data.printSchema()

# COMMAND ----------

from pyspark.sql.functions import count
totalCount = data.count()
print("Total records: {}".format(totalCount))
data.groupBy("ham_spam").count().show()

# COMMAND ----------

display(data.filter("label == 1"))

# COMMAND ----------

import re
def removePunctuation(text):
  text=re.sub("[^0-9a-zA-Z ]", "", text)
  return text

removePunctuationUDF = udf(removePunctuation)
data = data.withColumn("cleantext", removePunctuationUDF(data.text))
data.printSchema()
display(data)

# COMMAND ----------

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF

# COMMAND ----------

tokenizer = Tokenizer(inputCol="cleantext", outputCol="words")
tokenizedDF = tokenizer.transform(data)
tokenizedDF.printSchema()
print(tokenizedDF.first())
print(len(tokenizedDF.first().words))

# COMMAND ----------

add_stopwords = StopWordsRemover.loadDefaultStopWords('english') 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
stoppedDF = stopwordsRemover.transform(tokenizedDF)
stoppedDF.printSchema()
print(stoppedDF.first())
print(len(stoppedDF.first().filtered))

# COMMAND ----------

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
featurizedData = hashingTF.transform(stoppedDF)
display(featurizedData.take(20))

# COMMAND ----------

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
display(rescaledData.select("label", "features").take(20))

# COMMAND ----------

#split data into training and test
split_data = rescaledData.randomSplit([0.8, 0.2])
training_data = split_data[0]
test_data = split_data[1]
print("Training data: ", training_data.count())
print("Test data: ", test_data.count())

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol='label', regParam=0.1)
lrModel = lr.fit(training_data)
resultmodel = lrModel.transform(training_data)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

resultmodel.select("text", "label", "probability", "prediction").show(1000)

# COMMAND ----------

def displayResults(resultDF):
  totalCount = resultDF.count()
  tp = resultDF.where("prediction == 1 and label == 1").count()
  tn = resultDF.where("prediction == 0 and label == 0").count()
  fp = resultDF.where("prediction == 1 and label == 0").count()
  fn = resultDF.where("prediction == 0 and label == 1").count()
  acc = (tp + tn) / totalCount
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  fpr = fp / (fp + tn)

  print("True negatives: ", tn)
  print("True positives: ", tp)
  print("False positives: ", fp)
  print("False negatives: ", fn)


  print("Accuracy: {}%".format(round(acc * 100, 2)))
  print("Precision: {}%".format(round(precision * 100, 2)))
  print("Recall: {}%".format(round(recall * 100, 2)))
  print("F1: {}%".format(round(200*precision*recall/(precision+recall), 2)))

  print("FPR: {}%".format(round(fpr * 100, 2)))

# COMMAND ----------

displayResults(resultmodel)

# COMMAND ----------

predictedTestDF = lrModel.transform(test_data)
predictedTestDF.printSchema()
displayResults(predictedTestDF)

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -rf /tmp/mleap_model
# MAGIC mkdir /tmp/mleap_model

# COMMAND ----------

import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer
lrModel.serializeToBundle("jar:file:/tmp/mleap_model/SMSSpamFilterModel.zip", resultmodel)
dbutils.fs.cp("file:/tmp/mleap_model/SMSSpamFilterModel.zip", "/FileStore/mleap/SMSSpamFilterModel.zip")
