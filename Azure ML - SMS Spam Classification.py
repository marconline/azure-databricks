# Databricks notebook source
from azureml.core import Workspace
auth = None

ws = Workspace.from_config(auth = auth)
print('Workspace name: ' + ws.name,  'Azure region: ' + ws.location, 'Subscription id: ' + ws.subscription_id, 'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

import re
def removePunctuation(text):
  text=re.sub("[^0-9a-zA-Z ]", "", text)
  return text
removePunctuationUDF = udf(removePunctuation)

data = spark.read.format('csv').options(delimiter='\t', inferSchema='true').load('/FileStore/tables/SMSSpamCollection.csv')
data = data.withColumnRenamed("_c0", "ham_spam").withColumnRenamed("_c1", "text")
data = data.withColumn("cleantext", removePunctuationUDF(data.text))

#split data into training and test
split_data = data.randomSplit([0.8, 0.2])
training_data = split_data[0]
test_data = split_data[1]
print("Training data: ", training_data.count())
print("Test data: ", test_data.count())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer, HashingTF, IDF
stringIndexer = StringIndexer(inputCol="ham_spam", outputCol="label")
tokenizer = Tokenizer(inputCol="cleantext", outputCol="words")
add_stopwords = StopWordsRemover.loadDefaultStopWords('english') 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# COMMAND ----------

import shutil
import os
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from azureml.core.run import Run
from azureml.core.experiment import Experiment

#defining some wariables
model_name = "smsspamclassif_runs.mml"
model_dbfs = os.path.join("/dbfs", model_name)

#these are the different regularization parameter values we are going to test
regs = [0.0001, 0.001, 0.01, 0.1] 

myexperiment = Experiment(ws, "SMS_Spam_Classifier")
main_run = myexperiment.start_logging()

for reg in regs:
    print("Regularization rate: {}".format(reg))
    with main_run.child_run("reg-" + str(reg)) as run:
      lr = LogisticRegression(featuresCol="features", labelCol='label', regParam=reg)
      pipe = Pipeline(stages=[stringIndexer, tokenizer, stopwordsRemover, hashingTF, idf, lr])
      model_p = pipe.fit(training_data)
      
      # make prediction on test_data
      pred = model_p.transform(test_data)
      
      bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
      au_roc = bce.setMetricName('areaUnderROC').evaluate(pred)
      au_prc = bce.setMetricName('areaUnderPR').evaluate(pred)
      totalCount = pred.count()
      tp = pred.where("prediction == 1 and label == 1").count()
      tn = pred.where("prediction == 0 and label == 0").count()
      fp = pred.where("prediction == 1 and label == 0").count()
      fn = pred.where("prediction == 0 and label == 1").count()
      acc = (tp + tn) / totalCount
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * precision*recall/(precision+recall)

      run.log("reg", reg)
      run.log("au_roc", au_roc)
      run.log("au_prc", au_prc)
      run.log("TN", tn)
      run.log("TP", tp)
      run.log("FP", fp)
      run.log("FN", fn)
      run.log("Accuracy", round(acc, 2))
      run.log("Precision", round(precision, 2))
      run.log("Recall", round(recall, 2))
      run.log("F1", round(f1, 2))
      
      run.log_list("columns", training_data.columns)

      # save model
      model_p.write().overwrite().save(model_name)
        
      # upload the serialized model into run history record
      mdl, ext = model_name.split(".")
      model_zip = mdl + ".zip"
      shutil.make_archive(mdl, 'zip', model_dbfs)
      run.upload_file("outputs/" + model_zip, model_zip)        

      # now delete the serialized model from local folder since it is already uploaded to run history 
      shutil.rmtree(model_dbfs)
      os.remove(model_zip)
        
# Run completed
main_run.complete()
print ("run id:", main_run.id)
      

# COMMAND ----------

import os

#get the best model, based on AU ROC metric
metrics = main_run.get_metrics(recursive=True)
best_run_id = max(metrics, key = lambda k: metrics[k]['au_roc'])
print("Best Run ID: {} (regularization parameter {}) - AU ROC: {}".format(best_run_id, metrics[best_run_id]['reg'], metrics[best_run_id]['au_roc']))

#Get the best run
child_runs = {}

for r in main_run.get_children():
    child_runs[r.id] = r
   
best_run = child_runs[best_run_id]
best_model_file_name = "best_model.zip"
best_run.download_file(name = 'outputs/' + model_zip, output_file_path = best_model_file_name)

#unzip the model to dbfs and load it.
if os.path.isfile(model_dbfs) or os.path.isdir(model_dbfs):
    shutil.rmtree(model_dbfs)
shutil.unpack_archive(best_model_file_name, model_dbfs)

model_name_dbfs = os.path.join("/dbfs", model_name)
model_local = "file:" + os.getcwd() + "/" + model_name
dbutils.fs.cp(model_name, model_local, True)

# COMMAND ----------

#Register the model
from azureml.core.model import Model
mymodel = Model.register(model_path = model_name,
                       model_name = model_name,
                       description = "SMS Spam Classifier",
                       workspace = ws)

print(mymodel.name, mymodel.description, mymodel.version)

# COMMAND ----------

#%%writefile score_sparkml.py
score_sparkml = """
 
import json
 
def init():
    # One-time initialization of PySpark and predictive model
    import pyspark
    from azureml.core.model import Model
    from pyspark.ml import PipelineModel
 
    global trainedModel
    global spark
 
    spark = pyspark.sql.SparkSession.builder.appName("SMS Spam Classifier").getOrCreate()
    model_name = "{model_name}" #interpolated
    model_path = Model.get_model_path(model_name)
    trainedModel = PipelineModel.load(model_path)
    
def run(input_json):
    if isinstance(trainedModel, Exception):
        return json.dumps({{"trainedModel":str(trainedModel)}})
      
    try:
        sc = spark.sparkContext
        input_list = json.loads(input_json)
        input_rdd = sc.parallelize(input_list)
        input_df = spark.read.json(input_rdd)
    
        # Compute prediction
        prediction = trainedModel.transform(input_df)
        #result = prediction.first().prediction
        predictions = prediction.collect()
 
        #Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        #result = ",".join(preds)
        # you can return any data type as long as it is JSON-serializable
        return preds
    except Exception as e:
        result = str(e)
        return result
    
""".format(model_name=model_name)
 
exec(score_sparkml)
 
with open("score_sparkml.py", "w") as file:
    file.write(score_sparkml)

# COMMAND ----------

from azureml.core.conda_dependencies import CondaDependencies 

myacienv = CondaDependencies.create(conda_packages=[]) #showing how to add libs as an eg. - not needed for this model.

with open("mydeployenv.yml","w") as f:
    f.write(myacienv.serialize_to_string())

# COMMAND ----------

#deploy to ACI
from azureml.core.webservice import AciWebservice, Webservice

myaci_config = AciWebservice.deploy_configuration(
    cpu_cores = 2, 
    memory_gb = 2, 
    tags = {'name':'Databricks Azure ML ACI'}, 
    description = 'SMS Spam Classifier')

# COMMAND ----------

service_name = "smsspam"
runtime = "spark-py" 
driver_file = "score_sparkml.py"
my_conda_file = "mydeployenv.yml"

# image creation
from azureml.core.image import ContainerImage
myimage_config = ContainerImage.image_configuration(execution_script = driver_file, 
                                    runtime = runtime, 
                                    conda_file = my_conda_file)

# COMMAND ----------

# Webservice creation
myservice = Webservice.deploy_from_model(
  workspace=ws, 
  name=service_name,
  deployment_config = myaci_config,
  models = [mymodel],
  image_config = myimage_config
    )

myservice.wait_for_deployment(show_output=True)

# COMMAND ----------

#for using the Web HTTP API 
print(myservice.scoring_uri)

# COMMAND ----------

json_ex = """[
{ \"cleantext\": \"Incredible! You won a 1 month FREE membership in our prize ruffle! Text us at 09061701461 to claim \" },
{ \"cleantext\": \"Hi darling, this is a good message and I think you will receive it. Love you, see you later!\" }]"""
#input_list = json.loads(json_ex)
#input_rdd = sc.parallelize(input_list)
#input_df = spark.read.json(input_rdd)

# COMMAND ----------

myservice.run(input_data=json_ex)

# COMMAND ----------

myservice.delete()
