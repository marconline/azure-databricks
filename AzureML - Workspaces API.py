# Databricks notebook source
import azureml.core

# Check core SDK version number - based on build number of preview/master.
print("SDK version:", azureml.core.VERSION)

subscription_id = "94dfecbc-80b8-4725-ba8b-85e0f9b98672"
resource_group = "DataBricks"
workspace_name = "amlws_test"
workspace_region = "westeurope"

auth = None

# COMMAND ----------

from azureml.core import Workspace
ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      auth = auth,
                      exist_ok=True)

# COMMAND ----------

ws.get_details()

# COMMAND ----------

ws.write_config()

# COMMAND ----------

# import the Workspace class and check the azureml SDK version
from azureml.core import Workspace

ws = Workspace.from_config(auth = auth)
#ws = Workspace.from_config(<full path>)
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

ws = Workspace.get("mlservice", subscription_id = subscription_id)
ws.delete(delete_dependent_resources=True)

# COMMAND ----------

help(Workspace)
