# Databricks notebook source
import sentiment_analyzer
import pyspark
import pandas as pd
from pyspark.sql import SparkSession, functions as F
#import nltk
#nltk.download("all")
from pyspark.sql.types import *
from py4j.java_gateway import java_import 
java_import(spark._sc._jvm, "org.apache.spark.sql.api.python.*")
import json
from pyspark.sql.types import DecimalType
from datetime import datetime, timedelta
import datetime as dt

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# COMMAND ----------

# #import data on a rolling six months
# today = datetime.today()
# six_months = today - timedelta(days=6*30)
# df_dat = spark.read.table("comments_table_here").where(F.col("completion_date") >= F.lit(six_months))


#import data from survey table
df_dat = spark.read.table("table_with_text_column_here")

df_dat = df_dat.toPandas()

#for testing:
#df_dat = df_dat.head(5)




# COMMAND ----------

#run the sentiment algorithm:

df_out = sentiment_analyzer.get_sentiment_analysis(df_dat, "RESPONSE")




# COMMAND ----------

df = df_out

# COMMAND ----------

performance = df[(df.COMMENT_RATING == 'Positive') | (df.COMMENT_RATING == 'Negative')]

sentiment_analyzer.results_matrix(performance, 'COMMENT_RATING', 'OVERALL_SENTIMENT', 'before')

# COMMAND ----------

df_out = df

# COMMAND ----------

#prepare output columns for pyspark

df_out = sentiment_analyzer.pd_prep(df_out)


# COMMAND ----------

# prepare spark data for push to delta table:
df_out['SUM_SCORES'] = pd.to_numeric(df_out['SUM_SCORES'], downcast='integer')
df_out['OVERALL_SCORE'] = pd.to_numeric(df_out['OVERALL_SCORE'], downcast ='integer')
df_out['EDP_LOAD_DTS'] = datetime.now()
df_out['SENTIMENT_IND'] = df_out['OVERALL_SENTIMENT'].apply(lambda x: 1 if x =='Positive' else 0)

# COMMAND ----------

#convert data from pandas to pyspark

schema = StructType([
    StructField("UNIQUE_ID", IntegerType(), True),
    StructField("COMPLETION_DATE", TimestampType(), True),
    StructField("COMMENT_RATING", StringType()),
    StructField("COMMENT_SECTION", StringType()),
    StructField("RESPONSE", StringType()),
    StructField("SENTENCE_COMP", StringType()),
    StructField("SENTENCE_NUM", IntegerType()),
    StructField("SUM_SCORES", IntegerType()),
    StructField("OVERALL_SCORE", IntegerType()),
    StructField("OVERALL_SENTIMENT", StringType()),
    StructField("SENTIMENT_IND", IntegerType()),
    StructField("TEXT_WORDS", StringType()),
    StructField("VIOL_DSC", StringType()),
    StructField("VIOL_IND", IntegerType()),
    StructField("EDP_LOAD_DTS", DateType(), True),
])
     




output_df = spark.createDataFrame(df_out, schema = schema)

# COMMAND ----------

#write output to file:
output_df.write.mode("overwrite").save("delta_table_name") #could use append method and filter results by load DT.
