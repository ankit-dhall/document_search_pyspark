#Import Libraries
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import lower, col, regexp_replace, split, explode
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pprint import pprint
import string


def createInvertedIndex(sc, sqlContext):
	print("\n")
	print("########################################### Creating Inverted Index ###########################################")
    
    #Code to read the json data into dataframe ‘df2’
    sqlContext = SQLContext(sc)
   	df2 = sqlContext.read.json("/user/maria_dev/ankit/shakespeare_full.json")
   	
    #Calculating ‘N’ ( Total Number of Documents in the dataframe ‘df2’)
	N = df2.count()
	N = float(N)

	#Convert text_entry to Lower Case
   	columnName="text_entry"
   	df2 = df2.withColumn(columnName, lower(col(columnName)))
   	
   	#Remove punctuations from text_entry
	df2 = df2.withColumn(columnName, regexp_replace(col(columnName), '[^\sa-zA-Z0-9]', ''))

	#Drop extra columns from df2
	df2 = df2.drop('line_id', 'line_number', 'play_name', 'speaker', 'speech_number', 'type')

	#Split text_entry column into words by using the split function
	df2 = df2.withColumn("text_entry", split("text_entry", " "))

	#Explode eachtext_entry value into multiple rows to get _id with each word of text_entry
	df2 = df2.withColumn("token", explode(col("text_entry")))

	#Calculating Term Frequency by grouping based on ‘_id’ and ‘token’ and counting how many times each token occurs in each document
	df_tf = df2.groupby("_id", "token").agg(F.count("text_entry").alias("tf"))

	#Calculating Document Frequency by grouping on each token and counting the number of documents it occurs in
	df_idf = df2.groupby("token").agg(F.countDistinct("_id").alias("df"))

	#Converting ‘df’ column to Double Type in order for easy calculation later on
	df_idf = df_idf.withColumn("df", df_idf["df"].cast(DoubleType()))

	#Calculating IDF values
	df_idf = df_idf.withColumn("idf", F.log10(N/df_idf["df"]))

	#Joining df_tf and df_idf based on token columns
	tokensWithTfIdf = df_tf.join(df_idf, df_tf["token"] == df_idf["token"], how='left').drop(df_idf["token"])

	#Calculating TF-IDF Score
	tokensWithTfIdf = tokensWithTfIdf.withColumn("tf_idf", col("tf") * col("idf"))

	#Change ordering of Columns & Caching the Inverted Index
	tokensWithTfIdf = tokensWithTfIdf.select("token", "_id", "tf", "df", "idf", "tf_idf")
	print("\n")

	#Showing the top 20 rows of the Inverted Index
	tokensWithTfIdf.show()

	#Caching the Inverted Index for further usage
	tokensWithTfIdf.cache()

	print("###################################### Inverted Index Created and Saved ######################################")

	return tokensWithTfIdf
    
def search_words(sc, sqlContext, tokensWithTfIdf, query, N):
	print("\nSearching for :")

	#Printing the Query and the number of documents to be retrieved
	print(query, N)

	#Making the query to lower case
	query = query.lower()

	#Removing any punctuations
	query = query.translate(None, string.punctuation)

	#Splitting the query to words based on spaces
	words = query.split(" ")

	#Calculating the number of words in the query
	num_of_words = len(words)

	#Converting the query to a dataframe containing the query words
	query_df = sc.parallelize(words).map(lambda x:(x,)).toDF(["query_words"])

	#Dropping duplicate words from the query dataframe
	query_df = query_df.dropDuplicates()

	#Gets only those words from the Inverted Index that are present in the query
	query_subset = tokensWithTfIdf.join(query_df, query_df["query_words"] == tokensWithTfIdf["token"], how = "inner")

	#Counting the number of times a query word occurs in a document as well as summing up the tf-idf scores of the words that are present
	scored1 = query_subset.groupBy("_id").agg({"*":"count", "tf_idf":"sum"})

	#Renaming the count column to num_of_matched_words
	scored1 = scored1.withColumnRenamed("count(1)", "num_of_matched_words")

	#Renaming the sum(tf_idf) column to temp_score
	scored1 = scored1.withColumnRenamed("sum(tf_idf)", "temp_score")

	#Finds the actual score using the relevance scoring formula
	scored2 = scored1.select(scored1["_id"], (scored1["temp_score"] * scored1["num_of_matched_words"]) / num_of_words)

	#Renaming the score column
	scored = scored2.withColumnRenamed("((temp_score * num_of_matched_words) / " + str(num_of_words) + ")", "score")

	#Sorts the document id’s in descending order based on calculated scores
	result_docs = scored.sort("score", ascending=False)

	#Rounding the scores to 3 decimal places
	result_docs = result_docs.withColumn("score", F.round(result_docs["score"], 3))

	#Loading the actual data file to view results
	data_df = sqlContext.read.json("/user/maria_dev/ankit/shakespeare_full.json")

	#Joining the retrieved document dataframe with the actual data file in order to view results
	result_df = result_docs.join(data_df, result_docs["_id"] == data_df["_id"], how="inner").drop(data_df["_id"])

	#Dropping the unnecessary columns
	result_df = result_df.drop("line_id", "line_number", "play_name", "speaker", "speech_number", "type")

	#Sorting again after join according to scores and then keeping only N number of required documents
	result_df = result_df.sort("score", ascending=False).limit(int(N))

	#Storing the retrieved results in tuples
	final_tuples = tuple((row['_id'], row['score'], row['text_entry']) for row in result_df.collect())

	#Printing the result tuples
	pprint(final_tuples)

	return final_tuples
    

def main(sc, sqlContext):
	
	#Create Inverted Index
	tokensWithTfIdf = createInvertedIndex(sc, sqlContext)

	#Define the Queries
	queries = ["to be or not", "so far so", "if you said so"]

	#DEfine the number of documents to be retrieved
	n_docs = [1, 3, 5]

	print("\n")
	print("############################################### Searching ###############################################")

	for query in queries:
		for n in n_docs:
			result_tuples = search_words(sc, sqlContext, tokensWithTfIdf, query, n)
    
if __name__  == "__main__":
	conf = SparkConf().setAppName("TF-IDF Searching") #Set Spark configuration
	sc = SparkContext(conf = conf) #Set spark context with config
	sc.setLogLevel("ERROR")	#Display only ERROR logs
	sqlContext = SQLContext(sc) #Define SQLContext
	main(sc, sqlContext)
	sc.stop() #Stop SQLContext
    
    

#spark-submit --master yarn-client --executor-memory 512m --num-executors 3 --executor-cores 1 --driver-memory 512m tf_idf_search.py    
