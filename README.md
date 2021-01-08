# Document Search and Retrieval using PySpark
### Building a Search Engine using PySpark

This repository houses the source code and results of developing a search engine using PySpark on Hadoop.


## Table of Contents

- [Project Overview](#projectoverview)
- [Data Description](#datadescription)
- [Technical Overview](#technicaloverview)
- [Results](#results)

***

<a id='projectoverview'></a>
## Project Overview

This project aims to implement a search engine using PySPark on Hadoop.

It portrays a high-level functionality and an interesting use-case of PySpark to use TF-IDF to index an entire dataset and perform a search given a query.

This particular project utilizes Spark RDD's as well as specialized Spark functions that can be leveraged to perform high-performance distributed computing.

<a id='datadescription'></a>
## Data Description

The dataset used, is a collection of over 100,000 documents structured into a JSON file with each document containing textual data spanning varying lengths.

The dataset used can be can be referred here [shakespeare_full.json](https://github.com/ankit-dhall/document_search_pyspark/blob/main/shakespeare_full.json)

<a id='technicaloverview'></a>
## Technical Overview

The project has been divided into various steps which include:
* Data Exploration and Pre-Processing
* Indexing Dataset using TF-IDF
* Performing Search to Retrieve Top Documents using TF-IDF Scores
* Writing a Spark Job File to perform Search using Hadoop File Environment

<a id='results'></a>
## Results

The PySpark job file / source code can be referred to here [tf_idf_search_pyspark.py](https://github.com/ankit-dhall/document_search_pyspark/blob/main/tf_idf_search_pyspark.py)

The outputs of each part of the project, as well as individual code snippets have been documented in the file [TF_IDF_Search_PySpark.pdf](https://github.com/ankit-dhall/document_search_pyspark/blob/main/TF_IDF_Search_PySpark.pdf)