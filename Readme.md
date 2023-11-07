# NPath

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BKrdrLrWdxUZFPnSxUJWZW4wfoulavUx?usp=sharing)


## Description
Exploring path sequences in GA4 BigQuery data

## Setup
1. Create a new Google Cloud project
2. Enable the BigQuery API
3. Ensure that GA4 data is being sent to BigQuery
4. Get the dataset ID and table ID for the GA4 data
5. Create a service account with BigQuery read access. [Here](https://docs.aws.amazon.com/dms/latest/sbs/bigquery-redshift-migration-step-1.html) is a good guide.
6. Download the service account key as a JSON file
7. Create a new file in the root directory called `service_account.json` and paste the contents of the JSON file into it
8. Run `pip install -r requirements.txt` to install the required Python packages
9. Get your API key from OpenAI if you want to run analyze_clusters.
10. Open `demo.ipynb` in Jupyter Notebook and run the cells

## Components
* `plot_important_features_prefixspan`: Plots the important conversion path sequences of the PrefixSpan model.
* `convertor_review`: Sequence Patterns of Similarity and Anomalies in Non-Convertors that are clustered with Convertors
* `analyze_divergence`: Scores the similarity of non-convertor to convertor sequences.
* `analyze_clusters`: Clusters users based on their navigational paths and labels clusters.
