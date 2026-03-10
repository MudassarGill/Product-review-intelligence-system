import boto3
import pandas as pd
import logging
from io import StringIO

# Assuming you have a custom logger in src.logger
from src.logger import logging


class S3Operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name='us-east-1'):
        """
        Initialize the S3 connection class with AWS credentials and S3 bucket details.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info('Data ingestion from S3 bucket initialized.')

    def fetch_file_from_s3(self, file_key):
        """
        Fetch a CSV file from S3 bucket and return a dataframe.
        
        :param file_key: S3 file path
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 bucket with {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch '{file_key}' from S3: {e}")
            return None