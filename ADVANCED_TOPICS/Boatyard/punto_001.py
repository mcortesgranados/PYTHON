import boto3

def process_objects_in_bucket(bucket_name: str, challenge_token: str) -> str:
    """
    Process objects in the specified S3 bucket.
    
    Args:
        bucket_name (str): The name of the S3 bucket.
        challenge_token (str): The challenge token string.
        
    Returns:
        str: Processed string based on the objects in the bucket.
    """
    # Create a Boto3 client for interacting with S3
    s3_client = boto3.client('s3')
    
    try:
        # Call list_objects_v2 to list objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        processed_strings = []
        # Check if the response contains any objects
        if 'Contents' in response:
            # Iterate through the objects
            for obj in response['Contents']:
                object_key = obj['Key']
                # Check if the object contains the prefix __cb__
                if '__cb__' in object_key:
                    # Retrieve the object content
                    file_contents = s3_client.get_object(Bucket=bucket_name, Key=object_key)['Body'].read().decode('utf-8')
                    # Concatenate the file contents with the challenge token
                    concatenated_string = file_contents + challenge_token
                    # Replace every fourth character with an underscore
                    processed_string = ''.join([c if (i + 1) % 4 != 0 else '_' for i, c in enumerate(concatenated_string)])
                    processed_strings.append(processed_string)
        
        # Join the processed strings with newline characters and return
        return '\n'.join(processed_strings)
    except Exception as e:
        # Print error message and raise the exception
        print(f"Error processing objects: {e}")
        raise e

def lambda_handler(event, context):
    # Specify the name of the S3 bucket
    bucket_name = 'mcgbucketcoderbyte'
    # Specify the challenge token
    challenge_token = 'v153jil6f2'
    
    # Call the function to process objects in the bucket and return the result
    return process_objects_in_bucket(bucket_name, challenge_token)
