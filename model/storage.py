from google.cloud import storage

def download_file(bucket_name, file_name):
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)

  blob = bucket.blob(file_name)
  blob.download_to_filename(file_name)

  print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            file_name, bucket_name, file_name
        )
    )
  
  return file_name
