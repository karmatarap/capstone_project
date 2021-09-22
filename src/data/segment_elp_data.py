"""Download ELP data from AWS, segment out rumbles, upload to Azure Storage.

Author: Lucy Tan
"""

from datetime import datetime
import glob
import os
from pathlib import Path
import subprocess
import sys

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
 
from elephantcallscounter.data_processing.metadata_processing import \
   MetadataProcessing
from elephantcallscounter.services.data_processing_service import \
   create_file_segments
from elephantcallscounter.utils.path_utils import get_project_root, join_paths


AWS_BASE_PATH = 's3://congo8khz-pnnn/recordings/wav/'
AWS_CP_COMMAND = 'aws s3 cp {} ./elephantcallscounter/data/segments/TrainingSet/{}  --no-sign-request'
SEGMENTED_FILES_BASE_DIR = './elephantcallscounter/data/segments/CroppedTrainingSet'


def download_files(metadata_file_path):
   metadata = MetadataProcessing(metadata_filepath=metadata_file_path).load_metadata()
   for filename in metadata['filename'].unique():
      folder = filename.split('_')[0]
      aws_path = os.path.join(AWS_BASE_PATH, folder, filename)
      command = AWS_CP_COMMAND.format(aws_path, os.path.join(folder, filename))
      print(command)
      subprocess.run(command, shell=True)


def upload_to_azure(files_to_upload):
   now = datetime.now()
   connection_string = os.getenv('AZURE_CONNECTION_STRING')

   print(f'Copying {len(files_to_upload)} Blobs to Azure')

   blob_service_client = BlobServiceClient.from_connection_string(connection_string)

   timestamp = now.strftime("%Y%m%d%H%M%S")
   container_name = f'data-{timestamp}'
   container_client = blob_service_client.create_container(container_name)

   for file_to_upload in files_to_upload:
       filename = os.path.basename(file_to_upload)
       blob_client = container_client.get_blob_client(filename)
       blob_client.upload_blob(str(file_to_upload.resolve()))
       print(f'Copying {filename}')


def main(argv):
   metadata_file_path = join_paths(
              [get_project_root(), 'tests/test_fixtures/test_training_set.txt']
          )

   file_range = 30
   if len(argv) > 1:
      file_range = float(argv[1])

   download_files(metadata_file_path)
   create_file_segments(metadata_file_path, file_range)
   segmented_files = list(Path(SEGMENTED_FILES_BASE_DIR).rglob('*.wav'))
   upload_to_azure(segmented_files)


if __name__ == '__main__':
   main(sys.argv)