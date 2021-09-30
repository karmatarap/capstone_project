"""Download ELP data from AWS, segment out rumbles, upload to Azure Storage.

Author: Lucy Tan
"""

import os
from pathlib import Path
import subprocess
import sys

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from data_processing_service import create_file_segments
from metadata_processing import MetadataProcessing


AWS_BASE_PATH = 's3://congo8khz-pnnn/recordings/wav/'
AWS_CP_COMMAND = 'aws s3 cp {} ./data/segments/TrainingSet/{}  --no-sign-request'
SEGMENTED_FILES_BASE_DIR = './data/segments/CroppedTrainingSet'
AZURE_CONTAINER_NAME = 'elp-data'


def download_file_from_aws(filename):
    folder = filename.split('_')[0]
    aws_path = os.path.join(AWS_BASE_PATH, folder, filename)
    local_path = os.path.join(folder, filename)
    command = AWS_CP_COMMAND.format(aws_path, local_path)
    print(command)
    subprocess.run(command, shell=True)


def initialize_azure_blob_storage_container_client():
    connection_string = os.getenv('AZURE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    try:
        return blob_service_client.create_container(AZURE_CONTAINER_NAME)
    except Exception:
        return blob_service_client.get_container_client(AZURE_CONTAINER_NAME)


def upload_to_azure_and_delete_local(container_client, files_to_upload):
    print(f'Copying {len(files_to_upload)} Blobs to Azure')

    for file_to_upload in files_to_upload:
        filename = os.path.basename(file_to_upload)
        try:
            blob_client = container_client.get_blob_client(filename)
            with open(file_to_upload, 'rb') as f:
                blob_client.upload_blob(f.read())
            print(f'Copying {filename}')
            os.remove(file_to_upload.resolve())
        except Exception as e:
            print(f'Failed to upload {filename}: {e}')


def main(argv):
    file_range = 30
    if len(argv) > 1:
        file_range = float(argv[1])

    azure_container_client = initialize_azure_blob_storage_container_client()

    metadata_file_path = '../../data/nn_ele_hb_00-24hr_TrainingSet_v2.txt'
    metadata = MetadataProcessing(metadata_filepath=metadata_file_path).load_metadata()
    metadata_by_filename = MetadataProcessing.split_metadata_into_groups(metadata)
    for filename, cur_metadata in metadata_by_filename.items():
        download_file_from_aws(filename)
        create_file_segments(cur_metadata, file_range)
        segmented_files = list(Path(SEGMENTED_FILES_BASE_DIR).rglob('*.wav'))
        upload_to_azure_and_delete_local(azure_container_client, segmented_files)


if __name__ == '__main__':
    main(sys.argv)