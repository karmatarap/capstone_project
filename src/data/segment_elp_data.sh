# Download ELP data from AWS, segment out rumbles, upload to Azure Storage.
# Author: Lucy Tan

sudo apt install ffmpeg pipenv awscli
pipenv install azure-storage-blob librosa numpy~=1.19.2 pandas pydub --skip-lock

# Replace with the real connection string.
export AZURE_CONNECTION_STRING='DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;'


pipenv shell "python segment_elp_data.py 30"