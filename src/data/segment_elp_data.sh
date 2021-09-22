# Download ELP data from AWS, segment out rumbles, upload to Azure Storage.
# Author: Lucy Tan

sudo apt install ffmpeg pipenv awscli
git clone https://github.com/AI-Cloud-and-Edge-Implementations/Project15-G4.git
cd Project15-G4
pipenv install --requirements.txt || pipenv install azure-iot-device azure-eventhub azure-storage-blob==12.9.0 azure-storage-queue==12.1.5 azureml-core azureml-dataprep boto3~=1.16.49 botocore~=1.19.49 click==7.1.2 environs~=9.3.2 pydub~=0.24.1 pytest pandas==0.25.3 matplotlib~=3.3.3 scipy==1.4.1 numpy~=1.19.2 librosa==0.8.0 noisereduce~=1.1.0 soundfile==0.10.3.post1 split-folders==0.4.3 opencv-python==4.5.1.48 flask==1.1.2 flask-googlemaps==0.4.1 flask-script==2.0.6 flask-sqlalchemy==2.4.4 flask-migrate==2.7.0 tensorflow  requests~=2.25.1 scikit-learn~=0.24.2 SQLAlchemy~=1.4.15 alembic~=1.6.3 --skip-lock

cat > elephantcallscounter/services/data_processing_service.py << 'EOF'
from elephantcallscounter.data_processing.metadata_processing import \
    MetadataProcessing
from elephantcallscounter.data_processing.segment_files import SegmentFiles


def create_file_segments(file_name, file_range=30):
    metadata = MetadataProcessing(metadata_filepath=file_name)
    segment_files = SegmentFiles(False, file_range=file_range)
    segment_files.process_segments(
        segment_files.ready_file_segments(metadata.load_metadata())
    )


EOF

export AZURE_CONNECTION_STRING='DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;'


pipenv shell "python ../segment_elp_data.py 30"