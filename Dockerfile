FROM deepnote/python:3.7

RUN apt update && apt install -y ffmpeg libsm6 libxext6 libsndfile1


COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN jupyter nbextension install --py widgetsnbextension  && jupyter nbextension enable widgetsnbextension --py