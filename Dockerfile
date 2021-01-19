FROM python:3.8
COPY ./src src
COPY ./resource/Pipelines resource/Pipelines
COPY ./resource/df_full_preprocessed.pkl ./resource/df_full_preprocessed.pkl
COPY ./requirements.txt requirements.txt
RUN apt-get update && \
    apt-get -y install gcc mono-mcs
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
COPY ./resources/arial.ttf /usr/share/fonts/truetype/arial.ttf
CMD cd ./src/frontend && python ./app.py
EXPOSE 5000
