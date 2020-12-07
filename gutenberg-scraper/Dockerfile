FROM python:3.8-slim
COPY ./catalog.marc catalog.marc
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./main.py main.py
CMD ["python", "main.py"]