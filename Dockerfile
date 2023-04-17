FROM python:3.10

RUN mkdir -p /mouse-gait-analysis
COPY . /mouse-gait-analysis

WORKDIR /mouse-gait-analysis
RUN pip install -r requirements.txt
RUN pip install .
RUN python -m ipykernel install --name base --display-name base
