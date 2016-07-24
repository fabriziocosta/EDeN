# EDeN Docker container
#
# VERSION       0.1.0

FROM dmaticzka/docker-edenbase:latest

MAINTAINER Daniel Maticzka (maticzkd@informatik.uni-freiburg.de), Björn A. Grüning (bjoern.gruening@gmail.com) and Fabrizio Costa (xfcosta@gmail.com)

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get -qq update && apt-get install -y  python-rdkit librdkit1 rdkit-data

RUN mkdir /opt/EDeN
ADD . /opt/EDeN/
ENV PYTHONPATH $PYTHONPATH:/opt/EDeN/
ENV PATH $PATH:/opt/EDeN/bin/
WORKDIR /opt/EDeN
