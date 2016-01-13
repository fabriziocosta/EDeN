# EDeN Docker container
#
# VERSION       0.1.0

FROM dmaticzka/docker-edenbase:latest

MAINTAINER Björn A. Grüning, bjoern.gruening@gmail.com and Fabrizio Costa

ADD requirements.txt .

RUN mkdir /opt/EDeN
ADD . /opt/EDeN/
ENV PYTHONPATH $PYTHONPATH:/opt/EDeN/
ENV PATH $PATH:/opt/EDeN/bin/
WORKDIR /opt/EDeN
