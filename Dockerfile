# EDeN Docker container
#
# VERSION       0.1.0

FROM dmaticzka/docker-edenbase:latest

MAINTAINER Daniel Maticzka, maticzkd@gmail.com,Björn A. Grüning, bjoern.gruening@gmail.com and Fabrizio Costa

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN pip install jupyter

RUN mkdir /opt/EDeN
ADD . /opt/EDeN/
ENV PYTHONPATH $PYTHONPATH:/opt/EDeN/
ENV PATH $PATH:/opt/EDeN/bin/
WORKDIR /opt/EDeN

## from jupyter documentation
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--notebook-dir=/export/"]
