FROM python:3

RUN mkdir -p /raid/jtuskaeva
WORKDIR /raid/jtuskaeva
COPY requirements.txt ./

ADD ./requirements.txt /raid/jtuskaeva/requirements.txt

ADD . /raid/jtuskaeva

CMD [ "python", "fine-tunning.py" ]