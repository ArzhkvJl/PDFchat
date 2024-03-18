FROM python:3

WORKDIR /raid/jtuskaeva

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "fine-tunning.py" ]