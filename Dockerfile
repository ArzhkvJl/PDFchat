FROM python:3


COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
COPY fune-tuning.py ./

CMD [ "python", "fine-tunning.py" ]