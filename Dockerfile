FROM python:3

WORKDIR /Users/julsajul/Downloads/llama2

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./fine-tunning.py" ]