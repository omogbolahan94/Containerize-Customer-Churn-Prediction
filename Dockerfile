FROM python:3.9

RUN pip install pipenv

WORKDIR docker-app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["server.py", "model_c=0.1.bin", "./"]

EXPOSE 8080
ENTRYPOINT ["waitress-serve", "server:app"]

