# mlops-practico-7
# Trabajo Práctico 7 - MLOps

Proyecto de MLOps desarrollado con FastAPI, Docker, GitHub Actions y Amazon S3.  
El objetivo del proyecto es implementar un flujo completo que incluya integración continua, despliegue de una API de Machine Learning, reentrenamiento del modelo y almacenamiento de artefactos en S3.

## Tecnologías utilizadas

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Docker
- GitHub Actions
- Amazon S3
- Boto3

## Estructura del proyecto

```bash
mlops-practico-7/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retrain.yml
├── app/
│   ├── main.py
│   └── utils.py
├── model/
│   └── model.pkl
├── scripts/
│   └── train.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md

## Instalación y ejecución

1. Clonar el repositorio:

git clone https://github.com/TU-USUARIO/mlops-practico-7.git
cd mlops-practico-7

## Crear entorno virtual 

python -m venv .venv
.venv\Scripts\activate

##Instalar dependencias

pip install -r requirements.txt

##Ejecutar API

uvicorn app.main:app --reload

##Abrir en navegador 

http://127.0.0.1:8000
http://127.0.0.1:8000/docs

##Integración con S3

El modelo entrenado (model.pkl) y el dataset (iris.csv) se almacenan automáticamente en Amazon S3 mediante el script train.py.

Archivos generados:

models/model.pkl
data/iris.csv

Se utilizan variables de entorno para las credenciales de AWS.

##CI/CD

Se configuró GitHub Actions para:

Instalación automática de dependencias
Entrenamiento del modelo
Validación de la aplicación

Archivo:

.github/workflows/ci.yml

##Reentrenamiento 

Se implementó un workflow de reentrenamiento automático y manual usando GitHub Actions.

Archivo:

.github/workflows/retrain.yml

## Evidencias

API funcionando en /docs
Pipeline CI exitoso
Docker funcionando
Archivos almacenados en S3

##Repositorio 

https://github.com/yulissanavarromol-ux/mlops-practico-7