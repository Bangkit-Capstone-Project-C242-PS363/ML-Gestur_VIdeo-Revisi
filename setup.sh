#!/bin/bash

python -m venv sign_language_env

source sign_language_env/bin/activate

cat > requirements.txt << EOL
numpy==2.0.2
tensorflow==2.18.0
opencv-python==4.10.0.84
websockets==12.0
EOL

pip install -r requirements.txt