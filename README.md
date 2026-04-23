### To run Survey Demo Project locally:

```bash
git clone repo
python -m venv ./.venv
poetry install
poetry env activate
cd ./survey
python ./server.py
# Open http://localhost:8000/ in browser
```
Example results used in survey can also be found as .html files in /survey folder


### To run final Prototype locally: 
```bash
git clone repo
python -m venv ./.venv
poetry install
poetry env activate
uvicorn src.main:app --reload --reload-dir src
# Open http://127.0.0.1:8000/ in browser
```
