# sport-fittings

Different scripts and tools to fit your sport positions based on videos (e.g. bike fitting based on Mediapipe)

```bash
mkdir -p datasets/cycling-sebastian
uv run kaggle datasets download -p datasets/cycling-sebastian sebastianjrz/cycling-sebastian --unzip
```

```bash
curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o ~/Downloads/cycling-sebastian.zip\
  https://www.kaggle.com/api/v1/datasets/download/sebastianjrz/cycling-sebastian
unzip ~/Downloads/cycling-sebastian.zip -d datasets/cycling-sebastian
rm ~/Downloads/cycling-sebastian.zip
```
