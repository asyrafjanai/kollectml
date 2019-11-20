# Machine Learning Demo

This demo uses data from [OneDrive](https://onedrive.live.com/?id=58F0C85D32E24FFE%21127&cid=58F0C85D32E24FFE)

# Install Dependencies and prepare data.

This demo requires

- Python 3+
- Install the requirements and dependencies library

```
$ conda env create -f environment.yml
```

## Download the data

The first time you run this demo, fetch the data from onedrive/or 
used the sample data in storage folder

## Start training ML model

```
python train.py
```

Next, to see the model performance, run:
```
python inspect_performance.py
```

Then you can do batch prediction using

```
python predict.py 
```

and prediction results will be saved as CSV file in `storage` directory

## Making prediction using API endpoint

1. 
```
$ export FLASK_APP=app.py 
```
_Note: replace `export` to `set` for windows machine_

2. Run the flask app on local server
```
$ flask run
```

3. Go to your browser and go to the API endpoint

Try a different model:

```
python search.py rambo test_8
```
## Use it on your own data

If you want to reuse these scripts on your own data, choose the names of files and models, change or copy the settings.cfg. If you want to use your own file, change the utils.py.
