# Project Resources
Trello: https://trello.com/b/CVEqfIOC/lw-crypto

Spec Sheet: https://docs.google.com/document/d/1BKly3LA2Svv-t_KixOrGgIdR4fKdQtUMnaPj-9vjuE8/edit

Naive Baseline Model: https://www.tutorialspoint.com/time_series/time_series_naive_methods.htm

Gridsearch HyperParams: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# Project Notes 
20220816 - https://docs.google.com/document/d/1zS93xKPt-FLykjTJSjSv78_Z-riXiqAw5tALXtLWi2o/edit?usp=sharing



# Data analysis
- Document here the project: crypto_pred
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for crypto_pred in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/crypto_pred`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "crypto_pred"
git remote add origin git@github.com:{group}/crypto_pred.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
crypto_pred-run
```

# Install

Go to `https://github.com/{group}/crypto_pred` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/crypto_pred.git
cd crypto_pred
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
crypto_pred-run
```
