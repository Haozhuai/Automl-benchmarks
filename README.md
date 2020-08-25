# AutoML Benchmarks of Tabular Dataset

    Empirical Comparson among popular autoML toolkit include google autoML, autogluon, tpot, h2o, etc.
    
## Requirement
    google cloud
    autogluon
    pytorch-tabnet
    tpot
    h2o
    
## step

    git clone https://github.com/Haozhuai/Automl-benchmarks.git
    cd Automl-benchmarks  # should decompress dataset.7z ！！
    python tabnet.py  # tabnet benchmarks
    python wide_deep.py # wide and deep benchmarks
    python tpot_benchmarks.py # tpot benchmarks
    sh h2o_run.sh  # h2o benchmarks
    python google_split.py  # google dataset split
    
    
    