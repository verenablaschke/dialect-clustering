:: Use UTF-8 for the command line and file output.
chcp 65001
set pythonioencoding=utf8
:: Use the same hash seed every time so the results are consistent across runs.
set pythonhashseed=123
python cluster.py
