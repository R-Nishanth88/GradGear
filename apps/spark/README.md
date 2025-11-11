GradGear Spark Jobs
====================

Requirements
------------
- Python 3.11, Java 11+, Apache Spark 3.5+
- MongoDB accessible at the URI configured in `build_trending_skills.py` (defaults to `mongodb://localhost:27017`)

Install
-------
```
pip install -r requirements.txt
```

Run Intelligence Batch
----------------------
Populate trending skills and skill gap profiles that power the FastAPI intelligence endpoints:
```
spark-submit jobs/build_trending_skills.py
```
This reads the sample CSVs under `data/`, aggregates hot skills per domain, and writes the results into MongoDB collections `trending_skills` and `skill_gap_profiles`.

The FastAPI endpoints `/api/intelligence/trending`, `/api/intelligence/skill-gap`, and `/api/user/insights` will surface this data immediately after the job finishes.

Other Example Jobs
------------------
```
python jobs/gpa_predictor.py --input data/grades_sample.csv --output data/gpa_pred.parquet
```


