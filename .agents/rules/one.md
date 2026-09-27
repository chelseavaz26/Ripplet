---
trigger: always_on
---

Always derive all analytics modules from the single shared dataset in pipeline/data.parquet via pipeline/loader.py — never introduce a second dataset or bypass get_window().