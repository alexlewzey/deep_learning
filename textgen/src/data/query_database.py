"""Queries database to gather data required for project and complete as many joins as possible before bringing it
into the project and either caching it in the raw data folder or or piping it straight into the make dataset script
for processing.
As data is returned from the query its dtypes should be checked and made memory efficient before it is serialised and
cached."""

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
