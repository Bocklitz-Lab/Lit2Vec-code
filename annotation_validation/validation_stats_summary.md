# Expert Validation Summary

## Per-Expert Statistics

| expert   |   n_tasks |   field_rating_count |   field_rating_mean |   field_rating_median |   field_rating_std |   field_rating_pct_ge4 | field_rating_dist                         |   field_revision_nonempty_count |   field_revision_nonempty_pct |   summary_rating_count |   summary_rating_mean |   summary_rating_median |   summary_rating_std |   summary_rating_pct_ge4 | summary_rating_dist                       |   summary_revision_nonempty_count |   summary_revision_nonempty_pct |   summary_len_words_mean |   summary_revision_len_words_mean |
|:---------|----------:|---------------------:|--------------------:|----------------------:|-------------------:|-----------------------:|:------------------------------------------|--------------------------------:|------------------------------:|-----------------------:|----------------------:|------------------------:|---------------------:|-------------------------:|:------------------------------------------|----------------------------------:|--------------------------------:|-------------------------:|----------------------------------:|
| JAMILE   |        25 |                   24 |               4.625 |                     5 |           0.646899 |                   0.88 | {"1": 0, "2": 0, "3": 2, "4": 5, "5": 17} |                               7 |                          0.28 |                     24 |                 4.375 |                       5 |             0.923721 |                     0.84 | {"1": 0, "2": 2, "3": 1, "4": 7, "5": 14} |                                12 |                            0.48 |                    30.88 |                           36.1667 |
| SARA     |        25 |                   24 |               4.875 |                     5 |           0.448427 |                   0.92 | {"1": 0, "2": 0, "3": 1, "4": 1, "5": 22} |                               2 |                          0.08 |                     22 |                 5     |                       5 |             0        |                     0.88 | {"1": 0, "2": 0, "3": 0, "4": 0, "5": 22} |                                 3 |                            0.12 |                    34.4  |                           30      |


## Cross-Expert Agreement

| label                                               |   n_overlap |   kappa |   pearson_r |   spearman_r |   exact_match_pct |   within_1pt_pct |
|:----------------------------------------------------|------------:|--------:|------------:|-------------:|------------------:|-----------------:|
| Field Classification Rating (aligned on task_title) |           5 |     nan |         nan |          nan |              0.6  |              0.8 |
| Summary Rating (aligned on task_title)              |           4 |     nan |         nan |          nan |              0.75 |              1   |


If present, confusion matrices were written to CSV files:

- `confusion_field_rating_task_id.csv` or `_task_title.csv`

- `confusion_summary_rating_task_id.csv` or `_task_title.csv`
