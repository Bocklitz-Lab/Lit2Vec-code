import sqlite3
import json
import pandas as pd

# Connect to SQLite DB
conn = sqlite3.connect("label_studio3.sqlite4")
cursor = conn.cursor()

# Query tasks + annotations + project title
cursor.execute("""
SELECT task.id, task.data, task_completion.result, project.title
FROM task
JOIN task_completion ON task_completion.task_id = task.id
JOIN project ON task.project_id = project.id
""")

# Store all records per project
records_by_project = {
    "JAMILE": [],
    "SARA": []
}

for row in cursor.fetchall():
    task_id, data_json, result_json, project_title = row
    data = json.loads(data_json)
    result = json.loads(result_json)

    # Extract base data
    task_title = data.get("title", "")   # <-- Task title from JSON
    field_class = data.get("field_classification", "")
    summary = data.get("summary", "")
    field_rating = ""
    field_revision = ""
    summary_rating = ""
    summary_revision = ""

    # Parse annotation values
    for item in result:
        name = item.get("from_name")
        value = item.get("value", {})

        if name == "field_classification_rating":
            field_rating = value.get("rating", "")
        elif name == "field_classification_revision":
            field_revision = value.get("text", [""])[0]
        elif name == "summary_rating":
            summary_rating = value.get("rating", "")
        elif name == "summary_revision":
            summary_revision = value.get("text", [""])[0]

    # Normalize project label
    if "JAMILE" in project_title.upper():
        project_key = "JAMILE"
    elif "SARA" in project_title.upper():
        project_key = "SARA"
    else:
        continue  # Skip unknown projects

    # Add record to appropriate project
    records_by_project[project_key].append({
        "task_id": task_id,
        "task_title": task_title,   # <-- new column
        "field_classification": field_class,
        "field_classification_rating": field_rating,
        "field_classification_revision": field_revision,
        "summary": summary,
        "summary_rating": summary_rating,
        "summary_revision": summary_revision
    })

# Export each project to separate CSV
for project_key, records in records_by_project.items():
    df = pd.DataFrame(records)
    output_file = f"field_summary_annotations_{project_key}.csv"
    df.to_csv(output_file, index=False)
    print(f"Exported {len(records)} records to {output_file}")
