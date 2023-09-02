import pandas as pd


def dataset_by_course(input_csv, output_folder):
    df = pd.read_csv(input_csv)
    unique_courses = df["course"].unique()
    for course in unique_courses:
        course_df = df[df["course"] == course]
        output_csv_path = f"{output_folder}/{course}.csv"
        course_df.to_csv(output_csv_path, index=False)


dataset_by_course("data/base/moodle_numerico.csv", "data")
