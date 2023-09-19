from imblearn.over_sampling import SMOTE

OVERSAMPLING_TECHNIQUE = SMOTE
# moodle dataset
MOODLE_DATA_PATH = "data/"
MOODLE_IGNORED_FEAT = ["course"]
MOODLE_TARGET = "mark"
# kaggle dataset
KAGGLE_DATA_PATH = "data/kaggle/"
KAGGLE_TARGET = "Target"
# student prediction dataset
STUDENT_PRED_PATH = "data/student_grade_prediction/train_validate/csv/none.csv"
STUDENT_PRED_TARGET = "label (fail=1, pass=0)"