# tests/run_lessons.py
import importlib


LESSON_MODULES = {
    1: "tests.lesson_1_test_vec_add",
    2: "tests.lesson_2_test_vec_scl",
    3: "tests.lesson_3_test_vec_dot",
    4: "tests.lesson_4_test_vec_len",
    5: "tests.lesson_5_test_vec_nor",
    6: "tests.lesson_6_test_mat_siz",
    7: "tests.lesson_7_test_mat_add",
    8: "tests.lesson_8_test_mat_row",
    9: "tests.lesson_9_test_mat_col",
    10: "tests.lesson_10_test_mat_ide",
    11: "tests.lesson_11_test_mat_mul",
    12: "tests.lesson_12_test_mat_tra",
    13: "tests.lesson_13_test_mat_vec_mul",
    14: "tests.lesson_14_test_lu",
    15: "tests.lesson_15_test_solve",
    16: "tests.lesson_16_test_inv",
    17: "tests.lesson_17_test_vec_prj",
    18: "tests.lesson_18_test_mat_prj",
    19: "tests.lesson_19_test_ortho",
    20: "tests.lesson_20_test_qr",
    21: "tests.lesson_21_test_det",
}

def run_lesson(lesson_id: int):
    if lesson_id not in LESSON_MODULES:
        raise ValueError(f"Lesson ID {lesson_id} does not exist.")
    
    module_name = LESSON_MODULES[lesson_id]
    try:
        # Dynamically import the lesson module
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}: {e}")
    
    # Execute the `run()` function in the lesson module
    if hasattr(module, "run") and callable(module.run):
        module.run()
        print(f"Lesson {lesson_id} tests passed successfully.")
    else:
        raise AttributeError(f"{module_name} does not have a callable run() function.")
