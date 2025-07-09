import pandas as pd
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
from xgboost import XGBClassifier
# from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
# from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import joblib
from config import MODEL_OUTPUT_PATH_ONNX, MODEL_OUTPUT_PATH_JOBLIB


# def export_pipeline_to_onnx(pipeline, X_sample, file_path):
#     """
#     Exports the scikit-learn pipeline to ONNX format.
#     Further optimizations like quantization could be applied here for production.
#     """
#     print(f"\nExporting pipeline to ONNX format at {file_path}...")
    
#     # Define the input schema for the ONNX model based on the sample data
#     initial_types = []
#     for col, dtype in X_sample.dtypes.items():
#         if dtype == 'object':
#             # For categorical string features
#             initial_types.append((col, StringTensorType([None, 1])))
#         elif pd.api.types.is_integer_dtype(dtype):
#             # For integer features
#             initial_types.append((col, Int64TensorType([None, 1])))
#         else:
#             # For float/numerical features
#             initial_types.append((col, FloatTensorType([None, 1])))

#     update_registered_converter(
#     XGBClassifier,
#     "XGBoostXGBClassifier",
#     calculate_linear_classifier_output_shapes,
#     convert_xgboost,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )
#     try:
#         onx = convert_sklearn(pipeline, initial_types=initial_types, target_opset=15)
#         with open(file_path, "wb") as f:
#             f.write(onx.SerializeToString())
#         print("ONNX export successful.")
#     except Exception as e:
#         print(f"Error during ONNX export: {e}")


def export_pipeline_to_joblib(pipeline, file_path):
    """Exports the scikit-learn pipeline to a joblib file."""
    print(f"\nExporting pipeline to joblib format at {file_path}...")
    try:
        joblib.dump(pipeline, file_path)
        print("Joblib export successful.")
    except Exception as e:
        print(f"Error during joblib export: {e}")


def export_model(pipeline, X_sample, export_type="onnx"):
    """
    Wrapper function to export the model pipeline to the specified format.

    Args:
        pipeline: The trained scikit-learn pipeline to export.
        X_sample (pd.DataFrame): A sample of the input data (e.g., X_train.head(1))
                                 required for defining the ONNX input schema.
        export_type (str): The format to export to. Can be 'onnx' or 'joblib'.
    """
    if export_type.lower() == "onnx":
        pass
        # export_pipeline_to_onnx(pipeline, X_sample, MODEL_OUTPUT_PATH_ONNX)
    elif export_type.lower() == "joblib":
        export_pipeline_to_joblib(pipeline, MODEL_OUTPUT_PATH_JOBLIB)
    else:
        print(f"Error: Unknown export_type '{export_type}'. Please choose 'onnx' or 'joblib'.")