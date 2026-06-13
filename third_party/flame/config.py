from dataclasses import dataclass


@dataclass
class FlameConfig:
    flame_model_path: str = "./assets/flame2023.pkl"
    static_landmark_embedding_path: str = "./assets/flame_static_embedding.pkl"
    dynamic_landmark_embedding_path: str = "./assets/flame_dynamic_embedding.npy"

    shape_params: int = 50
    expression_params: int = 10
    pose_params: int = 6
    batch_size: int = 2

    use_face_contour: bool = False
    use_3D_translation: bool = True
