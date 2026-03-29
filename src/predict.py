"""
predict.py
==========
Load a saved model + scaler and predict on new input data.

Usage (CLI)
-----------
python src/predict.py \
    --area 6000 --bedrooms 3 --bathrooms 2 --stories 2 \
    --mainroad yes --guestroom no --basement no \
    --hotwaterheating no --airconditioning yes \
    --parking 1 --prefarea yes --furnishingstatus furnished
"""

import argparse
import numpy as np
import joblib
import os

FEATURE_ORDER = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "parking", "prefarea",
    "furnishingstatus_semi-furnished", "furnishingstatus_unfurnished",
]

BINARY_MAP = {"yes": 1, "no": 0}

FURNISHING_MAP = {
    "furnished":       (0, 0),
    "semi-furnished":  (1, 0),
    "unfurnished":     (0, 1),
}


def build_feature_vector(args) -> np.ndarray:
    semi, unfurn = FURNISHING_MAP.get(args.furnishingstatus, (0, 0))
    row = [
        args.area,
        args.bedrooms,
        args.bathrooms,
        args.stories,
        BINARY_MAP.get(args.mainroad, 0),
        BINARY_MAP.get(args.guestroom, 0),
        BINARY_MAP.get(args.basement, 0),
        BINARY_MAP.get(args.hotwaterheating, 0),
        BINARY_MAP.get(args.airconditioning, 0),
        args.parking,
        BINARY_MAP.get(args.prefarea, 0),
        semi,
        unfurn,
    ]
    return np.array(row).reshape(1, -1)


def predict(
    model_path: str = "models/best_model.pkl",
    scaler_path: str = "models/scaler.pkl",
    **kwargs,
) -> float:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Build feature array from kwargs
    class Args:
        pass

    args = Args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    X = build_feature_vector(args)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def main():
    parser = argparse.ArgumentParser(description="House Price Predictor")
    parser.add_argument("--area", type=int, required=True)
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=int, required=True)
    parser.add_argument("--stories", type=int, required=True)
    parser.add_argument("--mainroad", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--guestroom", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--basement", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--hotwaterheating", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--airconditioning", type=str, required=True, choices=["yes", "no"])
    parser.add_argument("--parking", type=int, required=True)
    parser.add_argument("--prefarea", type=str, required=True, choices=["yes", "no"])
    parser.add_argument(
        "--furnishingstatus", type=str, required=True,
        choices=["furnished", "semi-furnished", "unfurnished"],
    )
    parser.add_argument("--model", type=str, default="models/best_model.pkl")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl")

    args = parser.parse_args()

    price = predict(
        model_path=args.model,
        scaler_path=args.scaler,
        area=args.area,
        bedrooms=args.bedrooms,
        bathrooms=args.bathrooms,
        stories=args.stories,
        mainroad=args.mainroad,
        guestroom=args.guestroom,
        basement=args.basement,
        hotwaterheating=args.hotwaterheating,
        airconditioning=args.airconditioning,
        parking=args.parking,
        prefarea=args.prefarea,
        furnishingstatus=args.furnishingstatus,
    )

    print(f"\n🏠  Predicted House Price: ₹ {price:,.0f}\n")


if __name__ == "__main__":
    main()
