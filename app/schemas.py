"""Pydantic response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetection(BaseModel):
    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str


class ObjectsResponse(BaseModel):
    count: int
    detections: list[ObjectDetection]
    inference_ms: float
    model: str
    annotated_image: str | None = None


class FaceDetection(BaseModel):
    bbox: list[float]
    confidence: float
    landmarks: list[list[float]] = Field(default_factory=list)


class FacesResponse(BaseModel):
    count: int
    faces: list[FaceDetection]
    inference_ms: float
    annotated_image: str | None = None


class RecognizedFace(BaseModel):
    bbox: list[float]
    confidence: float
    name: str
    score: float
    matched: bool


class RecognizeResponse(BaseModel):
    count: int
    faces: list[RecognizedFace]
    inference_ms: float
    threshold: float
    known_identities: list[str]
    annotated_image: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict
    face_identities: list[str]
