from __future__ import annotations

# MELD has 7 distinct emotion classes
EMOTION_NAMES = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

EMOTION_NAME_TO_LABEL = {
    name: index for index, name in enumerate(EMOTION_NAMES)
}

SENTIMENT_NAMES = ["negative", "neutral", "positive"]

SENTIMENT_NAME_TO_LABEL = {
    name: index for index, name in enumerate(SENTIMENT_NAMES)
}