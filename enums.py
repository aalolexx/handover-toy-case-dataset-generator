"""
Enums and constants for the surgery dataset generator.
"""
from enum import Enum, auto


class PersonType(Enum):
    """Type of person in the scene."""
    DOCTOR = auto()
    ASSISTANT = auto()


class AssistantState(Enum):
    """States for an active assistant."""
    IDLE = auto()
    MOVING_TO_PREP_TABLE = auto()
    PREPARING = auto()
    HOLDING = auto()
    MOVING_TO_DOCTOR = auto()
    GIVING = auto()
    WAITING_BY_DOCTOR = auto()  # Waiting near doctor to receive instrument back
    RECEIVING = auto()
    MOVING_FROM_DOCTOR = auto()


class DoctorState(Enum):
    """States for an active doctor."""
    IDLE = auto()
    HOLDING = auto()
    WORKING = auto()
    GIVING = auto()
    RECEIVING = auto()


class InstrumentState(Enum):
    """States for an instrument."""
    ON_TABLE = auto()
    HELD = auto()
    IN_HANDOVER = auto()
    IN_USE = auto()


class ActionLabel(Enum):
    """YOLO action labels."""
    ASSISTANT_PREPARES = 0
    DOCTOR_WORKS = 1
    PERSON_HOLDS = 2
    ASSISTANT_GIVES = 3
    ASSISTANT_RECEIVES = 4
    HANDOVER = 5  # Additional label for any handover
    PERSON = 6  # Generic actor label
    DOCTOR = 7
    ASSISTANT = 8


# Mapping from label to string name (for YOLO classes.txt)
LABEL_NAMES = {
    ActionLabel.ASSISTANT_PREPARES: "green_prepares",
    ActionLabel.DOCTOR_WORKS: "blue_works",
    ActionLabel.PERSON_HOLDS: "actor_holds",
    ActionLabel.ASSISTANT_GIVES: "green_gives",
    ActionLabel.ASSISTANT_RECEIVES: "green_receives",
    ActionLabel.HANDOVER: "handover",
    ActionLabel.PERSON: "actor",
    ActionLabel.DOCTOR: "blue",
    ActionLabel.ASSISTANT: "green",
}