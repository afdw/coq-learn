from pydantic import BaseModel, ConfigDict


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    theorem_path: str
    goals: str
    tactic: str
