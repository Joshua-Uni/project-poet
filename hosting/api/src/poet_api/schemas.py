from pydantic import BaseModel, Field, field_validator

class GenerateRequest(BaseModel):
    title: str = Field(default = "Meowth's Shanty", description="Poem Title")
    temperature: float = Field(default=1, gt=0)
    max_tokens: int = Field(default=100, gt=0, lt=256)
    top_k: int | None = Field(default=None, gt=0, lt=10_000)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @field_validator("title")
    @classmethod
    def check_title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must have non-empty title")
        return v



class GenerateResponse(BaseModel):
    title: str
    poem: str
    tokens: int