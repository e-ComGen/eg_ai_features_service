class DeductionResult(BaseModel):
    context_clues: str = Field(
        ...,
        description="Logical deductions, synonyms, or industry standards found in the text relevant to the target feature."
    )
    confidence_score: int = Field(
        ...,
        description="An integer from 1 to 100. How confident are you that this context directly implies the value for the target feature?"
    )