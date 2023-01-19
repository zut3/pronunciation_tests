from pydantic import BaseModel

class SimRequest(BaseModel):
    first: list[str]
    second: list[str]
