## With pydantic
from pydantic import BaseModel
class User(BaseModel):
    name: str
    age: int
user = User(name="Alice", age="25")  # Pydantic will auto-convert or validate
val=user.age
print(type(val))
print(f"with pydantic: {user.age}")  # 25 (converted to int)


## With out pydantic
class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
user = User("Alice", "25")  # No error — even though age should be int
val=user.age
print(type(val))
print(f"with out pydantic: {user.age}")  # "25" (a string)
