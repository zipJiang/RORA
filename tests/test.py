from src.generator.generator_model import APIModel

model = APIModel(model="gpt-4")
result = model("Hello, my dog is cute.")
print(result)

print(model.gpt_usage("gpt-4"))