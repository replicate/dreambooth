import os
import replicate
import json

MODEL_NAME = "replicate/hello-word"
MODEL_VERSION = "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"

def train():
  # echo env var to make sure it's set
  print("REPLICATE_API_TOKEN: ", os.environ["REPLICATE_API_TOKEN"])
  print("Training model...")
  model = replicate.models.get(MODEL_NAME)
  # version = model.versions.get(MODEL_VERSION)
  return model.predict(text="GitHub Actions")

if __name__ == "__main__":
    output = train()
    print("Done training. Output:")
    print(json.dumps(output))
