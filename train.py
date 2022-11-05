import replicate
import json

# TODO: set this to the new DreamBooth training model
MODEL_NAME = "replicate/hello-world"
MODEL_VERSION = "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"

def train():
  print("Training model...")
  model = replicate.models.get(MODEL_NAME)
  version = model.versions.get(MODEL_VERSION)

  # TODO: use data.zip as input
  return version.predict(text="GitHub Actions")

if __name__ == "__main__":
    output = train()
    print("Done training. Output:")
    print(json.dumps(output))

    # TODO: download generated weights
