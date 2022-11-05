import os
import replicate
import json
import mimetypes
import zipfile

# TODO: set this to the new DreamBooth training model
MODEL_NAME = "replicate/hello-world"
MODEL_VERSION = "5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"

def train():
  print("Training model...")

  # TODO: uncomment this check
  # print("Checking training data")
  # # count files in the data directory by guessing mime type
  # image_count = 0
  # for file in os.listdir("data"):
  #   if mimetypes.guess_type(file)[0].startswith("image/"):
  #     image_count += 1  
  
  # if image_count == 0:
  #   raise Exception("No images found in data directory")

  print("Zipping training data")
  with zipfile.ZipFile("data.zip", "w") as zip:
    for file in os.listdir("data"):
      zip.write(os.path.join("data", file))

  print(f"Training on Replicate using model {MODEL_NAME}@{MODEL_VERSION}")
  print(f"https://replicate.com/{MODEL_NAME}/versions/{MODEL_VERSION}")
  model = replicate.models.get(MODEL_NAME)
  version = model.versions.get(MODEL_VERSION)

  # TODO: use data.zip as input
  return version.predict(text="GitHub Actions")


if __name__ == "__main__":
    output = train()
    print("Done training. Output:")
    print(json.dumps(output))

    # TODO: download generated weights
