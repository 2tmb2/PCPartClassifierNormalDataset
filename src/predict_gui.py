import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Laptop Part Classifier</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1d2925;
      --muted: #64736c;
      --paper: #f3f5f0;
      --surface: #ffffff;
      --line: #d7ded8;
      --green: #245c47;
      --green-dark: #194533;
      --lime: #d9e8a4;
      --coral: #c65a43;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: var(--green-dark);
      color: white;
      padding: 22px max(24px, calc((100vw - 1100px) / 2));
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 21px; font-weight: 650; }
    #device { color: #d9e8df; font-size: 13px; }
    main {
      width: min(1100px, calc(100% - 40px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(290px, .8fr);
      gap: 22px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 22px;
    }
    .field { margin-bottom: 18px; }
    label, .eyebrow {
      display: block;
      margin-bottom: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    select, input, button {
      min-height: 42px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: white;
      color: var(--ink);
      font: inherit;
      padding: 9px 11px;
    }
    button {
      border: 0;
      background: var(--green);
      color: white;
      cursor: pointer;
      font-weight: 650;
      transition: background .15s ease, transform .15s ease;
    }
    button:hover { background: var(--green-dark); }
    button:active { transform: translateY(1px); }
    button:disabled { cursor: wait; opacity: .65; }
    .preview {
      min-height: 390px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px dashed #b6c4ba;
      border-radius: 4px;
      background: #e9eee8;
      color: var(--muted);
    }
    .preview img { display: block; max-width: 100%; max-height: 520px; object-fit: contain; }
    .result { border-top: 1px solid var(--line); padding-top: 20px; margin-top: 20px; }
    .result + .result { margin-top: 16px; }
    .runner-up { border-top: none; padding-top: 0; margin-top: 12px; }
    .runner-up .eyebrow { font-size: 11px; margin-bottom: 4px; }
    .runner-up .value { font-size: 14px; }
    .runner-up #confidence-2, .runner-up #confidence-3 { font-size: 12px; }
    .value { overflow-wrap: anywhere; font-size: 16px; font-weight: 650; }
    #prediction { color: var(--green); font-size: 20px; }
    #confidence { color: var(--muted); font-size: 13px; margin-top: 4px; }
    #confidence-2, #confidence-3 { color: var(--muted); font-size: 12px; margin-top: 2px; }
    #status { min-height: 22px; color: var(--muted); font-size: 13px; margin-top: 14px; }
    #classes { max-height: 300px; overflow-y: auto; border: 1px solid var(--line); border-radius: 4px; padding: 8px; font-size: 13px; }
    .class-item { padding: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .class-item:nth-child(odd) { background: var(--paper); }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; gap: 3px; }
      main { grid-template-columns: 1fr; width: min(100% - 24px, 600px); margin: 12px auto; gap: 12px; }
      .panel { padding: 17px; }
      .preview { min-height: 250px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Laptop Part Classifier</h1>
    <div id="device">Loading local models...</div>
  </header>
  <main>
    <section class="panel" aria-label="Image preview">
      <div class="field">
        <label for="image">Choose an image from your computer</label>
        <input id="image" type="file" accept="image/*" disabled>
      </div>
      <div id="preview" class="preview">Choose an image to preview it.</div>
    </section>
    <section class="panel" aria-label="Prediction controls and result">
      <div class="field">
        <label for="model">Model checkpoint</label>
        <select id="model" disabled></select>
      </div>
      <button id="run" disabled>Run prediction</button>
      <div class="result">
        <div class="eyebrow">Actual (filename)</div>
        <div id="actual" class="value">-</div>
      </div>
      <div class="result">
        <div class="eyebrow">Predicted class</div>
        <div id="prediction" class="value">-</div>
        <div id="confidence"></div>
      </div>
      <div class="result runner-up">
        <div class="eyebrow">2nd Place</div>
        <div id="prediction-2" class="value">-</div>
        <div id="confidence-2"></div>
      </div>
      <div class="result runner-up">
        <div class="eyebrow">3rd Place</div>
        <div id="prediction-3" class="value">-</div>
        <div id="confidence-3"></div>
      </div>
      <div class="result">
        <div class="eyebrow">Available classes</div>
        <div id="classes">Select a model to see available classes.</div>
      </div>
      <div id="status" role="status">Choose an image and model.</div>
    </section>
  </main>
  <script>
    const imageInput = document.querySelector("#image");
    const modelSelect = document.querySelector("#model");
    const runButton = document.querySelector("#run");
    const status = document.querySelector("#status");
    let previewUrl = null;

    function updatePreview() {
      const file = imageInput.files[0];
      document.querySelector("#actual").textContent = file ? file.name : "-";
      document.querySelector("#prediction").textContent = "-";
      document.querySelector("#confidence").textContent = "";
      document.querySelector("#prediction-2").textContent = "-";
      document.querySelector("#confidence-2").textContent = "";
      document.querySelector("#prediction-3").textContent = "-";
      document.querySelector("#confidence-3").textContent = "";
      const preview = document.querySelector("#preview");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = file ? URL.createObjectURL(file) : null;
      preview.replaceChildren();
      if (previewUrl) {
        const image = document.createElement("img");
        image.alt = "Selected image";
        image.src = previewUrl;
        image.onerror = () => { preview.textContent = "This file cannot be previewed as an image."; };
        preview.append(image);
      } else {
        preview.textContent = "Choose an image to preview it.";
      }
      runButton.disabled = !modelSelect.value || !file;
    }

    async function initialize() {
      try {
        const response = await fetch("/api/options");
        const options = await response.json();
        if (!response.ok) throw new Error(options.error || "Could not load options.");
        document.querySelector("#device").textContent = `Inference device: ${options.device}`;
        for (const model of options.models) {
          const option = document.createElement("option");
          option.value = model.path;
          option.textContent = model.path;
          modelSelect.append(option);
        }
        modelSelect.disabled = options.models.length === 0;
        imageInput.disabled = options.models.length === 0;
        runButton.disabled = true;
        status.textContent = options.models.length ? "Choose an image and model." : "No model checkpoints found in models/.";
        if (options.models.length > 0) {
          modelSelect.value = options.models[0].path;
          modelSelect.dispatchEvent(new Event("change"));
        }
      } catch (error) {
        status.textContent = error.message;
      }
    }

    imageInput.addEventListener("change", updatePreview);
    
    modelSelect.addEventListener("change", async () => {
      const classesDiv = document.querySelector("#classes");
      if (!modelSelect.value) {
        classesDiv.textContent = "Select a model to see available classes.";
        return;
      }
      try {
        classesDiv.textContent = "Loading classes...";
        const response = await fetch(`/api/classes?model=${modelSelect.value}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not load classes.");
        classesDiv.replaceChildren();
        for (const className of result.classes) {
          const item = document.createElement("div");
          item.className = "class-item";
          item.textContent = className;
          item.title = className;
          classesDiv.append(item);
        }
      } catch (error) {
        classesDiv.textContent = `Error: ${error.message}`;
      }
    });
    
    runButton.addEventListener("click", async () => {
      const file = imageInput.files[0];
      if (!file || !modelSelect.value) return;
      runButton.disabled = true;
      status.textContent = "Running inference...";
      try {
        const query = new URLSearchParams({ model: modelSelect.value, filename: file.name });
        const response = await fetch(`/api/predict?${query}`, {
          method: "POST",
          headers: { "Content-Type": file.type || "application/octet-stream" },
          body: file
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Prediction failed.");
        document.querySelector("#actual").textContent = result.actual;
        document.querySelector("#prediction").textContent = result.predicted;
        document.querySelector("#confidence").textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
        if (result.top_predictions && result.top_predictions.length > 1) {
          document.querySelector("#prediction-2").textContent = result.top_predictions[1].class;
          document.querySelector("#confidence-2").textContent = `Confidence: ${(result.top_predictions[1].confidence * 100).toFixed(1)}%`;
        }
        if (result.top_predictions && result.top_predictions.length > 2) {
          document.querySelector("#prediction-3").textContent = result.top_predictions[2].class;
          document.querySelector("#confidence-3").textContent = `Confidence: ${(result.top_predictions[2].confidence * 100).toFixed(1)}%`;
        }
        status.textContent = `Predicted with ${result.model}.`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        runButton.disabled = !modelSelect.value || !imageInput.files[0];
      }
    });

    initialize();
  </script>
</body>
</html>
"""


class Predictor:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.models = self.find_models()
        self.loaded_model_path = None
        self.model = None
        self.transform = None
        self.class_names = []

    @staticmethod
    def find_models():
        if not MODEL_DIR.is_dir():
            return {}
        return {
            path.relative_to(MODEL_DIR).as_posix(): path
            for path in sorted(MODEL_DIR.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in {".pth", ".pt"}
            and "feature" not in path.stem.lower()
        }

    def load_model(self, relative_path):
        model_path = self.models[relative_path]
        if model_path == self.loaded_model_path:
            return

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "resnet18":
            raise ValueError("This app supports ResNet-18 checkpoints only.")

        class_names = checkpoint["class_names"]
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        interpolation = ResNet18_Weights.DEFAULT.transforms().interpolation
        image_transform = transforms.Compose([
            transforms.Resize(
                (checkpoint["image_size"], checkpoint["image_size"]),
                interpolation=interpolation,
            ),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["image_mean"], checkpoint["image_std"]),
        ])
        self.model = model
        self.class_names = class_names
        self.transform = image_transform
        self.loaded_model_path = model_path

    def predict(self, model_relative_path, image_bytes, filename):
      self.load_model(model_relative_path)
      with Image.open(BytesIO(image_bytes)) as image:
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
      input_tensor = input_tensor.to(self.device)

      with torch.inference_mode():
        probabilities = torch.softmax(self.model(input_tensor), dim=1)[0]
        top_3_probs, top_3_indices = torch.topk(probabilities, k=3)
        top_3_probs = top_3_probs.cpu().numpy().tolist()
        top_3_indices = top_3_indices.cpu().numpy().tolist()
        
        return {
            "actual": Path(filename).name,
            "predicted": self.class_names[top_3_indices[0]],
            "confidence": top_3_probs[0],
            "top_predictions": [
                {"class": self.class_names[idx], "confidence": conf}
                for idx, conf in zip(top_3_indices, top_3_probs)
            ],
            "model": model_relative_path,
        }


class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if request.path == "/api/options":
            predictor = self.server.predictor
            self.send_json(200, {
                "device": str(predictor.device),
                "models": [{"path": path} for path in predictor.models],
            })
            return

        if request.path == "/api/classes":
            predictor = self.server.predictor
            query = parse_qs(urlparse(self.path).query)
            model_path = query.get("model", [""])[0]
            if model_path not in predictor.models:
                self.send_json(400, {"error": "Select a listed model."})
                return
            try:
                predictor.load_model(model_path)
                self.send_json(200, {"classes": predictor.class_names})
            except Exception as error:
                self.send_json(400, {"error": str(error)})
            return

        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
          content_length = int(self.headers.get("Content-Length", "0"))
          if content_length <= 0:
            self.send_json(400, {"error": "Choose an image to upload."})
            return
          if content_length > MAX_UPLOAD_SIZE:
                self.send_json(413, {"error": "Request is too large."})
                return
          query = parse_qs(urlparse(self.path).query)
          model_path = query.get("model", [""])[0]
          filename = query.get("filename", [""])[0]
          predictor = self.server.predictor
          if model_path not in predictor.models:
            self.send_json(400, {"error": "Select a listed model."})
            return
          image_bytes = self.rfile.read(content_length)
          result = predictor.predict(model_path, image_bytes, filename)
          self.send_json(200, result)
        except Exception as error:
            self.send_json(400, {"error": str(error)})

    def log_message(self, format_string, *args):
        print(f"[predict-gui] {format_string % args}")


class PredictorServer(HTTPServer):
    def __init__(self, address, predictor):
        super().__init__(address, RequestHandler)
        self.predictor = predictor


def main():
    predictor = Predictor()
    server = PredictorServer(("0.0.0.0", 12345), predictor)
    url = f"http://0.0.0.0:{server.server_port}/"
    print(f"Laptop Part Classifier is running at {url}")
    print("Press Ctrl+C in this terminal to close it.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Closing Laptop Part Classifier.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()