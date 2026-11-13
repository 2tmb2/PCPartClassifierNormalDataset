import json
import mimetypes
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
UPLOADED_FILES_DIR = ROOT / "uploaded_files"
IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
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
    .image-sources {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(180px, .8fr);
      gap: 10px;
    }
    .image-sources > div { min-width: 0; }
    #image { min-width: 0; }
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
    #select-image { background: #315f73; }
    #select-image:hover { background: #24495c; }
    #model { overflow: hidden; text-overflow: ellipsis; }
    #camera-dialog {
      width: min(680px, calc(100% - 24px));
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 20px;
      color: var(--ink);
    }
    #camera-dialog::backdrop { background: rgb(18 31 25 / 70%); }
    #camera-video {
      display: block;
      width: 100%;
      max-height: 65vh;
      margin-bottom: 14px;
      background: #101713;
      object-fit: contain;
    }
    .camera-actions { display: flex; gap: 10px; }
    .camera-actions button { flex: 1; }
    #cancel-camera { background: #64736c; }
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
      main { grid-template-columns: minmax(0, 1fr); width: min(100% - 24px, 600px); margin: 12px auto; gap: 12px; }
      .panel { min-width: 0; padding: 17px; }
      select, input, button { min-width: 0; max-width: 100%; font-size: 16px; }
      #select-image { min-height: 50px; }
      .preview { width: 100%; min-width: 0; height: min(42dvh, 320px); min-height: 200px; aspect-ratio: 4 / 3; overflow-wrap: anywhere; }
      .preview img { max-height: 100%; }
      #camera-dialog { max-height: calc(100dvh - 24px); overflow-y: auto; padding: 14px; }
      #camera-video { max-height: min(65vh, calc(100dvh - 190px)); }
      .camera-actions button { min-height: 48px; }
    }
    @media (max-width: 500px) {
      .image-sources { grid-template-columns: 1fr; }
      .image-sources { gap: 12px; }
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
        <div class="image-sources">
          <div>
            <label for="select-image">Camera or computer</label>
            <input id="image" type="file" accept="image/*" capture="environment" hidden disabled>
            <button id="select-image" type="button" disabled>Select image</button>
          </div>
          <div>
            <label for="uploaded-file">From uploaded files</label>
            <select id="uploaded-file" disabled>
              <option value="">Loading uploaded files...</option>
            </select>
          </div>
        </div>
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
  <dialog id="camera-dialog" aria-label="Take a photo">
    <video id="camera-video" autoplay playsinline></video>
    <div class="camera-actions">
      <button id="capture-photo" type="button">Capture photo</button>
      <button id="cancel-camera" type="button">Cancel</button>
    </div>
  </dialog>
  <script>
    const imageInput = document.querySelector("#image");
    const uploadedFileSelect = document.querySelector("#uploaded-file");
    const modelSelect = document.querySelector("#model");
    const runButton = document.querySelector("#run");
    const selectImageButton = document.querySelector("#select-image");
    const cameraDialog = document.querySelector("#camera-dialog");
    const cameraVideo = document.querySelector("#camera-video");
    const status = document.querySelector("#status");
    let previewUrl = null;
    let currentFile = null;
    let cameraStream = null;

    function updatePreview(file, uploadedFileName = "") {
      currentFile = file;
      uploadedFileSelect.value = uploadedFileName;
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
      runButton.disabled = !modelSelect.value || !currentFile;
    }

    function stopCamera() {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
      }
      cameraVideo.srcObject = null;
    }

    async function openCamera() {
      cameraDialog.showModal();
      status.textContent = "Waiting for camera permission...";
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!cameraDialog.open) {
          stopCamera();
          return;
        }
        cameraVideo.srcObject = cameraStream;
        status.textContent = "Position the part in view, then capture a photo.";
      } catch (error) {
        cameraDialog.close();
        status.textContent = "Camera unavailable. Choose or capture an image...";
        imageInput.click();
      }
    }

    function selectImage() {
      if (window.isSecureContext && navigator.mediaDevices?.getUserMedia) {
        openCamera();
        return;
      }
      status.textContent = "Choose or capture an image...";
      imageInput.click();
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
        selectImageButton.disabled = options.models.length === 0;
        uploadedFileSelect.disabled = true;
        runButton.disabled = true;
        status.textContent = options.models.length ? "Choose an image and model." : "No model checkpoints found in models/.";
        if (options.models.length > 0) {
          modelSelect.value = options.models[0].path;
          modelSelect.dispatchEvent(new Event("change"));
        }
        const uploadedResponse = await fetch("/api/uploaded-files");
        const uploadedResult = await uploadedResponse.json();
        if (!uploadedResponse.ok) throw new Error(uploadedResult.error || "Could not load uploaded files.");
        uploadedFileSelect.replaceChildren();
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = uploadedResult.files.length ? "Choose an uploaded file..." : "No image files found";
        uploadedFileSelect.append(placeholder);
        for (const fileName of uploadedResult.files) {
          const option = document.createElement("option");
          option.value = fileName;
          option.textContent = fileName;
          uploadedFileSelect.append(option);
        }
        uploadedFileSelect.disabled = options.models.length === 0 || uploadedResult.files.length === 0;
      } catch (error) {
        status.textContent = error.message;
      }
    }

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0] || null;
      imageInput.value = "";
      if (file) updatePreview(file);
    });
    uploadedFileSelect.addEventListener("change", async () => {
      const fileName = uploadedFileSelect.value;
      if (!fileName) return;
      status.textContent = `Loading ${fileName}...`;
      try {
        const query = new URLSearchParams({ name: fileName });
        const response = await fetch(`/api/uploaded-file?${query}`);
        if (!response.ok) {
          const result = await response.json();
          throw new Error(result.error || "Could not load the selected file.");
        }
        const blob = await response.blob();
        updatePreview(new File([blob], fileName, { type: blob.type || "application/octet-stream" }), fileName);
        status.textContent = "Uploaded file selected. Run prediction when ready.";
      } catch (error) {
        uploadedFileSelect.value = "";
        status.textContent = error.message;
      }
    });
    selectImageButton.addEventListener("click", selectImage);
    document.querySelector("#cancel-camera").addEventListener("click", () => cameraDialog.close());
    cameraDialog.addEventListener("close", stopCamera);
    document.querySelector("#capture-photo").addEventListener("click", () => {
      if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
        status.textContent = "The camera is not ready yet.";
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = cameraVideo.videoWidth;
      canvas.height = cameraVideo.videoHeight;
      canvas.getContext("2d").drawImage(cameraVideo, 0, 0);
      canvas.toBlob(blob => {
        if (!blob) {
          status.textContent = "Could not capture a photo. Please try again.";
          return;
        }
        const filename = `webcam-${new Date().toISOString().replaceAll(":", "-")}.jpg`;
        imageInput.value = "";
        updatePreview(new File([blob], filename, { type: "image/jpeg" }));
        cameraDialog.close();
        status.textContent = "Photo captured. Run prediction when ready.";
      }, "image/jpeg", 0.92);
    });
    
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
      const file = currentFile;
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
        const confidenceLabel = result.confidence_label || "Confidence";
        document.querySelector("#confidence").textContent = `${confidenceLabel}: ${(result.confidence * 100).toFixed(1)}%`;
        if (result.top_predictions && result.top_predictions.length > 1) {
          document.querySelector("#prediction-2").textContent = result.top_predictions[1].class;
          document.querySelector("#confidence-2").textContent = `${confidenceLabel}: ${(result.top_predictions[1].confidence * 100).toFixed(1)}%`;
        }
        if (result.top_predictions && result.top_predictions.length > 2) {
          document.querySelector("#prediction-3").textContent = result.top_predictions[2].class;
          document.querySelector("#confidence-3").textContent = `${confidenceLabel}: ${(result.top_predictions[2].confidence * 100).toFixed(1)}%`;
        }
        status.textContent = `Predicted with ${result.model}.`;
      } catch (error) {
        status.textContent = error.message;
      } finally {
        runButton.disabled = !modelSelect.value || !currentFile;
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
        self.pca = None
        self.classifier = None

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

        # These local checkpoints are trusted and may contain sklearn objects.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "resnet18":
            raise ValueError("This app supports ResNet-18 checkpoints only.")

        class_names = checkpoint["class_names"]
        classifier_type = checkpoint.get("classifier")
        if classifier_type is None:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            pca = None
            classifier = None
        elif classifier_type in {"centroid", "svm", "pca_nearest_centroid", "pca_svm"}:
            pca = checkpoint["pca"]
            classifier = checkpoint["classifier_model"]
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported checkpoint classifier: {classifier_type}")
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
        self.pca = pca
        self.classifier = classifier
        self.loaded_model_path = model_path

    def predict(self, model_relative_path, image_bytes, filename):
      self.load_model(model_relative_path)
      with Image.open(BytesIO(image_bytes)) as image:
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0)
      input_tensor = input_tensor.to(self.device)

      with torch.inference_mode():
        output = self.model(input_tensor)
        confidence_label = "Confidence"
        if self.classifier is None:
            probabilities = torch.softmax(output, dim=1)[0]
            class_indices = list(range(len(self.class_names)))
        else:
            features = self.pca.transform(output.cpu().numpy())
            class_indices = self.classifier.classes_.tolist()
            if hasattr(self.classifier, "predict_proba"):
                probabilities = torch.as_tensor(
                    self.classifier.predict_proba(features)[0]
                )
            else:
                # Older checkpoints lack probabilities; show normalized scores.
                confidence_label = "Relative score"
                if hasattr(self.classifier, "centroids_"):
                    distances = torch.linalg.vector_norm(
                        torch.as_tensor(features[0])
                        - torch.as_tensor(self.classifier.centroids_),
                        dim=1,
                    )
                    scores = -distances
                else:
                    scores = torch.as_tensor(
                        self.classifier.decision_function(features)[0]
                    )
                    if scores.ndim == 0:
                        scores = torch.stack((-scores, scores))
                probabilities = torch.softmax(scores, dim=0)
        top_3_probs, top_3_indices = torch.topk(
            probabilities, k=min(3, len(class_indices))
        )
        top_3_probs = top_3_probs.cpu().numpy().tolist()
        top_3_indices = [
            int(class_indices[index]) for index in top_3_indices.cpu().tolist()
        ]
        
        return {
            "actual": Path(filename).name,
            "predicted": self.class_names[top_3_indices[0]],
            "confidence": top_3_probs[0],
            "confidence_label": confidence_label,
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

        if request.path == "/api/uploaded-files":
          upload_root = UPLOADED_FILES_DIR.resolve()
          files = sorted(
            path.name
            for path in UPLOADED_FILES_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.resolve().parent == upload_root
          ) if UPLOADED_FILES_DIR.is_dir() else []
          self.send_json(200, {"files": files})
          return

        if request.path == "/api/uploaded-file":
          query = parse_qs(request.query)
          file_name = query.get("name", [""])[0]
          file_path = UPLOADED_FILES_DIR / file_name
          try:
            resolved_path = file_path.resolve(strict=True)
          except OSError:
            self.send_json(404, {"error": "Uploaded file not found."})
            return
          if (
            not file_name
            or Path(file_name).name != file_name
            or resolved_path.parent != UPLOADED_FILES_DIR.resolve()
            or resolved_path.suffix.lower() not in IMAGE_SUFFIXES
            or not resolved_path.is_file()
          ):
            self.send_json(404, {"error": "Uploaded file not found."})
            return
          file_size = resolved_path.stat().st_size
          if file_size > MAX_UPLOAD_SIZE:
            self.send_json(413, {"error": "Selected file is too large."})
            return
          body = resolved_path.read_bytes()
          self.send_response(200)
          self.send_header("Content-Type", mimetypes.guess_type(resolved_path.name)[0] or "application/octet-stream")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
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