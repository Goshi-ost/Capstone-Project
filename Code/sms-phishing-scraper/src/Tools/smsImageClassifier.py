"""Deep-learning image classifier: SMS screenshot vs. regular social media post.

Approach: transfer learning. A ResNet18 pretrained on ImageNet is fine-tuned
with a new 2-class head on labelled screenshots.

Uses labelled SMS and non-SMS image directories. When no non-SMS directory
is provided, synthetic non-SMS images are generated instead.

Usage:
    python smsImageClassifier.py train --data ../../Resources/smsMessages --non-sms-data ../../Resources/NonSMSMessages
    python smsImageClassifier.py predict path/to/image.png [more.png ...]

Requires: pip install torch torchvision

"""

import argparse
import hashlib
import os
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sms_image_classifier.pt")
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

train_tf = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ColorJitter(0.2, 0.2, 0.2),
	transforms.RandomRotation(5),
	transforms.RandomHorizontalFlip(p=0.1),
	transforms.ToTensor(),
	transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(MEAN, STD),
])


def _device():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=2, pretrained=True):
	weights = models.ResNet18_Weights.DEFAULT if pretrained else None
	model = models.resnet18(weights=weights)
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	return model


def generate_synthetic_non_sms(img_size=IMG_SIZE):
	"""Generate a synthetic non-SMS image (noise, gradients, patterns, shapes)."""
	synthetic_type = random.choice(['noise', 'gradient', 'pattern', 'shapes'])
	
	img = Image.new('RGB', (img_size, img_size))
	
	if synthetic_type == 'noise':
		pixels = [random.randint(0, 255) for _ in range(img_size * img_size * 3)]
		img.putdata([(pixels[i], pixels[i+1], pixels[i+2]) 
					 for i in range(0, len(pixels), 3)])
	
	elif synthetic_type == 'gradient':
		draw = ImageDraw.Draw(img)
		for y in range(img_size):
			color = (
				int(255 * y / img_size),
				int(128 * (1 - y / img_size)),
				200
			)
			draw.line([(0, y), (img_size, y)], fill=color)
	
	elif synthetic_type == 'pattern':
		draw = ImageDraw.Draw(img)
		for i in range(0, img_size, 20):
			draw.rectangle([i, i, i+15, i+15], fill=(random.randint(50, 200), 
														   random.randint(50, 200), 
														   random.randint(50, 200)))
	
	else:
		draw = ImageDraw.Draw(img)
		for _ in range(5):
			x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
			x2, y2 = random.randint(0, img_size), random.randint(0, img_size)
			draw.ellipse([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=(random.randint(50, 200), 
												   random.randint(50, 200), 
												   random.randint(50, 200)))
	
	return img


class SMSClassifierDataset(Dataset):
	"""Dataset combining SMS images with real or synthetic non-SMS images."""
	
	def __init__(self, sms_image_paths, num_synthetic=None, transform=None, is_train=True, non_sms_image_paths=None):
		self.sms_paths = sms_image_paths
		self.non_sms_paths = non_sms_image_paths
		self.transform = transform
		self.is_train = is_train
		
		if self.non_sms_paths is not None:
			num_synthetic = 0
		elif num_synthetic is None:
			num_synthetic = len(sms_image_paths)
		
		self.num_synthetic = num_synthetic
		self.synthetic_cache = {}
		
		if not is_train:
			for i in range(num_synthetic):
				self.synthetic_cache[i] = generate_synthetic_non_sms()
	
	def __len__(self):
		return len(self.sms_paths) + (len(self.non_sms_paths) if self.non_sms_paths is not None else self.num_synthetic)
	
	def __getitem__(self, idx):
		if idx < len(self.sms_paths):
			with Image.open(self.sms_paths[idx]) as image:
				img = image.convert('RGB')
			label = 0
		elif self.non_sms_paths is not None:
			with Image.open(self.non_sms_paths[idx - len(self.sms_paths)]) as image:
				img = image.convert('RGB')
			label = 1
		else:
			synthetic_idx = idx - len(self.sms_paths)
			if self.is_train:
				img = generate_synthetic_non_sms()
			else:
				img = self.synthetic_cache[synthetic_idx]
			label = 1
		
		if self.transform:
			img = self.transform(img)
		
		return img, label


def _unique_images(data_dir):
	data_dir = Path(data_dir)
	if not data_dir.is_dir():
		raise ValueError(f"Image directory does not exist: {data_dir}")
	images = {}
	for path in sorted(data_dir.iterdir()):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			digest = hashlib.sha256(path.read_bytes()).digest()
			images.setdefault(digest, path)
	if not images:
		raise ValueError(f"No images found in {data_dir}")
	return images


def _split_images(paths, val_split):
	if len(paths) < 2:
		raise ValueError("Each image class must contain at least two unique images")
	n_val = min(len(paths) - 1, max(1, int(len(paths) * val_split)))
	train_idx, val_idx = random_split(
		range(len(paths)), [len(paths) - n_val, n_val],
		generator=torch.Generator().manual_seed(42),
	)
	return [paths[i] for i in train_idx], [paths[i] for i in val_idx]


# Train the image classifier and save the checkpoint with the best validation accuracy.
def train(data_dir, epochs=15, batch_size=32, lr=1e-3, val_split=0.2, model_path=MODEL_PATH, non_sms_dir=None):
	device = _device()
	print(f"Using device: {device}")
	if not 0 < val_split < 1:
		raise ValueError("val_split must be between 0 and 1")
	sms_images = _unique_images(data_dir)
	train_paths, val_paths = _split_images(list(sms_images.values()), val_split)
	print(f"Found {len(sms_images)} unique SMS images")
	train_non_sms = val_non_sms = None
	if non_sms_dir is not None:
		non_sms_images = _unique_images(non_sms_dir)
		if sms_images.keys() & non_sms_images.keys():
			raise ValueError("Identical images found in both SMS and non-SMS directories")
		train_non_sms, val_non_sms = _split_images(list(non_sms_images.values()), val_split)
		print(f"Found {len(non_sms_images)} unique non-SMS images")
		print(f"Train: {len(train_paths)} SMS + {len(train_non_sms)} non-SMS")
		print(f"Val: {len(val_paths)} SMS + {len(val_non_sms)} non-SMS")
	else:
		print(f"Train: {len(train_paths)} SMS images + {len(train_paths)} synthetic")
		print(f"Val: {len(val_paths)} SMS images + {len(val_paths)} synthetic")

	train_set = SMSClassifierDataset(train_paths, transform=train_tf, is_train=True, non_sms_image_paths=train_non_sms)
	val_set = SMSClassifierDataset(val_paths, transform=eval_tf, is_train=False, non_sms_image_paths=val_non_sms)
	
	train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	val_dl = DataLoader(val_set, batch_size=batch_size, num_workers=0)
	
	model = build_model(num_classes=2).to(device)
	classes = ("sms", "non_sms")
	
	optimizer = torch.optim.Adam([
		{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": lr / 10},
		{"params": model.fc.parameters(), "lr": lr},
	])
	loss_fn = nn.CrossEntropyLoss()
	best_acc = -1.0
	
	print("\nStarting training...")
	for epoch in range(1, epochs + 1):
		model.train()
		train_loss = 0.0
		
		for x, y in train_dl:
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			loss = loss_fn(model(x), y)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		
		model.eval()
		correct = 0
		val_loss = 0.0
		
		with torch.no_grad():
			for x, y in val_dl:
				x, y = x.to(device), y.to(device)
				logits = model(x)
				val_loss += loss_fn(logits, y).item()
				correct += (logits.argmax(1).cpu() == y).sum().item()
		
		acc = correct / len(val_set)
		avg_val_loss = val_loss / len(val_dl)
		
		print(f"epoch {epoch}/{epochs}  train_loss={train_loss / len(train_dl):.4f}  val_loss={avg_val_loss:.4f}  val_acc={acc:.3f}")
		
		if acc > best_acc:
			best_acc = acc
			os.makedirs(os.path.dirname(model_path), exist_ok=True)
			torch.save({
				"state_dict": model.state_dict(),
				"classes": classes,
			}, model_path)

	print(f"\nBest val_acc={best_acc:.3f}; saved to {model_path}")


class SmsImageClassifier:
	"""Load a trained model and classify images."""

	# Load the saved class labels and model weights onto the selected device.
	def __init__(self, model_path=MODEL_PATH):
		self.device = _device()
		ckpt = torch.load(model_path, map_location=self.device)
		self.classes = ckpt["classes"]
		self.model = build_model(len(self.classes), pretrained=False)
		self.model.load_state_dict(ckpt["state_dict"])
		self.model.to(self.device).eval()

	# Predict the class label and confidence score for one image.
	def predict(self, image_path):
		img = Image.open(image_path).convert("RGB")
		x = eval_tf(img).unsqueeze(0).to(self.device)
		with torch.no_grad():
			probs = torch.softmax(self.model(x), dim=1)[0]
		idx = int(probs.argmax())
		return self.classes[idx], float(probs[idx])

	# Check whether an image is classified as SMS above the confidence threshold.
	def is_sms(self, image_path, threshold=0.5):
		label, conf = self.predict(image_path)
		return label == "sms" and conf >= threshold


# Parse CLI options and dispatch to the training or prediction workflow.
def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest="cmd", required=True)
	
	t = sub.add_parser("train")
	t.add_argument("--data", required=True, help="Path to SMS images directory")
	t.add_argument("--non-sms-data", help="Path to non-SMS images directory; omit to use synthetic images")
	t.add_argument("--epochs", type=int, default=15)
	t.add_argument("--batch-size", type=int, default=32)
	t.add_argument("--lr", type=float, default=1e-3)
	t.add_argument("--model", default=MODEL_PATH)
	
	p = sub.add_parser("predict")
	p.add_argument("images", nargs="+")
	p.add_argument("--model", default=MODEL_PATH)
	
	args = parser.parse_args()

	if args.cmd == "train":
		train(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model_path=args.model, non_sms_dir=args.non_sms_data)
	else:
		clf = SmsImageClassifier(args.model)
		for path in args.images:
			label, conf = clf.predict(path)
			print(f"{path}: {label} ({conf:.2%})")


if __name__ == "__main__":
	main()
