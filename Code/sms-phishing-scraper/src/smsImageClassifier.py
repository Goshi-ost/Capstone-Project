"""Deep-learning image classifier: SMS screenshot vs. regular social media post.

Approach: transfer learning. A ResNet18 pretrained on ImageNet is fine-tuned
with a new 2-class head on labelled screenshots.

Dataset layout (ImageFolder; class names come from folder names):
    data/classifier/
        sms/      <- screenshots of text messages
        social/   <- Reddit/Twitter/Instagram etc. posts

Usage:
    python smsImageClassifier.py train --data ../data/classifier --epochs 10
    python smsImageClassifier.py predict path/to/image.png [more.png ...]

Requires: pip install torch torchvision

"""

"""
What we need to do:

1. Build a dataset in two folders, data/classifier/sms/ and data/classifier/social/.
2. SMS images can come from the r/Scams and r/phishing images your scraper already caches.
3. Social post images can come from other subreddits, for example memes, tweets or Instagram posts.
4. Aim for roughly balanced classes, 200 or more images each.
5. Use varied devices and themes, such as dark mode and iPhone versus Android. The model can otherwise learn the wrong cue, like the subreddit
"""

import argparse
import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sms_image_classifier.pt")
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.ColorJitter(0.2, 0.2, 0.2),
	transforms.RandomRotation(3),
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


def train(data_dir, epochs=10, batch_size=16, lr=1e-3, val_split=0.2, model_path=MODEL_PATH):
	device = _device()
	train_full = datasets.ImageFolder(data_dir, transform=train_tf)
	eval_full = datasets.ImageFolder(data_dir, transform=eval_tf)
	classes = train_full.classes

	# Same split indices for both datasets so validation uses no augmentation
	n_val = max(1, int(len(train_full) * val_split))
	train_idx, val_idx = random_split(
		range(len(train_full)), [len(train_full) - n_val, n_val],
		generator=torch.Generator().manual_seed(42),
	)
	train_set = torch.utils.data.Subset(train_full, list(train_idx))
	val_set = torch.utils.data.Subset(eval_full, list(val_idx))

	train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_dl = DataLoader(val_set, batch_size=batch_size)

	model = build_model(len(classes)).to(device)
	# Backbone fine-tuned at a low LR, new head at a higher one
	optimizer = torch.optim.Adam([
		{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": lr / 10},
		{"params": model.fc.parameters(), "lr": lr},
	])
	loss_fn = nn.CrossEntropyLoss()
	best_acc = -1.0

	for epoch in range(1, epochs + 1):
		model.train()
		for x, y in train_dl:
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			loss_fn(model(x), y).backward()
			optimizer.step()

		model.eval()
		correct = 0
		with torch.no_grad():
			for x, y in val_dl:
				correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
		acc = correct / len(val_set)
		print(f"epoch {epoch}/{epochs}  val_acc={acc:.3f}")

		if acc > best_acc:
			best_acc = acc
			os.makedirs(os.path.dirname(model_path), exist_ok=True)
			torch.save({"state_dict": model.state_dict(), "classes": classes}, model_path)

	print(f"Best val_acc={best_acc:.3f}; saved to {model_path}")


class SmsImageClassifier:
	"""Load a trained model and classify images."""

	def __init__(self, model_path=MODEL_PATH):
		self.device = _device()
		ckpt = torch.load(model_path, map_location=self.device)
		self.classes = ckpt["classes"]
		self.model = build_model(len(self.classes), pretrained=False)
		self.model.load_state_dict(ckpt["state_dict"])
		self.model.to(self.device).eval()

	def predict(self, image_path):
		"""Return (label, confidence) for one image."""
		img = Image.open(image_path).convert("RGB")
		x = eval_tf(img).unsqueeze(0).to(self.device)
		with torch.no_grad():
			probs = torch.softmax(self.model(x), dim=1)[0]
		idx = int(probs.argmax())
		return self.classes[idx], float(probs[idx])

	def is_sms(self, image_path, threshold=0.5):
		label, conf = self.predict(image_path)
		return label == "sms" and conf >= threshold


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest="cmd", required=True)
	t = sub.add_parser("train")
	t.add_argument("--data", required=True)
	t.add_argument("--epochs", type=int, default=10)
	t.add_argument("--batch-size", type=int, default=16)
	p = sub.add_parser("predict")
	p.add_argument("images", nargs="+")
	p.add_argument("--model", default=MODEL_PATH)
	args = parser.parse_args()

	if args.cmd == "train":
		train(args.data, args.epochs, args.batch_size)
	else:
		clf = SmsImageClassifier(args.model)
		for path in args.images:
			label, conf = clf.predict(path)
			print(f"{path}: {label} ({conf:.2%})")


if __name__ == "__main__":
	main()
