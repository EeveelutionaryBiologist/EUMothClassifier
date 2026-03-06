
import torch
from pathlib import Path
import shutil

from data_loader import create_dataloaders
from model import create_effnetb2_model, create_vit_model
from engine import train

MODEL_PATH = Path("App") / "Model"
MODEL_FILE = MODEL_PATH / "vitb16_eu_moths.pth"
EPOCHS = 15

# Set device 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Initialize model (-> from model.py module)
    model, transforms = create_vit_model()

    # Setup data loader
    train_dataloader, test_dataloader, class_names = create_dataloaders(
        data_dir=Path("eu-moths-dataset") / "images",
        transform=transforms,
    )

    # Loss and Optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())

    # Training of the model
    results = train(
        model=model.to(DEVICE),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
    )

    # Write out class names
    with open("class_names.txt", 'w') as f:
        for label in class_names:
            f.write("%s\n" % label.capitalize().replace('_', ' '))

    shutil.copy("class_names.txt", "App/")
        
    # Save model state dict
    MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)

    print(f"\n[INFO] Saving model to: {MODEL_FILE}")
    torch.save(obj=model.state_dict(),
            f=MODEL_FILE)


if __name__ == "__main__":
    main()

