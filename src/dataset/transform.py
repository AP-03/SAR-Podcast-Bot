from torchvision import transforms

def get_basic_transforms():
    return transforms.Compose([
        transforms.ToTensor(),  # HWC -> CHW, scales to [0,1]
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
