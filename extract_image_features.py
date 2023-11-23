import os
from PIL import Image
import argparse
import wandb
import torch
import yaml
import pandas as pd
from tqdm import tqdm
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet


# def extract_image_features(image_path, model, transform):
#     img = Image.open(image_path).convert('RGB')
#     img = transform(img).unsqueeze(0)

#     with torch.no_grad():
#         # Extract features from mid-level layers
#         features = model.extract_features(img)
#         ic(features.size())

#         # Concatenate features from different layers
#         features = torch.cat([features[0], features[3], features[6], features[9]], dim=1)

#         # Return features as a NumPy array
#         return features.squeeze().numpy()


def extract_image_features(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model.conv1(img)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
    return features.squeeze().numpy()


def get_image_features(model, transform, dataset='train'):

    # Path to the directory containing real estate announcements
    data_dir = f'./data/reduced_images_ILB/reduced_images/{dataset}'  # Change this to your actual data directory

    # Lists to store features and labels
    features_list = []

    # Iterate through each announcement directory
    for announcement_dir in tqdm(os.listdir(data_dir)):
        announcement_path = os.path.join(data_dir, announcement_dir)

        if os.path.isdir(announcement_path):
            # Extract tabular features based on the announcement directory name (for demonstration)
            # Replace this with your actual tabular feature extraction logic
            id_ann = [int(announcement_dir.split('_')[1])]

            # Extract image features and average over all images
            image_features = []
            for image_file in os.listdir(announcement_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(announcement_path, image_file)
                    image_feature = extract_image_features(image_path, model, transform)
                    image_features.append(image_feature)

            if image_features:
                average_image_feature = sum(image_features) / len(image_features)
                combined_features = id_ann + list(average_image_feature)
                features_list.append(combined_features)

    return pd.DataFrame(features_list, columns=['id_annonce', * [f'image_feature_{i}' for i in range(average_image_feature.size)]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-path', type=str, help='', default='./config_image_features.yml')
    parser.add_argument('--run-name', type=str, help='', default=None)

    args = vars(parser.parse_args())

    # Get config and params
    with open(args['config_path']) as file_:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        default_config = yaml.load(file_, Loader=yaml.FullLoader)

    wandb_config = {key: value for key, value in default_config.items()}

    wandb.login()

    run = wandb.init(
        # set the wandb project where this run will be logged
        project='ILB',
        tags=default_config['tags'],
        name=args['run_name'],
        # track hyperparameters and run metadata
        config=wandb_config
    )
    run_name = run.name
    config = run.config

    if config.get('model_name'):
        # Load pre-trained NoisyStudent (EfficientNet-B1)
        model = models.resnet18(weights="DEFAULT")

    model.eval()

    # Transformation for images
    transform = transforms.Compose([
        transforms.Resize(*config.get('im_size')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_features_df = get_image_features(model, transform, 'train')
    train_features_df.to_csv('./data/X_train_image_features.csv', index=False)
    artifact = wandb.Artifact(name="train_image_features", type="dataset")
    artifact.add_file(local_path='./data/X_train_image_features.csv')
    run.log_artifact(artifact)

    test_features_df = get_image_features(model, transform, 'test')
    test_features_df.to_csv('./data/X_test_image_features.csv', index=False)
    artifact = wandb.Artifact(name="test_image_features", type="dataset")
    artifact.add_file(local_path='./data/X_test_image_features.csv')
    run.log_artifact(artifact)

    run.log({'n_features': len(train_features_df.columns - 1)})

    run.finish()
