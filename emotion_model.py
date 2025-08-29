"""
RAVDESS video emotion recognition
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import json
from tqdm import tqdm
import logging
import random

# Import our video dataset
from ravdess_video_dataset import RAVDESSVideoDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class VideoEmotionNet(nn.Module):
    """Multimodal CNN/RNN network for video emotion recognition"""
    
    def __init__(self, audio_feature_dim, video_feature_dim, num_emotions=8, 
                 hidden_dim=256, num_layers=2, dropout=0.3):
        super(VideoEmotionNet, self).__init__()
        
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        
        # Audio processing branch
        self.audio_cnn = nn.Sequential(
            nn.Linear(audio_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        # Video processing branch
        self.video_cnn = nn.Sequential(
            nn.Linear(video_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64 + 64,  # Combined audio + video features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_emotions)
        )
    
    def forward(self, audio_features, video_features):
        batch_size = audio_features.size(0)
        
        # Process audio and video features
        audio_out = self.audio_cnn(audio_features)
        video_out = self.video_cnn(video_features)
        
        # Combine features
        combined_features = torch.cat([audio_out, video_out], dim=1)
        
        # Add sequence dimension for LSTM
        combined_features = combined_features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Final classification
        logits = self.classifier(attended_features)
        
        return logits, attention_weights

def train_video_emotion_model():
    """Main training function for video emotion recognition"""
    
    logger.info("Starting RAVDESS video emotion recognition training")
    
    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading video dataset...")
    dataset = RAVDESSVideoDataset()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Get feature dimensions
    sample = dataset[0]
    audio_dim = sample['audio_features'].shape[0]
    video_dim = sample['video_features'].shape[0]
    num_emotions = len(dataset.emotion_to_idx)
    
    logger.info(f"Audio features: {audio_dim}, Video features: {video_dim}, Emotions: {num_emotions}")
    
    # Create model
    model = VideoEmotionNet(
        audio_feature_dim=audio_dim,
        video_feature_dim=video_dim,
        num_emotions=num_emotions,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            audio_features = batch['audio_features'].to(device)
            video_features = batch['video_features'].to(device)
            emotions = batch['emotion'].squeeze().to(device)
            
            optimizer.zero_grad()
            logits, attention_weights = model(audio_features, video_features)
            loss = criterion(logits, emotions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == emotions).sum().item()
            total_samples += emotions.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio_features = batch['audio_features'].to(device)
                video_features = batch['video_features'].to(device)
                emotions = batch['emotion'].squeeze().to(device)
                
                logits, attention_weights = model(audio_features, video_features)
                loss = criterion(logits, emotions)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == emotions).sum().item()
                val_total += emotions.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(emotions.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'emotion_labels': dataset.idx_to_emotion,
                'audio_feature_dim': audio_dim,
                'video_feature_dim': video_dim,
                'best_accuracy': best_accuracy,
                'epoch': epoch
            }, f'models/video_emotion_model_acc_{best_accuracy:.4f}.pth')
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    # Generate final evaluation
    logger.info("\nFinal Model Evaluation:")
    
    # Classification report
    class_names = [dataset.idx_to_emotion[i] for i in range(len(dataset.idx_to_emotion))]
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Training curves
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax1.plot(val_losses, label='Validation Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy', marker='o')
    ax2.plot(val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/video_emotion_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'final_accuracy': accuracy_score(all_labels, all_predictions),
        'best_accuracy': best_accuracy,
        'classification_report': report,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        },
        'config': config
    }
    
    with open('results/video_emotion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    logger.info("Results saved to results/video_emotion_training_results.png")
    logger.info("Detailed results saved to results/video_emotion_results.json")
    
    return model, best_accuracy

if __name__ == "__main__":
    model, accuracy = train_video_emotion_model()
    print(f"\n🎉 Video emotion recognition training completed!")
    print(f"Best accuracy: {accuracy:.4f}")
    print(f"Model saved in models/ directory")