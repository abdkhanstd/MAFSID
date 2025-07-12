"""
few_shot_models.py - Few-Shot Learning Digital Twin for Intrusion Detection
Novel enhancements for network security with minimal training data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, deque
import random
from datetime import datetime
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for Few-Shot Intrusion Detection"""
    
    def __init__(self, input_dim=42, embedding_dim=128, num_classes=5):
        super(PrototypicalNetwork, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Feature embedding network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU()
        )
        
        # Attention mechanism for prototype refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim // 2,
            num_heads=8,
            dropout=0.1
        )
        
        # Temporal context encoder
        self.temporal_encoder = nn.LSTM(
            embedding_dim // 2, 
            embedding_dim // 4, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
    def forward(self, support_x, support_y=None, query_x=None):
        """Forward pass through the prototypical network"""
        if support_y is None and query_x is None:
            # Simple embedding mode
            embeddings = self.encoder(support_x)
            embeddings_att = embeddings.unsqueeze(1)
            attended, _ = self.attention(embeddings_att, embeddings_att, embeddings_att)
            return attended.squeeze(1)
        
        # Few-shot prediction mode
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Get embeddings
            support_embeddings = self.encoder(support_x)
            query_embeddings = self.encoder(query_x)
            
            # Apply attention
            support_att = support_embeddings.unsqueeze(1)
            support_attended, _ = self.attention(support_att, support_att, support_att)
            support_attended = support_attended.squeeze(1)
            
            query_att = query_embeddings.unsqueeze(1)
            query_attended, _ = self.attention(query_att, query_att, query_att)
            query_attended = query_attended.squeeze(1)
            
            # Compute prototypes for each class
            unique_labels = torch.unique(support_y)
            prototypes = []
            
            for label in unique_labels:
                mask = support_y == label
                if mask.sum() > 0:
                    prototype = support_attended[mask].mean(dim=0)
                    prototypes.append(prototype)
            
            if len(prototypes) == 0:
                return torch.zeros(query_x.size(0), self.num_classes).to(support_x.device)
            
            prototypes = torch.stack(prototypes)
            
            # Compute distances and convert to logits
            distances = torch.cdist(query_attended, prototypes)
            logits = -distances  # Negative distance as logits
            
            # Pad logits if we have fewer classes than expected
            if logits.size(1) < self.num_classes:
                padding = torch.full((logits.size(0), self.num_classes - logits.size(1)), 
                                   float('-inf')).to(logits.device)
                logits = torch.cat([logits, padding], dim=1)
            
            return logits
    
    def predict(self, support_x, support_y, query_x):
        """Predict labels for query samples based on support set"""
        self.eval()
        with torch.no_grad():
            # Use the forward method to get logits
            logits = self.forward(support_x, support_y, query_x)
            
            # Return predicted labels
            predictions = torch.argmax(logits, dim=1)
            return predictions
    
    def save(self, filepath):
        """Save model state"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model state"""
        self.load_state_dict(torch.load(filepath, weights_only=False))


class MetaLearningDigitalTwin(nn.Module):
    """Novel Meta-Learning Digital Twin for Few-Shot Intrusion Detection"""
    
    def __init__(self, input_dim=42, embedding_dim=128, adaptation_steps=5, num_classes=5):
        super(MetaLearningDigitalTwin, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.adaptation_steps = adaptation_steps
        self.num_classes = num_classes
        
        # Base feature extractor
        self.feature_extractor = PrototypicalNetwork(input_dim, embedding_dim, num_classes)
        
        # Meta-learner for fast adaptation
        self.meta_learner = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, num_classes)  # Multiclass output
        )
        
        # Novelty detection module
        self.novelty_detector = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Digital twin state memory
        self.prototype_memory = {}
        self.adaptation_memory = deque(maxlen=1000)
        
    def compute_prototypes(self, support_features, support_labels):
        """Compute class prototypes from support set"""
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            label_mask = (support_labels == label)
            label_features = support_features[label_mask]
            # Mean prototype with attention weighting
            prototype = torch.mean(label_features, dim=0)
            prototypes[label.item()] = prototype
            
        return prototypes
    
    def forward(self, support_x, support_y=None, query_x=None):
        """Forward pass with few-shot learning"""
        if support_y is None and query_x is None:
            # Simple embedding mode
            return self.feature_extractor(support_x)
        
        # Few-shot prediction mode
        query_embeddings = self.feature_extractor(support_x)  # Support_x is actually query in this context
        
        if query_x is not None:
            # We have actual support and query sets
            support_embeddings = self.feature_extractor(support_x)
            query_embeddings = self.feature_extractor(query_x)
            
            # Compute prototypes
            prototypes = self.compute_prototypes(support_embeddings, support_y)
            
            # Distance-based classification
            distances = []
            unique_labels = sorted(prototypes.keys())
            
            for label in unique_labels:
                prototype = prototypes[label]
                dist = torch.cdist(query_embeddings, prototype.unsqueeze(0))
                distances.append(-dist.squeeze())  # Negative distance for similarity
            
            if len(distances) == 0:
                return torch.zeros(query_x.size(0), self.num_classes).to(query_x.device)
            
            logits = torch.stack(distances, dim=1)
            
            # Meta-learning prediction
            meta_pred = self.meta_learner(query_embeddings)
            
            # Combine prototype-based and meta predictions
            combined_logits = 0.7 * logits + 0.3 * meta_pred
            
            # Pad logits if we have fewer classes than expected
            if combined_logits.size(1) < self.num_classes:
                padding = torch.full((combined_logits.size(0), self.num_classes - combined_logits.size(1)), 
                                   float('-inf')).to(combined_logits.device)
                combined_logits = torch.cat([combined_logits, padding], dim=1)
            
            return combined_logits
        else:
            # Standard inference
            logits = self.meta_learner(query_embeddings)
            return logits
    
    def predict(self, support_x, support_y, query_x):
        """Predict labels for query samples based on support set"""
        self.eval()
        with torch.no_grad():
            # Use meta-train mode for prediction
            outputs, _ = self.forward(query_x, support_x, support_y, mode='meta_train')
            predictions = torch.argmax(outputs, dim=1)
            return predictions
    
    def save(self, filepath):
        """Save model state"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model state"""
        self.load_state_dict(torch.load(filepath, weights_only=False))


class ContrastiveLearningModule(nn.Module):
    """Contrastive Learning for Enhanced Few-Shot Performance"""
    
    def __init__(self, embedding_dim=64, temperature=0.1):
        super(ContrastiveLearningModule, self).__init__()
        
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
    def forward(self, embeddings):
        """Apply projection for contrastive learning"""
        return F.normalize(self.projection_head(embeddings), dim=1)
    
    def contrastive_loss(self, embeddings, labels):
        """Compute contrastive loss"""
        projections = self.forward(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(positive_mask.size(0)).to(positive_mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        positive_sim = exp_sim * positive_mask
        negative_sim = exp_sim * negative_mask
        
        # InfoNCE loss
        positive_sum = positive_sim.sum(dim=1)
        negative_sum = negative_sim.sum(dim=1)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum + 1e-8))
        return loss.mean()


class FewShotDigitalTwinAgent(nn.Module):
    """Enhanced Few-Shot Learning Agent for Intrusion Detection"""
    
    def __init__(self, input_dim=42, n_way=5, k_shot=5, adaptation_lr=0.01, num_classes=5):
        super(FewShotDigitalTwinAgent, self).__init__()
        self.input_dim = input_dim
        self.n_way = n_way  # Number of classes
        self.k_shot = k_shot  # Shots per class
        self.adaptation_lr = adaptation_lr
        self.num_classes = num_classes
        
        # Initialize networks
        self.meta_twin = MetaLearningDigitalTwin(input_dim, num_classes=num_classes).to(device)
        self.contrastive_module = ContrastiveLearningModule().to(device)
        
        # Optimizers
        self.meta_optimizer = torch.optim.Adam(self.meta_twin.parameters(), lr=0.001)
        self.contrastive_optimizer = torch.optim.Adam(
            self.contrastive_module.parameters(), lr=0.001
        )
        
        # Novel enhancements
        self.prototype_bank = {}
        self.attack_memory = deque(maxlen=1000)
        self.adaptation_history = []
        
        # Metrics
        self.meta_accuracy = 0.0
        self.novelty_detection_rate = 0.0
        
    def forward(self, support_x, support_y=None, query_x=None):
        """Forward pass for compatibility with training script"""
        if support_y is None and query_x is None:
            # Simple forward for embeddings
            return self.meta_twin(support_x)
        else:
            # Few-shot forward
            return self.meta_twin(support_x, support_y, query_x)
    
    def train(self):
        """Put the model in training mode"""
        self.meta_twin.train()
        self.contrastive_module.train()
        
    def eval(self):
        """Put the model in evaluation mode"""
        self.meta_twin.eval()
        self.contrastive_module.eval()
        
    def predict(self, support_x, support_y, query_x):
        """Predict labels for query samples based on support set"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.meta_twin(query_x, support_x, support_y, mode='meta_train')
            predictions = torch.argmax(logits, dim=1)
            return predictions
    
    def create_episode(self, data_loader, episode_size=100):
        """Create few-shot learning episode"""
        all_data = []
        all_labels = []
        
        for batch_data, batch_labels in data_loader:
            all_data.append(batch_data)
            all_labels.append(batch_labels)
            
        features = torch.cat(all_data, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Sample episode
        episode_indices = random.sample(range(len(features)), min(episode_size, len(features)))
        episode_features = features[episode_indices]
        episode_labels = labels[episode_indices]
        
        # Split into support and query
        support_size = self.k_shot * self.n_way
        support_features = episode_features[:support_size]
        support_labels = episode_labels[:support_size]
        query_features = episode_features[support_size:]
        query_labels = episode_labels[support_size:]
        
        return (support_features.to(device), support_labels.to(device), 
                query_features.to(device), query_labels.to(device))
    
    def meta_train_step(self, support_features, support_labels, query_features, query_labels):
        """Single meta-training step with novel enhancements"""
        
        # Forward pass
        logits, query_embeddings = self.meta_twin(
            query_features, support_features, support_labels, mode='meta_train'
        )
        
        # Classification loss
        classification_loss = F.cross_entropy(logits, query_labels)
        
        # Contrastive loss for better embeddings
        contrastive_loss = self.contrastive_module.contrastive_loss(
            query_embeddings, query_labels
        )
        
        # Novel enhancement: Temporal consistency loss
        temporal_loss = self._compute_temporal_consistency_loss(query_embeddings)
        
        # Novel enhancement: Prototype diversity loss
        diversity_loss = self._compute_prototype_diversity_loss(support_features, support_labels)
        
        # Combined loss with novel components
        total_loss = (classification_loss + 
                     0.3 * contrastive_loss + 
                     0.1 * temporal_loss + 
                     0.1 * diversity_loss)
        
        # Backward pass
        self.meta_optimizer.zero_grad()
        self.contrastive_optimizer.zero_grad()
        total_loss.backward()
        self.meta_optimizer.step()
        self.contrastive_optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            accuracy = (pred == query_labels).float().mean()
            self.meta_accuracy = 0.9 * self.meta_accuracy + 0.1 * accuracy.item()
        
        return total_loss.item(), accuracy.item()
    
    def _compute_temporal_consistency_loss(self, embeddings):
        """Novel enhancement: Temporal consistency regularization"""
        if len(self.adaptation_history) < 2:
            return torch.tensor(0.0).to(device)
        
        # Compare current embeddings with recent history
        prev_embeddings = self.adaptation_history[-1]
        consistency_loss = F.mse_loss(
            embeddings.mean(dim=0), 
            prev_embeddings.mean(dim=0)
        )
        
        self.adaptation_history.append(embeddings.detach())
        if len(self.adaptation_history) > 10:
            self.adaptation_history.pop(0)
        
        return consistency_loss
    
    def _compute_prototype_diversity_loss(self, support_features, support_labels):
        """Novel enhancement: Encourage diverse prototypes"""
        embeddings = self.meta_twin.feature_extractor(support_features, return_embeddings=True)
        prototypes = self.meta_twin.compute_prototypes(embeddings, support_labels)
        
        if len(prototypes) < 2:
            return torch.tensor(0.0).to(device)
        
        # Compute pairwise distances between prototypes
        prototype_list = list(prototypes.values())
        diversity_loss = 0
        count = 0
        
        for i in range(len(prototype_list)):
            for j in range(i+1, len(prototype_list)):
                distance = F.pairwise_distance(
                    prototype_list[i].unsqueeze(0), 
                    prototype_list[j].unsqueeze(0)
                )
                diversity_loss += torch.exp(-distance)  # Encourage larger distances
                count += 1
        
        return diversity_loss / count if count > 0 else torch.tensor(0.0).to(device)
    
    def adapt_to_new_attacks(self, new_attack_data, new_attack_labels, adaptation_steps=5):
        """Novel enhancement: Fast adaptation to new attack types"""
        
        print(f"🔄 Adapting to new attack type with {len(new_attack_data)} samples...")
        
        # Create temporary optimizer for adaptation
        temp_optimizer = torch.optim.SGD(
            self.meta_twin.parameters(), 
            lr=self.adaptation_lr
        )
        
        # Few-shot adaptation
        for step in range(adaptation_steps):
            # Create mini support/query split
            perm = torch.randperm(len(new_attack_data))
            support_idx = perm[:len(perm)//2]
            query_idx = perm[len(perm)//2:]
            
            support_data = new_attack_data[support_idx].to(device)
            support_labels = new_attack_labels[support_idx].to(device)
            query_data = new_attack_data[query_idx].to(device)
            query_labels = new_attack_labels[query_idx].to(device)
            
            # Adaptation step
            logits, _ = self.meta_twin(
                query_data, support_data, support_labels, mode='meta_train'
            )
            loss = F.cross_entropy(logits, query_labels)
            
            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()
            
            print(f"   Adaptation step {step+1}: loss = {loss.item():.4f}")
        
        # Store in attack memory
        self.attack_memory.extend(zip(new_attack_data.cpu(), new_attack_labels.cpu()))
        
    def predict_intrusion(self, network_state):
        """Predict intrusion with few-shot capabilities"""
        self.meta_twin.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(network_state).unsqueeze(0).to(device)
            logits, novelty_score, embeddings = self.meta_twin(state_tensor, mode='inference')
            
            # Get prediction
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            
            # Check for novel attacks
            is_novel = novelty_score.item() > 0.7
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'is_novel_attack': is_novel,
            'novelty_score': novelty_score.item(),
            'embedding': embeddings.cpu().numpy()
        }
    
    def save_model(self, filepath):
        """Save the few-shot model"""
        torch.save({
            'meta_twin': self.meta_twin.state_dict(),
            'contrastive_module': self.contrastive_module.state_dict(),
            'prototype_bank': self.prototype_bank,
            'meta_accuracy': self.meta_accuracy
        }, filepath)
    
    def load_model(self, filepath):
        """Load the few-shot model"""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.meta_twin.load_state_dict(checkpoint['meta_twin'])
        self.contrastive_module.load_state_dict(checkpoint['contrastive_module'])
        self.prototype_bank = checkpoint['prototype_bank']
        self.meta_accuracy = checkpoint.get('meta_accuracy', 0.0)


class FewShotDataLoader:
    """Custom data loader for few-shot episodes"""
    
    def __init__(self, features, labels, n_way=2, k_shot=5, q_queries=15, num_episodes=1000):
        self.features = features
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_episodes = num_episodes
        
        # Group data by class
        self.class_data = defaultdict(list)
        for i, label in enumerate(labels):
            self.class_data[label].append(i)
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self._sample_episode()
    
    def __len__(self):
        return self.num_episodes
    
    def _sample_episode(self):
        """Sample a few-shot episode"""
        # Sample classes
        available_classes = list(self.class_data.keys())
        episode_classes = random.sample(available_classes, self.n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for i, cls in enumerate(episode_classes):
            # Sample indices for this class
            cls_indices = random.sample(
                self.class_data[cls], 
                self.k_shot + self.q_queries
            )
            
            # Split into support and query
            support_indices = cls_indices[:self.k_shot]
            query_indices = cls_indices[self.k_shot:]
            
            # Add to episode
            for idx in support_indices:
                support_data.append(self.features[idx])
                support_labels.append(i)  # Use episode-specific label
            
            for idx in query_indices:
                query_data.append(self.features[idx])
                query_labels.append(i)
        
        return (torch.stack(support_data), torch.tensor(support_labels),
                torch.stack(query_data), torch.tensor(query_labels))


# Novel enhancement: Attack Evolution Tracker
class AttackEvolutionTracker:
    """Track and adapt to evolving attack patterns"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.attack_history = deque(maxlen=window_size)
        self.pattern_drift_threshold = 0.3
        
    def update(self, attack_embedding, attack_type):
        """Update with new attack observation"""
        self.attack_history.append({
            'embedding': attack_embedding,
            'type': attack_type,
            'timestamp': datetime.now()
        })
    
    def detect_pattern_drift(self):
        """Detect if attack patterns have evolved"""
        if len(self.attack_history) < self.window_size:
            return False
        
        # Compare recent vs older embeddings
        recent = [item['embedding'] for item in list(self.attack_history)[-20:]]
        older = [item['embedding'] for item in list(self.attack_history)[:20]]
        
        recent_center = np.mean(recent, axis=0)
        older_center = np.mean(older, axis=0)
        
        drift = np.linalg.norm(recent_center - older_center)
        
        return drift > self.pattern_drift_threshold
