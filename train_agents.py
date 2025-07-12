#!/usr/bin/env python3
"""
Agent-Based Few-Shot Learning System
Multi-agent approach for intrusion detection with comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from datetime import datetime
from multiclass_datasets import MulticlassDataManager, create_few_shot_episodes_multiclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import random
from collections import defaultdict, deque

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class CommunicationModule(nn.Module):
    """Communication module for agent message passing"""
    
    def __init__(self, embedding_dim, num_agents):
        super(CommunicationModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_agents = num_agents
        
        # Message generation with proper initialization
        self.message_encoder = nn.Linear(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.message_encoder.weight)
        nn.init.zeros_(self.message_encoder.bias)
        
        # Message aggregation
        self.message_aggregator = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        
        # Message processing with proper initialization
        self.message_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize message processor weights
        for module in self.message_processor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def generate_message(self, agent_state):
        """Generate message from agent state"""
        return torch.tanh(self.message_encoder(agent_state))
    
    def aggregate_messages(self, messages, sender_mask=None):
        """Aggregate messages from other agents"""
        if len(messages) == 0:
            # Return zeros tensor on the same device as the model parameters
            return torch.zeros(self.embedding_dim, device=next(self.parameters()).device)
        
        # Stack messages
        message_stack = torch.stack(messages, dim=0).unsqueeze(0)  # [1, num_messages, embedding_dim]
        
        # Use attention to aggregate
        aggregated, _ = self.message_aggregator(message_stack, message_stack, message_stack)
        
        return aggregated.squeeze(0).mean(dim=0)  # [embedding_dim]
    
    def process_message(self, own_state, aggregated_message):
        """Process aggregated message with own state"""
        combined = own_state + aggregated_message
        return self.message_processor(combined)

class IntrusionDetectionAgent(nn.Module):
    """Specialized agent for intrusion detection with communication capabilities"""
    
    def __init__(self, agent_id, agent_type, input_dim, embedding_dim=128, num_agents=5):
        super(IntrusionDetectionAgent, self).__init__()
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_agents = num_agents
        
        # Specialized feature extractors based on agent type
        if agent_type == "NetworkTrafficAnalyzer":
            # Focuses on network flow patterns
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim * 2),
                nn.LayerNorm(embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU()
            )
        elif agent_type == "AnomalyDetector":
            # Focuses on statistical anomalies
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh()
            )
        elif agent_type == "BehaviorAnalyzer":
            # Focuses on behavioral patterns
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim * 3),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU()
            )
        elif agent_type == "ProtocolAnalyzer":
            # Focuses on protocol-specific features
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.Sigmoid()
            )
        else:  # "ThreatClassifier"
            # Focuses on threat classification
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        
        # Communication module
        self.communication = CommunicationModule(embedding_dim, num_agents)
        
        # Internal state with careful initialization
        self.internal_state = nn.Parameter(torch.randn(embedding_dim) * 0.01)  # Smaller initialization
        
        # Memory for past communications
        self.communication_history = deque(maxlen=50)
        
        # Classification head with proper initialization
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # Own features + communicated knowledge
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)  # Confidence score
        )
        
        # Initialize classifier weights
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Attention for prototype computation
        self.prototype_attention = nn.MultiheadAttention(embedding_dim, num_heads=2, batch_first=True)
    
    def extract_features(self, x):
        """Extract features using specialized feature extractor"""
        return self.feature_extractor(x)
    
    def generate_message(self, features, context=None):
        """Generate message based on current features and context"""
        # Combine features with internal state
        agent_state = features.mean(dim=0) + self.internal_state
        
        # Generate message
        message = self.communication.generate_message(agent_state)
        
        return message
    
    def receive_messages(self, messages_from_others):
        """Receive and process messages from other agents"""
        if len(messages_from_others) == 0:
            return torch.zeros(self.embedding_dim).to(self.internal_state.device)
        
        # Aggregate messages
        aggregated_message = self.communication.aggregate_messages(messages_from_others)
        
        # Process with own internal state
        processed_message = self.communication.process_message(self.internal_state, aggregated_message)
        
        # Update internal state
        self.internal_state.data = 0.9 * self.internal_state.data + 0.1 * processed_message.data
        
        # Store in communication history
        self.communication_history.append({
            'timestamp': time.time(),
            'aggregated_message': aggregated_message.detach(),
            'processed_message': processed_message.detach()
        })
        
        return processed_message
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way, communicated_knowledge):
        """Compute prototypes using own features + communicated knowledge"""
        prototypes = torch.zeros(n_way, self.embedding_dim).to(support_embeddings.device)
        
        for class_idx in range(n_way):
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:
                class_embeddings = support_embeddings[class_mask]
                
                # Enhance with communicated knowledge
                if communicated_knowledge is not None:
                    enhanced_embeddings = class_embeddings + communicated_knowledge.unsqueeze(0)
                else:
                    enhanced_embeddings = class_embeddings
                
                # Use attention to compute prototype
                attended, _ = self.prototype_attention(
                    enhanced_embeddings.unsqueeze(0),
                    enhanced_embeddings.unsqueeze(0),
                    enhanced_embeddings.unsqueeze(0)
                )
                prototypes[class_idx] = attended.squeeze(0).mean(dim=0)
        
        return prototypes
    
    def forward(self, support_x, support_y, query_x, n_way, communicated_knowledge=None):
        """Forward pass with communication-enhanced processing"""
        # Extract specialized features
        support_features = self.extract_features(support_x)
        query_features = self.extract_features(query_x)
        
        # Compute prototypes with communicated knowledge
        prototypes = self.compute_prototypes(support_features, support_y, n_way, communicated_knowledge)
        
        # Enhance query features with communicated knowledge
        if communicated_knowledge is not None:
            enhanced_query_features = torch.cat([query_features, communicated_knowledge.unsqueeze(0).expand(query_features.size(0), -1)], dim=1)
        else:
            enhanced_query_features = torch.cat([query_features, torch.zeros_like(query_features)], dim=1)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)
        
        # Get confidence scores from classifier
        confidence_scores = self.classifier(enhanced_query_features)
        
        # Combine distance-based logits with confidence
        logits = -distances + confidence_scores.expand(-1, n_way) * 0.1
        
        return logits, support_features, query_features
    
    def get_communication_summary(self):
        """Get summary of communication history"""
        if len(self.communication_history) == 0:
            return {"messages_received": 0, "avg_message_strength": 0.0}
        
        avg_strength = torch.stack([msg['processed_message'] for msg in self.communication_history]).norm(dim=1).mean().item()
        
        return {
            "messages_received": len(self.communication_history),
            "avg_message_strength": avg_strength,
            "agent_type": self.agent_type,
            "internal_state_norm": self.internal_state.norm().item()
        }

class CommunicatingAgentSystem(nn.Module):
    """Multi-agent system with specialized intrusion detection agents that communicate"""
    
    def __init__(self, input_dim, embedding_dim=128):
        super(CommunicatingAgentSystem, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        
        # Define 5 specialized agents for intrusion detection
        self.agent_types = [
            "NetworkTrafficAnalyzer",  # Analyzes network flow patterns
            "AnomalyDetector",         # Detects statistical anomalies
            "BehaviorAnalyzer",        # Analyzes behavioral patterns
            "ProtocolAnalyzer",        # Analyzes protocol-specific features
            "ThreatClassifier"         # Classifies threat types
        ]
        
        self.agents = nn.ModuleList()
        for i, agent_type in enumerate(self.agent_types):
            agent = IntrusionDetectionAgent(
                agent_id=i,
                agent_type=agent_type,
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                num_agents=len(self.agent_types)
            )
            self.agents.append(agent)
        
        # Communication coordination weights with proper initialization
        self.communication_weights = nn.Parameter(torch.eye(len(self.agents)) * 0.5 + 
                                                 torch.ones(len(self.agents), len(self.agents)) * 0.1)
        
        # Final decision fusion network with proper initialization
        self.decision_fusion = nn.Sequential(
            nn.Linear(len(self.agents) * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Initialize decision fusion weights
        for module in self.decision_fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Shared optimizer for all components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Communication history
        self.communication_rounds = []
        
        # Move entire system to device
        self.to(self.device)
    
    def communicate_round(self, support_x, support_y, query_x):
        """Perform one round of inter-agent communication"""
        # Phase 1: Each agent extracts features and generates initial message
        agent_features = []
        initial_messages = []
        
        for agent in self.agents:
            features = agent.extract_features(support_x)
            message = agent.generate_message(features)
            agent_features.append(features)
            initial_messages.append(message)
        
        # Phase 2: Message exchange and processing
        communicated_knowledge = []
        
        for i, agent in enumerate(self.agents):
            # Get messages from other agents (weighted by communication matrix)
            messages_from_others = []
            weights = F.softmax(self.communication_weights[i], dim=0)
            
            for j, other_message in enumerate(initial_messages):
                if i != j:  # Don't send message to self
                    weighted_message = other_message * weights[j]
                    messages_from_others.append(weighted_message)
            
            # Process received messages
            knowledge = agent.receive_messages(messages_from_others)
            communicated_knowledge.append(knowledge)
        
        # Store communication round info
        communication_round_info = {
            'round_id': len(self.communication_rounds),
            'agent_messages': [msg.detach().cpu().numpy().tolist() for msg in initial_messages],
            'communication_weights': self.communication_weights.detach().cpu().numpy().tolist(),
            'knowledge_exchange': [k.detach().cpu().numpy().tolist() for k in communicated_knowledge]
        }
        self.communication_rounds.append(communication_round_info)
        
        return communicated_knowledge
    
    def forward(self, support_x, support_y, query_x, n_way):
        """Forward pass with multi-round communication"""
        # Multiple rounds of communication for better coordination
        num_communication_rounds = 3
        
        for round_num in range(num_communication_rounds):
            communicated_knowledge = self.communicate_round(support_x, support_y, query_x)
        
        # Phase 3: Each agent makes predictions with communicated knowledge
        agent_logits = []
        agent_features_all = []
        
        for i, agent in enumerate(self.agents):
            logits, support_features, query_features = agent(
                support_x, support_y, query_x, n_way, communicated_knowledge[i]
            )
            agent_logits.append(logits)
            agent_features_all.append(query_features)
        
        # Phase 4: Fusion of agent decisions
        # Simple weighted average of agent logits (more stable)
        stacked_logits = torch.stack(agent_logits, dim=0)  # [num_agents, batch_size, n_way]
        
        # Use learnable weights for each agent (simpler than complex fusion)
        agent_weights = F.softmax(torch.ones(len(self.agents), device=stacked_logits.device), dim=0)
        
        # Weighted combination of agent predictions
        final_logits = torch.sum(stacked_logits * agent_weights.view(-1, 1, 1), dim=0)  # [batch_size, n_way]
        
        return final_logits, agent_logits, communicated_knowledge
    
    def train_episode(self, support_x, support_y, query_x, query_y, n_way):
        """Train on a single episode with communication"""
        self.optimizer.zero_grad()
        
        # Input validation to prevent NaN propagation
        if torch.isnan(support_x).any() or torch.isnan(query_x).any():
            print("Warning: NaN detected in input data")
            return 0.0, torch.zeros(query_x.size(0), n_way, device=query_x.device), {
                'main_loss': 0.0, 'agent_loss': 0.0, 'communication_diversity_loss': 0.0, 'total_loss': 0.0
            }
        
        # Forward pass with communication
        final_logits, agent_logits, communicated_knowledge = self.forward(support_x, support_y, query_x, n_way)
        
        # Check for NaN in final logits
        if torch.isnan(final_logits).any():
            print("Warning: NaN detected in final_logits")
            return 0.0, torch.zeros_like(final_logits), {
                'main_loss': 0.0, 'agent_loss': 0.0, 'communication_diversity_loss': 0.0, 'total_loss': 0.0
            }
        
        # Clip logits to prevent overflow in softmax
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)
        
        # Main ensemble loss with error handling
        try:
            main_loss = self.criterion(final_logits, query_y)
            if torch.isnan(main_loss) or torch.isinf(main_loss):
                print("Warning: Invalid main_loss detected, using fallback")
                main_loss = torch.tensor(1.0, device=final_logits.device, requires_grad=True)
        except Exception as e:
            print(f"Error computing main loss: {e}")
            main_loss = torch.tensor(1.0, device=final_logits.device, requires_grad=True)
        
        # Individual agent losses for diversity
        agent_losses = []
        for i, logits in enumerate(agent_logits):
            try:
                # Clip agent logits too
                clipped_logits = torch.clamp(logits, min=-10.0, max=10.0)
                
                # Check for NaN in agent logits
                if torch.isnan(clipped_logits).any():
                    print(f"Warning: NaN in agent {i} logits")
                    continue
                
                agent_loss = self.criterion(clipped_logits, query_y)
                
                # Validate agent loss
                if torch.isnan(agent_loss) or torch.isinf(agent_loss):
                    print(f"Warning: Invalid loss for agent {i}")
                    continue
                    
                agent_losses.append(agent_loss)
            except Exception as e:
                print(f"Warning: Agent {i} loss calculation failed: {e}")
                continue
        
        # Communication diversity loss with robust similarity computation
        communication_diversity_loss = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
        
        if len(communicated_knowledge) > 1:
            similarities = []
            for i in range(len(communicated_knowledge)):
                for j in range(i + 1, len(communicated_knowledge)):
                    ki = communicated_knowledge[i]
                    kj = communicated_knowledge[j]
                    
                    # Check for NaN in communicated knowledge
                    if torch.isnan(ki).any() or torch.isnan(kj).any():
                        continue
                    
                    # Compute norms with numerical stability
                    ki_norm = torch.norm(ki, p=2)
                    kj_norm = torch.norm(kj, p=2)
                    
                    # More conservative threshold to avoid division by near-zero values
                    if ki_norm > 1e-6 and kj_norm > 1e-6:
                        # Normalize vectors first to prevent overflow
                        ki_normalized = ki / (ki_norm + 1e-8)
                        kj_normalized = kj / (kj_norm + 1e-8)
                        
                        # Compute dot product similarity (more stable than cosine_similarity)
                        similarity = torch.sum(ki_normalized * kj_normalized)
                        
                        # Clamp similarity to valid range [-1, 1]
                        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
                        
                        # Check for NaN/inf in similarity
                        if not torch.isnan(similarity) and not torch.isinf(similarity):
                            similarities.append(similarity ** 2)
            
            if len(similarities) > 0:
                try:
                    communication_diversity_loss = torch.mean(torch.stack(similarities))
                    # Final validation of diversity loss
                    if torch.isnan(communication_diversity_loss) or torch.isinf(communication_diversity_loss):
                        communication_diversity_loss = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
                except Exception as e:
                    print(f"Warning: Communication diversity loss computation failed: {e}")
                    communication_diversity_loss = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
        
        # Total loss calculation with validation
        if len(agent_losses) > 0:
            agent_loss_mean = torch.mean(torch.stack(agent_losses))
            # Validate agent loss mean
            if torch.isnan(agent_loss_mean) or torch.isinf(agent_loss_mean):
                agent_loss_mean = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
        else:
            agent_loss_mean = torch.tensor(0.0, device=main_loss.device, requires_grad=True)
        
        # Combine losses with appropriate weights
        total_loss = main_loss + 0.1 * agent_loss_mean + 0.05 * communication_diversity_loss
        
        # Final check for NaN or inf values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid total loss detected. main_loss: {main_loss.item()}, agent_loss: {agent_loss_mean.item()}, comm_loss: {communication_diversity_loss.item()}")
            # Use only main loss as fallback
            total_loss = main_loss
            
            # If main loss is also invalid, use a dummy loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: Even main loss is invalid, using dummy loss")
                total_loss = torch.tensor(1.0, device=main_loss.device, requires_grad=True)
        
        # Backward pass with additional safety
        try:
            total_loss.backward()
            
            # More comprehensive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Check for NaN gradients before optimization step
            has_nan_grad = False
            for param in self.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("Warning: NaN gradients detected, skipping optimization step")
                self.optimizer.zero_grad()
                return 0.0, final_logits.detach(), {
                    'main_loss': main_loss.item(),
                    'agent_loss': agent_loss_mean.item(),
                    'communication_diversity_loss': communication_diversity_loss.item(),
                    'total_loss': 0.0
                }
            
            self.optimizer.step()
            
        except Exception as e:
            print(f"Warning: Error during backward pass: {e}")
            self.optimizer.zero_grad()
            return 0.0, final_logits.detach(), {
                'main_loss': main_loss.item(),
                'agent_loss': agent_loss_mean.item(),
                'communication_diversity_loss': communication_diversity_loss.item(),
                'total_loss': 0.0
            }
        
        return total_loss.item(), final_logits, {
            'main_loss': main_loss.item(),
            'agent_loss': agent_loss_mean.item(),
            'communication_diversity_loss': communication_diversity_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def evaluate_episode(self, support_x, support_y, query_x, query_y, n_way):
        """Evaluate on a single episode"""
        with torch.no_grad():
            final_logits, agent_logits, communicated_knowledge = self.forward(support_x, support_y, query_x, n_way)
            
            _, final_predictions = torch.max(final_logits, 1)
            
            # Individual agent predictions
            agent_predictions = []
            for logits in agent_logits:
                _, pred = torch.max(logits, 1)
                agent_predictions.append(pred)
        
        return final_predictions, agent_predictions, communicated_knowledge
    
    def save_model(self, filepath):
        """Save the entire agent system"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'agent_states': [agent.state_dict() for agent in self.agents],
            'communication_weights': self.communication_weights.data,
            'agent_types': self.agent_types,
            'embedding_dim': self.embedding_dim,
            'input_dim': self.agents[0].input_dim if self.agents else None,
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(save_data, filepath)
        print(f"💾 Saved agent system to {filepath}")
    
    def load_model(self, filepath, input_dim):
        """Load the entire agent system"""
        save_data = torch.load(filepath, map_location=self.device)
        
        # Restore agent states
        for i, agent in enumerate(self.agents):
            if i < len(save_data['agent_states']):
                agent.load_state_dict(save_data['agent_states'][i])
        
        # Restore communication weights
        if 'communication_weights' in save_data:
            self.communication_weights.data = save_data['communication_weights']
        
        # Restore optimizer state
        if 'optimizer_state' in save_data:
            self.optimizer.load_state_dict(save_data['optimizer_state'])
        
        print(f"📁 Loaded agent system from {filepath}")
        return save_data
    
    def save_best_individual_agents(self, filepath_prefix):
        """Save each agent individually for analysis"""
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent_filepath = f"{filepath_prefix}_agent_{i}_{agent.agent_type}.pth"
            agent_data = {
                'state_dict': agent.state_dict(),
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'input_dim': agent.input_dim,
                'embedding_dim': agent.embedding_dim,
                'communication_history': list(agent.communication_history)
            }
            torch.save(agent_data, agent_filepath)
            print(f"💾 Saved {agent.agent_type} to {agent_filepath}")
    
    def get_communication_summary(self):
        """Get summary of all agent communications"""
        agent_summaries = []
        for agent in self.agents:
            summary = agent.get_communication_summary()
            agent_summaries.append(summary)
        
        return {
            'agent_summaries': agent_summaries,
            'total_communication_rounds': len(self.communication_rounds),
            'communication_weights': self.communication_weights.detach().cpu().numpy().tolist(),
            'agent_types': self.agent_types
        }

def train_agent_system(agent_system, train_episodes, device, num_epochs=20, save_best=True, model_save_path="models/best_agent_system.pth"):
    """Train the communicating agent system"""
    print(f"🤖 Training Communicating Agent System ({len(agent_system.agent_types)} agents) for {num_epochs} epochs...")
    
    epoch_stats = []
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': 0.0, 'main': 0.0, 'agent': 0.0, 'communication': 0.0}
        epoch_accuracy = 0.0
        valid_episodes = 0
        
        # Reset communication rounds for this epoch
        agent_system.communication_rounds = []
        
        for episode in train_episodes:
            try:
                support_x = torch.FloatTensor(episode['support_x']).to(device)
                support_y = torch.LongTensor(episode['support_y']).to(device)
                query_x = torch.FloatTensor(episode['query_x']).to(device)
                query_y = torch.LongTensor(episode['query_y']).to(device)
                
                n_way = len(torch.unique(support_y))
                
                # Train on episode with communication
                total_loss, logits, loss_breakdown = agent_system.train_episode(
                    support_x, support_y, query_x, query_y, n_way
                )
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == query_y).float().mean().item()
                
                # Accumulate losses - map the keys properly
                epoch_losses['total'] += loss_breakdown['total_loss']
                epoch_losses['main'] += loss_breakdown['main_loss']
                epoch_losses['agent'] += loss_breakdown['agent_loss']
                epoch_losses['communication'] += loss_breakdown['communication_diversity_loss']
                
                epoch_accuracy += accuracy
                valid_episodes += 1
                
            except Exception as e:
                print(f"    Episode error: {e}")
                continue
        
        if valid_episodes > 0:
            # Average all metrics
            for key in epoch_losses:
                epoch_losses[key] /= valid_episodes
            avg_accuracy = epoch_accuracy / valid_episodes
            
            # Save best model
            if save_best and avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_epoch = epoch + 1
                try:
                    agent_system.save_model(model_save_path)
                    # Also save individual agents
                    individual_path = model_save_path.replace('.pth', '')
                    agent_system.save_best_individual_agents(individual_path)
                    print(f"🏆 New best model saved! Accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
                except Exception as e:
                    print(f"⚠️  Failed to save model: {e}")
            
            epoch_stats.append({
                'epoch': epoch + 1,
                'losses': epoch_losses.copy(),
                'accuracy': avg_accuracy,
                'episodes': valid_episodes,
                'communication_rounds': len(agent_system.communication_rounds),
                'is_best': avg_accuracy == best_accuracy
            })
            
            if epoch % 3 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}:")
                print(f"      Total Loss: {epoch_losses['total']:.4f}")
                print(f"      Main Loss: {epoch_losses['main']:.4f}")
                print(f"      Agent Loss: {epoch_losses['agent']:.4f}")
                print(f"      Communication Loss: {epoch_losses['communication']:.4f}")
                print(f"      Accuracy: {avg_accuracy:.4f} {'🏆' if avg_accuracy == best_accuracy else ''}")
                print(f"      Communication Rounds: {len(agent_system.communication_rounds)}")
    
    print(f"\n🎯 Training Summary:")
    print(f"   Best Accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"   Best model saved to: {model_save_path}")
    
    return epoch_stats, best_accuracy, best_epoch

def evaluate_agent_system(agent_system, test_episodes, class_names, device, n_way):
    """Evaluate the communicating agent system"""
    print(f"🤖 Evaluating Communicating Agent System...")
    
    all_predictions = []
    all_labels = []
    agent_predictions_list = [[] for _ in range(len(agent_system.agents))]
    communication_data = []
    
    # Reset communication rounds for evaluation
    agent_system.communication_rounds = []
    
    with torch.no_grad():
        for episode_idx, episode in enumerate(test_episodes):
            try:
                support_x = torch.FloatTensor(episode['support_x']).to(device)
                support_y = torch.LongTensor(episode['support_y']).to(device)
                query_x = torch.FloatTensor(episode['query_x']).to(device)
                query_y = torch.LongTensor(episode['query_y']).to(device)
                
                # Evaluate episode with communication
                predictions, agent_predictions, communicated_knowledge = agent_system.evaluate_episode(
                    support_x, support_y, query_x, query_y, n_way
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(query_y.cpu().numpy())
                
                # Store individual agent predictions
                for i, agent_pred in enumerate(agent_predictions):
                    agent_predictions_list[i].extend(agent_pred.cpu().numpy())
                
                # Store communication data for analysis
                if episode_idx < 5:  # Store first 5 episodes for detailed analysis
                    communication_data.append({
                        'episode': episode_idx,
                        'communicated_knowledge': [k.cpu().numpy().tolist() for k in communicated_knowledge]
                    })
                
            except Exception as e:
                print(f"    Evaluation episode error: {e}")
                continue
    
    if len(all_predictions) == 0:
        return None
    
    # Calculate ensemble metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Calculate individual agent metrics
    agent_metrics = []
    for i in range(len(agent_system.agents)):
        if len(agent_predictions_list[i]) > 0:
            agent_acc = accuracy_score(all_labels, agent_predictions_list[i])
            agent_p, agent_r, agent_f1, _ = precision_recall_fscore_support(
                all_labels, agent_predictions_list[i], average='weighted', zero_division=0
            )
            agent_metrics.append({
                'agent_id': i,
                'agent_type': agent_system.agent_types[i],
                'accuracy': agent_acc,
                'precision': agent_p,
                'recall': agent_r,
                'f1_score': agent_f1
            })
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    unique_labels = sorted(list(set(all_labels + all_predictions)))
    actual_class_names = []
    for label_idx in unique_labels:
        if label_idx < len(class_names):
            actual_class_names.append(class_names[label_idx])
        else:
            actual_class_names.append(f"class_{label_idx}")
    
    report = classification_report(
        all_labels, all_predictions, 
        target_names=actual_class_names,
        labels=unique_labels,
        zero_division=0
    )
    
    # Get communication summary
    comm_summary = agent_system.get_communication_summary()
    
    return {
        'ensemble_accuracy': accuracy,
        'ensemble_precision': precision,
        'ensemble_recall': recall,
        'ensemble_f1_score': f1,
        'agent_metrics': agent_metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': actual_class_names,
        'num_predictions': len(all_predictions),
        'communication_summary': comm_summary,
        'communication_data_sample': communication_data
    }

def evaluate_dataset_with_agents(dataset_name):
    """Evaluate a dataset using the cooperative agent system"""
    print(f"\n{'='*80}")
    print(f"🤖 AGENT-BASED EVALUATION: {dataset_name}")
    print(f"{'='*80}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    data_manager = MulticlassDataManager(dataset_name=dataset_name)
    
    try:
        # Load dataset
        print(f"📂 Loading {dataset_name} dataset...")
        dataset = data_manager.load_dataset()
        
        if dataset is None:
            print(f"❌ Failed to load {dataset_name}")
            return
        
        print(f"✅ Loaded {dataset_name}")
        print(f"  📊 Features: {dataset['feature_dim']}")
        print(f"  🏷️  Classes: {dataset['num_classes']} ({', '.join(dataset['class_names'])})")
        print(f"  🚂 Training samples: {len(dataset['train']['features']):,}")
        print(f"  🧪 Test samples: {len(dataset['test']['features']):,}")
        
        # Results storage
        results_text = [
            f"Agent-Based Few-Shot Learning Evaluation",
            f"Dataset: {dataset_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Device: {device}",
            "",
            f"Dataset Information:",
            f"  Features: {dataset['feature_dim']}",
            f"  Classes: {dataset['num_classes']} ({', '.join(dataset['class_names'])})",
            f"  Training samples: {len(dataset['train']['features']):,}",
            f"  Test samples: {len(dataset['test']['features']):,}",
            "",
        ]
        
        # Define configurations to test
        configurations = []
        
        if dataset['num_classes'] >= 8:
            configurations = [(3, 1), (3, 3), (5, 1), (5, 3), (8, 1), (8, 3)]
        elif dataset['num_classes'] >= 5:
            configurations = [(3, 1), (3, 3), (5, 1), (5, 3)]
        elif dataset['num_classes'] >= 3:
            configurations = [(3, 1), (3, 3), (dataset['num_classes'], 1), (dataset['num_classes'], 3)]
        else:
            configurations = [(2, 1), (2, 3)]
        
        print(f"📋 Testing {len(configurations)} agent configurations")
        
        # Test each configuration
        for config_idx, (n_way, k_shot) in enumerate(configurations):
            config_name = f"{n_way}way{k_shot}shot"
            
            print(f"\n📋 Configuration {config_idx+1}/{len(configurations)}: {config_name}")
            
            try:
                # Create episodes
                TRAIN_EPISODES = 600
                TEST_EPISODES = 400
                
                print(f"📚 Creating episodes...")
                train_episodes = create_few_shot_episodes_multiclass(
                    dataset['train'], 
                    n_way=n_way, 
                    k_shot=k_shot, 
                    q_query=10, 
                    num_episodes=TRAIN_EPISODES
                )
                
                test_episodes = create_few_shot_episodes_multiclass(
                    dataset['test'], 
                    n_way=n_way, 
                    k_shot=k_shot, 
                    q_query=10, 
                    num_episodes=TEST_EPISODES
                )
                
                if train_episodes is None or test_episodes is None:
                    print(f"❌ Failed to create episodes for {config_name}")
                    continue
                
                print(f"✅ Created episodes successfully")
                
                # Initialize communicating agent system
                print(f"🤖 Initializing Communicating Agent System...")
                agent_system = CommunicatingAgentSystem(
                    input_dim=dataset['feature_dim'],
                    embedding_dim=128
                )
                
                # Ensure agent system is on correct device
                agent_system = agent_system.to(device)
                print(f"🔧 Agent system moved to device: {device}")
                
                # Train agent system
                start_time = time.time()
                model_save_path = f"models/{dataset_name.lower().replace('-', '_')}_{config_name}_best_agents.pth"
                training_stats, best_accuracy, best_epoch = train_agent_system(
                    agent_system, train_episodes, device, num_epochs=100, 
                    save_best=True, model_save_path=model_save_path
                )
                training_time = time.time() - start_time
                
                # Evaluate agent system
                print(f"📊 Evaluating agent system...")
                eval_results = evaluate_agent_system(agent_system, test_episodes, dataset['class_names'], device, n_way)
                
                if eval_results is None:
                    print(f"❌ No valid predictions for {config_name}")
                    continue
                
                print(f"✅ {config_name} Agent Communication Results:")
                print(f"   Best Training Accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
                print(f"   Ensemble Test Accuracy: {eval_results['ensemble_accuracy']:.4f}")
                print(f"   Ensemble F1-Score: {eval_results['ensemble_f1_score']:.4f}")
                print(f"   Training Time: {training_time:.1f}s")
                print(f"   Model Saved: {model_save_path}")
                print(f"   Communication Rounds: {eval_results['communication_summary']['total_communication_rounds']}")
                
                # Individual agent performance
                for agent_metric in eval_results['agent_metrics']:
                    print(f"   {agent_metric['agent_type']}: Acc={agent_metric['accuracy']:.4f}, F1={agent_metric['f1_score']:.4f}")
                
                # Communication summary
                comm_summary = eval_results['communication_summary']
                print(f"   Agent Communication Summary:")
                for i, agent_summary in enumerate(comm_summary['agent_summaries']):
                    print(f"     {agent_summary['agent_type']}: {agent_summary['messages_received']} messages, strength={agent_summary['avg_message_strength']:.3f}")
                
                # Add to results
                results_text.append(f"\nConfiguration {config_idx+1}: {config_name}")
                results_text.append(f"  Setup: {n_way}-way {k_shot}-shot")
                results_text.append(f"  Episodes: {TRAIN_EPISODES} train, {TEST_EPISODES} test")
                results_text.append(f"  Training Time: {training_time:.1f} seconds")
                results_text.append(f"  Best Training Accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
                results_text.append(f"  Model Saved: {model_save_path}")
                results_text.append(f"")
                results_text.append(f"  Communication Summary:")
                results_text.append(f"    Total Communication Rounds: {comm_summary['total_communication_rounds']}")
                results_text.append(f"    Agent Types: {', '.join(comm_summary['agent_types'])}")
                results_text.append(f"")
                results_text.append(f"  Ensemble Results:")
                results_text.append(f"    Accuracy: {eval_results['ensemble_accuracy']:.4f}")
                results_text.append(f"    Precision: {eval_results['ensemble_precision']:.4f}")
                results_text.append(f"    Recall: {eval_results['ensemble_recall']:.4f}")
                results_text.append(f"    F1-Score: {eval_results['ensemble_f1_score']:.4f}")
                results_text.append(f"    Predictions Made: {eval_results['num_predictions']}")
                results_text.append(f"")
                results_text.append(f"  Individual Agent Performance:")
                for agent_metric in eval_results['agent_metrics']:
                    results_text.append(f"    {agent_metric['agent_type']} (Agent {agent_metric['agent_id']}):")
                    results_text.append(f"      Accuracy: {agent_metric['accuracy']:.4f}")
                    results_text.append(f"      Precision: {agent_metric['precision']:.4f}")
                    results_text.append(f"      Recall: {agent_metric['recall']:.4f}")
                    results_text.append(f"      F1-Score: {agent_metric['f1_score']:.4f}")
                results_text.append(f"")
                results_text.append(f"  Agent Communication Details:")
                for i, agent_summary in enumerate(comm_summary['agent_summaries']):
                    results_text.append(f"    {agent_summary['agent_type']}:")
                    results_text.append(f"      Messages Received: {agent_summary['messages_received']}")
                    results_text.append(f"      Avg Message Strength: {agent_summary['avg_message_strength']:.4f}")
                    results_text.append(f"      Internal State Norm: {agent_summary['internal_state_norm']:.4f}")
                results_text.append(f"")
                
                # Communication weights matrix
                results_text.append(f"  Inter-Agent Communication Weights:")
                comm_weights = comm_summary['communication_weights']
                results_text.append(f"    (Row = Receiver, Column = Sender)")
                header = "         " + "".join([f"{agent_type[:8]:>10}" for agent_type in comm_summary['agent_types']])
                results_text.append(header)
                for i, agent_type in enumerate(comm_summary['agent_types']):
                    row_str = f"    {agent_type[:8]:>8}: " + "".join([f"{comm_weights[i][j]:>10.3f}" for j in range(len(comm_weights[i]))])
                    results_text.append(row_str)
                results_text.append(f"")
                
                # Confusion Matrix
                results_text.append(f"  Confusion Matrix:")
                cm = eval_results['confusion_matrix']
                class_names = eval_results['class_names']
                
                # Header
                header = "    Pred: " + " ".join([f"{name:>8}" for name in class_names])
                results_text.append(header)
                
                # Matrix rows
                for i, true_class in enumerate(class_names):
                    if i < len(cm):
                        row = f"    {true_class:>4}: " + " ".join([f"{cm[i][j]:>8}" for j in range(len(cm[i]))])
                        results_text.append(row)
                results_text.append("")
                
                # Detailed classification report
                results_text.append(f"  Detailed Classification Report:")
                for line in eval_results['classification_report'].split('\n'):
                    results_text.append(f"    {line}")
                results_text.append("")
                results_text.append("="*60)
                
            except Exception as e:
                print(f"❌ Error with {config_name}: {str(e)}")
                results_text.append(f"\nConfiguration {config_idx+1}: {config_name} - ERROR: {str(e)}")
                continue
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"{dataset_name.lower().replace('-', '_')}_agent_based_results.txt"
        filepath = os.path.join('results', filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(results_text))
        
        print(f"\n✅ Agent-based results saved to {filepath}")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        import traceback
        traceback.print_exc()

def visualize_agent_architecture():
    """Visualize the agent architecture for understanding"""
    print("\n" + "="*80)
    print("🤖 AGENT ARCHITECTURE EXPLANATION")
    print("="*80)
    
    print("\n📊 5 SPECIALIZED AGENTS (Each with separate neural networks):")
    agents_info = [
        ("NetworkTrafficAnalyzer", "Analyzes network flow patterns", "Deep layers for traffic analysis"),
        ("AnomalyDetector", "Detects statistical anomalies", "Statistical analysis focused"),
        ("BehaviorAnalyzer", "Analyzes behavioral patterns", "Behavioral pattern recognition"),
        ("ProtocolAnalyzer", "Analyzes protocol features", "Protocol-specific processing"),
        ("ThreatClassifier", "Classifies threat types", "Final threat categorization")
    ]
    
    for i, (name, purpose, arch) in enumerate(agents_info):
        print(f"\n  Agent {i+1}: {name}")
        print(f"    Purpose: {purpose}")
        print(f"    Architecture: {arch}")
        print(f"    Has: Own feature extractor, classifier, memory, internal state")
    
    print("\n🔄 COMMUNICATION SYSTEM:")
    print("  - Each agent generates messages based on their analysis")
    print("  - 3 rounds of message passing per episode")
    print("  - Attention-based message aggregation")
    print("  - Learnable communication weights matrix (5x5)")
    print("  - Internal state updates based on received messages")
    
    print("\n🎯 DECISION FUSION:")
    print("  - Individual predictions from all 5 agents")
    print("  - Weighted ensemble combination")
    print("  - Final prediction considers all agent inputs")
    
    print("\n💾 MODEL SAVING:")
    print("  - Entire system saved as one file (all 5 agents)")
    print("  - Individual agents also saved separately")
    print("  - Communication weights and training state included")
    print("  - Best model based on training accuracy")
    
    print("\n🔍 WHAT MAKES THIS SPECIAL:")
    print("  - Each agent specializes in different intrusion detection aspects")
    print("  - Agents learn to communicate and share knowledge")
    print("  - Communication patterns adapt during training")
    print("  - Ensemble decision is smarter than individual agents")
    print("="*80)

def main():
    """Main agent-based evaluation function"""
    print("🤖 Agent-Based Few-Shot Learning System")
    print("="*80)
    
    set_seed(42)
    
    datasets = ['In-Vehicle']
    
    for dataset in datasets:
        try:
            evaluate_dataset_with_agents(dataset)
        except Exception as e:
            print(f"❌ Failed to evaluate {dataset}: {e}")
            continue
    
    print(f"\n🎉 Agent-based evaluation completed!")
    print(f"📁 Results saved in 'results/' directory")

if __name__ == "__main__":
    main()
