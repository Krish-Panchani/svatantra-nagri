"""
Google Cloud-Integrated Infrastructure Health AI Agent
Production-ready implementation with GCP services, multi-agent coordination, and Digital Twin
Aligned with the Agentic AI Day urban infrastructure monitoring vision
"""

import time
import json
import datetime
import logging
import asyncio
import threading
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

# Google Cloud imports
try:
    from google.cloud import pubsub_v1
    from google.cloud import bigquery
    from google.cloud import storage
    from google.cloud import monitoring_v3
    from google.cloud import aiplatform
    from google.cloud import functions_v1
    import google.auth
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("Warning: Google Cloud libraries not available. Install google-cloud libraries for full functionality.")

# ML/AI imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. Install scikit-learn and pandas for full functionality.")

# Web and visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from flask import Flask, jsonify, request
    import dash
    from dash import dcc, html, Input, Output
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("Warning: Web libraries not available for dashboard.")


# =============================================================================
# CONFIGURATION AND ENUMS
# =============================================================================

class AgentType(Enum):
    INFRASTRUCTURE_HEALTH = "infrastructure_health"
    TRAFFIC_FLOW = "traffic_flow"
    ENVIRONMENTAL = "environmental"
    PUBLIC_SAFETY = "public_safety"
    ORCHESTRATOR = "orchestrator"

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    ROUTINE = "routine"

class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AlertLevel(Enum):
    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class GCPConfiguration:
    """Google Cloud Platform configuration"""
    project_id: str
    region: str = "us-central1"
    zone: str = "us-central1-a"
    
    # Pub/Sub configuration
    pubsub_topic: str = "infrastructure-monitoring"
    pubsub_subscription: str = "infrastructure-monitoring-sub"
    
    # BigQuery configuration
    dataset_id: str = "infrastructure_monitoring"
    sensor_data_table: str = "sensor_data"
    health_assessments_table: str = "health_assessments"
    predictions_table: str = "ml_predictions"
    
    # Cloud Storage configuration
    bucket_name: str = "infrastructure-monitoring-data"
    
    # AI Platform configuration
    ai_model_endpoint: Optional[str] = None
    
    # Monitoring configuration
    monitoring_project: Optional[str] = None

@dataclass
class InfrastructureConfig:
    """Configuration for a single infrastructure asset"""
    infrastructure_id: str
    infrastructure_type: str  # e.g., "bridge", "road", "tunnel"
    sensor_sampling_rates: Dict[str, int] = field(default_factory=dict)
    anomaly_thresholds: Dict[str, float] = field(default_factory=dict)
    digital_twin_enabled: bool = True
    simulation_enabled: bool = True

@dataclass
class AgentConfiguration:
    agent_id: str
    agent_type: AgentType
    infrastructures: List['InfrastructureConfig']
    gcp_config: GCPConfiguration
    data_retention_days: int = 365
    health_assessment_interval: int = 300  # 5 minutes
    model_retrain_interval: int = 86400  # 24 hours
    prediction_horizon_days: int = 30
    batch_size: int = 100
    max_workers: int = 4
    digital_twin_enabled: bool = True
    simulation_enabled: bool = True
    coordination_enabled: bool = True
    orchestrator_endpoint: Optional[str] = None
    anomaly_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    sensor_sampling_rates: Dict[str, int] = field(default_factory=lambda: {
        'strain_sensors': 60,
        'vibration_monitors': 60,
        'environmental_sensors': 300,
        'load_sensors': 120
    })

    def __post_init__(self):
        if self.anomaly_thresholds is None:
            self.anomaly_thresholds = {
                'strain_sensors': {'moderate': 2.0, 'high': 3.0, 'critical': 4.0},
                'vibration_monitors': {'moderate': 1.5, 'high': 2.5, 'critical': 3.5},
                'displacement_sensors': {'moderate': 2.0, 'high': 3.0, 'critical': 4.0},
            }
        if self.infrastructures is None:
            self.infrastructures = []

@dataclass
class MaintenanceAction:
    id: str
    action_type: str
    component: str
    priority: Priority
    estimated_cost: float
    recommended_date: datetime.datetime
    description: str
    required_resources: List[str]
    failure_probability: float = 0.0
    estimated_rul_days: int = 0
    risk_score: float = 0.0


# =============================================================================
# GOOGLE CLOUD INTEGRATION LAYER
# =============================================================================

class GoogleCloudIntegration:
    """Google Cloud Platform integration services"""
    
    def __init__(self, config: GCPConfiguration):
        self.config = config
        self.publisher = None
        self.subscriber = None
        self.bigquery_client = None
        self.storage_client = None
        self.monitoring_client = None
        self.aiplatform_client = None
        
        if GCP_AVAILABLE:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        try:
            # Pub/Sub clients
            self.publisher = pubsub_v1.PublisherClient()
            self.subscriber = pubsub_v1.SubscriberClient()
            
            # BigQuery client
            self.bigquery_client = bigquery.Client(project=self.config.project_id)
            
            # Cloud Storage client
            self.storage_client = storage.Client(project=self.config.project_id)
            
            # Cloud Monitoring client
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            
            # AI Platform client
            aiplatform.init(project=self.config.project_id, location=self.config.region)
            
            logging.info("Google Cloud clients initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Google Cloud clients: {e}")
            raise
    
    def publish_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Publish sensor data to Pub/Sub"""
        if not self.publisher:
            return False
            
        try:
            topic_path = self.publisher.topic_path(self.config.project_id, self.config.pubsub_topic)
            
            # Convert data to JSON string
            data_json = json.dumps(sensor_data, default=str)
            data_bytes = data_json.encode('utf-8')
            
            # Publish message
            future = self.publisher.publish(topic_path, data_bytes)
            future.result()  # Wait for publish to complete
            
            return True
            
        except Exception as e:
            logging.error(f"Error publishing to Pub/Sub: {e}")
            return False
    
    def store_in_bigquery(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """Store data in BigQuery"""
        if not self.bigquery_client:
            return False
            
        try:
            table_id = f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"
            table = self.bigquery_client.get_table(table_id)
            
            errors = self.bigquery_client.insert_rows_json(table, data)
            
            if not errors:
                logging.info(f"Successfully inserted {len(data)} rows into {table_name}")
                return True
            else:
                logging.error(f"Errors inserting data into BigQuery: {errors}")
                return False
                
        except Exception as e:
            logging.error(f"Error storing data in BigQuery: {e}")
            return False
    
    def query_bigquery(self, query: str) -> pd.DataFrame:
        """Query data from BigQuery"""
        if not self.bigquery_client:
            return pd.DataFrame()
            
        try:
            return self.bigquery_client.query(query).to_dataframe()
        except Exception as e:
            logging.error(f"Error querying BigQuery: {e}")
            return pd.DataFrame()
    
    def upload_to_storage(self, blob_name: str, data: bytes) -> bool:
        """Upload data to Cloud Storage"""
        if not self.storage_client:
            return False
            
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data)
            
            logging.info(f"Uploaded {blob_name} to Cloud Storage")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading to Cloud Storage: {e}")
            return False
    
    def send_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> bool:
        """Send custom metric to Cloud Monitoring"""
        if not self.monitoring_client:
            return False
            
        try:
            project_name = f"projects/{self.config.project_id}"
            
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/{metric_name}"
            
            if labels:
                for key, val in labels.items():
                    series.metric.labels[key] = val
            
            series.resource.type = "global"
            
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            point = monitoring_v3.Point(
                {"interval": interval, "value": {"double_value": value}}
            )
            series.points = [point]
            
            self.monitoring_client.create_time_series(
                name=project_name, time_series=[series]
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Error sending metric to Cloud Monitoring: {e}")
            return False


# =============================================================================
# VERTEX AI INTEGRATION FOR ADVANCED ML
# =============================================================================

class VertexAIPredictor:
    """Vertex AI integration for advanced ML predictions"""
    
    def __init__(self, config: GCPConfiguration):
        self.config = config
        self.model_endpoint = None
        
        if GCP_AVAILABLE and config.ai_model_endpoint:
            self._initialize_endpoint()
    
    def _initialize_endpoint(self):
        """Initialize Vertex AI endpoint"""
        try:
            self.model_endpoint = aiplatform.Endpoint(self.config.ai_model_endpoint)
            logging.info("Vertex AI endpoint initialized")
        except Exception as e:
            logging.error(f"Error initializing Vertex AI endpoint: {e}")
    
    def predict_failure_probability(self, features: List[float]) -> Dict[str, Any]:
        """Predict failure probability using Vertex AI model"""
        if not self.model_endpoint:
            return self._fallback_prediction(features)
            
        try:
            instances = [{"features": features}]
            prediction = self.model_endpoint.predict(instances=instances)
            
            return {
                'failure_probability': prediction.predictions[0]['failure_probability'],
                'confidence': prediction.predictions[0]['confidence'],
                'model_version': 'vertex_ai_v1',
                'prediction_time': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error making Vertex AI prediction: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: List[float]) -> Dict[str, Any]:
        """Fallback prediction when Vertex AI is not available"""
        # Simple heuristic-based prediction
        avg_feature = sum(features) / len(features) if features else 0
        failure_prob = min(0.95, max(0.05, avg_feature / 1000))
        
        return {
            'failure_probability': failure_prob,
            'confidence': 0.7,
            'model_version': 'fallback_v1',
            'prediction_time': datetime.datetime.now().isoformat()
        }


# =============================================================================
# ADVANCED DIGITAL TWIN WITH SIMULATION
# =============================================================================

class InfrastructureDigitalTwin:
    """Advanced Digital Twin with real-time simulation capabilities"""
    
    def __init__(self, infrastructure_id: str, config: AgentConfiguration):
        self.infrastructure_id = infrastructure_id
        self.config = config
        self.current_state = {}
        self.historical_states = []
        self.simulation_scenarios = []
        self.health_concerns = []
        self.component_health = {}
        self.performance_metrics = {}
        
        # Initialize baseline state
        self.baseline_state = self._establish_baseline()
        self.current_state = self.baseline_state.copy()
        
        # Simulation engine
        self.simulation_engine = SimulationEngine() if config.simulation_enabled else None
        
    def _establish_baseline(self) -> Dict[str, Any]:
        """Establish baseline infrastructure state"""
        return {
            'structural_health': 1.0,
            'operational_capacity': 1.0,
            'safety_level': 1.0,
            'environmental_impact': 0.1,
            'maintenance_cost': 1000.0,
            'expected_lifetime': 50.0,  # years
            'last_updated': datetime.datetime.now(),
            'confidence_level': 0.95
        }
    
    def update_state(self, sensor_data: Dict[str, Any], analysis_results: Dict[str, Any]):
        """Update digital twin state with new data"""
        try:
            # Update component health based on sensor data
            self._update_component_health(sensor_data, analysis_results)
            
            # Update overall state metrics
            self._update_state_metrics(analysis_results)
            
            # Store historical state
            self._store_historical_state()
            
            # Run simulations if enabled
            if self.simulation_engine:
                self._run_predictive_simulations()
                
            logging.info(f"Digital twin state updated for {self.infrastructure_id}")
            
        except Exception as e:
            logging.error(f"Error updating digital twin state: {e}")
    
    def _update_component_health(self, sensor_data: Dict[str, Any], analysis_results: Dict[str, Any]):
        """Update individual component health scores"""
        for component, data in sensor_data.items():
            current_health = self.component_health.get(component, 1.0)
            
            # Calculate health degradation based on anomaly scores
            anomaly_score = analysis_results.get('anomaly_scores', {}).get(component, 0)
            
            if anomaly_score > 3.0:
                health_change = -0.02  # Significant degradation
            elif anomaly_score > 2.0:
                health_change = -0.01  # Moderate degradation
            elif anomaly_score > 1.0:
                health_change = -0.005  # Minor degradation
            else:
                health_change = 0.001  # Slight recovery
            
            new_health = max(0.0, min(1.0, current_health + health_change))
            self.component_health[component] = new_health
    
    def _update_state_metrics(self, analysis_results: Dict[str, Any]):
        """Update overall state metrics"""
        # Calculate overall structural health
        if self.component_health:
            avg_component_health = sum(self.component_health.values()) / len(self.component_health)
            self.current_state['structural_health'] = avg_component_health
        
        # Update operational capacity based on health concerns
        critical_concerns = analysis_results.get('critical_concerns_count', 0)
        capacity_reduction = min(0.3, critical_concerns * 0.1)
        self.current_state['operational_capacity'] = max(0.3, 1.0 - capacity_reduction)
        
        # Update safety level
        safety_impact = analysis_results.get('safety_impact', 0)
        self.current_state['safety_level'] = max(0.2, 1.0 - safety_impact)
        
        # Update confidence level based on data quality
        data_quality = analysis_results.get('data_quality', 0.95)
        self.current_state['confidence_level'] = data_quality
        
        self.current_state['last_updated'] = datetime.datetime.now()
    
    def _store_historical_state(self):
        """Store current state in historical records"""
        state_snapshot = {
            'timestamp': datetime.datetime.now(),
            'state': self.current_state.copy(),
            'component_health': self.component_health.copy()
        }
        
        self.historical_states.append(state_snapshot)
        
        # Keep only last 1000 historical states
        if len(self.historical_states) > 1000:
            self.historical_states.pop(0)
    
    def _run_predictive_simulations(self):
        """Run predictive simulations for decision support"""
        try:
            scenarios = [
                {'name': 'no_intervention', 'interventions': []},
                {'name': 'preventive_maintenance', 'interventions': ['preventive_maintenance']},
                {'name': 'component_replacement', 'interventions': ['replace_critical_components']},
                {'name': 'full_rehabilitation', 'interventions': ['full_rehabilitation']}
            ]
            
            simulation_results = []
            
            for scenario in scenarios:
                result = self.simulation_engine.run_scenario(
                    current_state=self.current_state,
                    scenario=scenario,
                    time_horizon_days=365
                )
                simulation_results.append(result)
            
            self.simulation_scenarios = simulation_results
            
        except Exception as e:
            logging.error(f"Error running predictive simulations: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current digital twin state"""
        return {
            'infrastructure_id': self.infrastructure_id,
            'current_state': self.current_state,
            'component_health': self.component_health,
            'health_concerns': self.health_concerns,
            'simulation_scenarios': self.simulation_scenarios,
            'last_updated': self.current_state.get('last_updated')
        }
    
    def predict_future_state(self, days_ahead: int) -> Dict[str, Any]:
        """Predict future state using simulation engine"""
        if not self.simulation_engine:
            return self.current_state
            
        try:
            prediction = self.simulation_engine.predict_state(
                current_state=self.current_state,
                component_health=self.component_health,
                days_ahead=days_ahead
            )
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error predicting future state: {e}")
            return self.current_state


class SimulationEngine:
    """Physics-based simulation engine for infrastructure modeling"""
    
    def __init__(self):
        self.degradation_models = self._initialize_degradation_models()
        self.intervention_effects = self._initialize_intervention_effects()
    
    def _initialize_degradation_models(self) -> Dict[str, Any]:
        """Initialize component degradation models"""
        return {
            'structural': {
                'base_rate': 0.001,  # 0.1% per day
                'acceleration_factors': {
                    'high_stress': 2.0,
                    'environmental': 1.5,
                    'age': 1.2
                }
            },
            'operational': {
                'base_rate': 0.0005,
                'acceleration_factors': {
                    'overload': 3.0,
                    'maintenance_delay': 2.0
                }
            }
        }
    
    def _initialize_intervention_effects(self) -> Dict[str, Any]:
        """Initialize intervention effect models"""
        return {
            'preventive_maintenance': {
                'health_improvement': 0.1,
                'cost': 5000,
                'duration_days': 7
            },
            'replace_critical_components': {
                'health_improvement': 0.3,
                'cost': 25000,
                'duration_days': 21
            },
            'full_rehabilitation': {
                'health_improvement': 0.8,
                'cost': 100000,
                'duration_days': 90
            }
        }
    
    def run_scenario(self, current_state: Dict[str, Any], scenario: Dict[str, Any], 
                    time_horizon_days: int) -> Dict[str, Any]:
        """Run a specific scenario simulation"""
        try:
            simulated_state = current_state.copy()
            total_cost = 0.0
            interventions_applied = []
            
            for day in range(time_horizon_days):
                # Apply natural degradation
                simulated_state = self._apply_degradation(simulated_state, 1)
                
                # Apply interventions if scheduled
                for intervention in scenario.get('interventions', []):
                    if self._should_apply_intervention(intervention, day, simulated_state):
                        effect = self.intervention_effects.get(intervention, {})
                        simulated_state = self._apply_intervention(simulated_state, effect)
                        total_cost += effect.get('cost', 0)
                        interventions_applied.append({
                            'day': day,
                            'intervention': intervention,
                            'cost': effect.get('cost', 0)
                        })
            
            return {
                'scenario_name': scenario.get('name', 'unknown'),
                'final_state': simulated_state,
                'total_cost': total_cost,
                'interventions_applied': interventions_applied,
                'time_horizon_days': time_horizon_days,
                'roi': self._calculate_roi(current_state, simulated_state, total_cost)
            }
            
        except Exception as e:
            logging.error(f"Error running scenario simulation: {e}")
            return {'error': str(e)}
    
    def _apply_degradation(self, state: Dict[str, Any], days: int) -> Dict[str, Any]:
        """Apply natural degradation over time"""
        degraded_state = state.copy()
        
        # Apply structural degradation
        structural_degradation = self.degradation_models['structural']['base_rate'] * days
        degraded_state['structural_health'] = max(0.0, 
            degraded_state.get('structural_health', 1.0) - structural_degradation)
        
        # Apply operational degradation
        operational_degradation = self.degradation_models['operational']['base_rate'] * days
        degraded_state['operational_capacity'] = max(0.0, 
            degraded_state.get('operational_capacity', 1.0) - operational_degradation)
        
        return degraded_state
    
    def _should_apply_intervention(self, intervention: str, day: int, state: Dict[str, Any]) -> bool:
        """Determine if intervention should be applied"""
        if intervention == 'preventive_maintenance':
            return day % 180 == 0  # Every 6 months
        elif intervention == 'replace_critical_components':
            return state.get('structural_health', 1.0) < 0.7
        elif intervention == 'full_rehabilitation':
            return state.get('structural_health', 1.0) < 0.5
        
        return False
    
    def _apply_intervention(self, state: Dict[str, Any], effect: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intervention effects to state"""
        improved_state = state.copy()
        
        health_improvement = effect.get('health_improvement', 0)
        improved_state['structural_health'] = min(1.0, 
            improved_state.get('structural_health', 1.0) + health_improvement)
        improved_state['operational_capacity'] = min(1.0, 
            improved_state.get('operational_capacity', 1.0) + health_improvement * 0.5)
        
        return improved_state
    
    def _calculate_roi(self, initial_state: Dict[str, Any], final_state: Dict[str, Any], 
                      total_cost: float) -> float:
        """Calculate return on investment for scenario"""
        if total_cost == 0:
            return 0.0
            
        # Simplified ROI calculation based on health improvement
        initial_health = initial_state.get('structural_health', 1.0)
        final_health = final_state.get('structural_health', 1.0)
        health_benefit = (final_health - initial_health) * 100000  # Value per health point
        
        return (health_benefit - total_cost) / total_cost if total_cost > 0 else 0.0
    
    def predict_state(self, current_state: Dict[str, Any], component_health: Dict[str, Any], 
                     days_ahead: int) -> Dict[str, Any]:
        """Predict future state without interventions"""
        return self._apply_degradation(current_state, days_ahead)


# =============================================================================
# AI-POWERED ANOMALY DETECTION AND PREDICTION
# =============================================================================

class AdvancedAnomalyDetector:
    """Advanced anomaly detection with multiple ML algorithms"""
    
    def __init__(self, gcp_config: GCPConfiguration):
        self.gcp_config = gcp_config
        self.vertex_ai = VertexAIPredictor(gcp_config) if GCP_AVAILABLE else None
        self.local_models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize local models as fallback
        if ML_AVAILABLE:
            self._initialize_local_models()
    
    def _initialize_local_models(self):
        """Initialize local ML models"""
        self.local_models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def train_models(self, historical_data: pd.DataFrame) -> bool:
        """Train anomaly detection models"""
        try:
            if historical_data.empty or len(historical_data) < 100:
                logging.warning("Insufficient data for model training")
                return False
            
            # Train local models
            if ML_AVAILABLE:
                self._train_local_models(historical_data)
            
            # Train Vertex AI models (would be done separately in production)
            if self.vertex_ai:
                logging.info("Vertex AI models assumed to be pre-trained")
            
            self.is_trained = True
            logging.info("Anomaly detection models trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training models: {e}")
            return False
    
    def _train_local_models(self, data: pd.DataFrame):
        """Train local fallback models"""
        # Prepare features
        features = self._extract_features(data)
        if features is None or len(features) == 0:
            return
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        self.scalers['default'] = scaler
        
        # Train isolation forest for anomaly detection
        if 'isolation_forest' in self.local_models:
            self.local_models['isolation_forest'].fit(scaled_features)
        
        # Train random forest for prediction
        if 'random_forest' in self.local_models and len(data) > 1:
            # Create target variable (next value prediction)
            targets = data['value'].shift(-1).dropna()
            features_for_prediction = scaled_features[:-1]  # Remove last row
            
            if len(targets) == len(features_for_prediction):
                self.local_models['random_forest'].fit(features_for_prediction, targets)
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from sensor data"""
        try:
            if 'value' not in data.columns:
                return None
            
            # Basic statistical features
            data_sorted = data.sort_values('timestamp') if 'timestamp' in data.columns else data
            values = data_sorted['value'].values
            
            features = []
            window_size = min(10, len(values))
            
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                feature_vector = [
                    values[i],  # Current value
                    np.mean(window),  # Moving average
                    np.std(window),   # Moving standard deviation
                    np.max(window),   # Moving maximum
                    np.min(window),   # Moving minimum
                    values[i] - values[i-1] if i > 0 else 0,  # First difference
                ]
                features.append(feature_vector)
            
            return np.array(features) if features else None
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in sensor data"""
        try:
            results = {}
            
            for sensor_type, data in sensor_data.items():
                # Try Vertex AI first
                if self.vertex_ai:
                    vertex_result = self._detect_with_vertex_ai(sensor_type, data)
                    if vertex_result:
                        results[sensor_type] = vertex_result
                        continue
                
                # Fallback to local models
                if self.is_trained and ML_AVAILABLE:
                    local_result = self._detect_with_local_models(sensor_type, data)
                    results[sensor_type] = local_result
                else:
                    # Simple threshold-based detection
                    results[sensor_type] = self._detect_with_thresholds(sensor_type, data)
            
            return results
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {e}")
            return {}
    
    def _detect_with_vertex_ai(self, sensor_type: str, data: Any) -> Optional[Dict[str, Any]]:
        """Detect anomalies using Vertex AI"""
        try:
            # Extract numeric features from data
            features = self._extract_single_features(data)
            if not features:
                return None
            
            prediction = self.vertex_ai.predict_failure_probability(features)
            
            # Convert to anomaly score
            failure_prob = prediction.get('failure_probability', 0)
            anomaly_score = failure_prob * 5  # Scale to 0-5 range
            
            if anomaly_score > 3.0:
                alert_level = 'critical'
            elif anomaly_score > 2.0:
                alert_level = 'high'
            elif anomaly_score > 1.0:
                alert_level = 'moderate'
            else:
                alert_level = 'normal'
            
            return {
                'anomaly_score': round(anomaly_score, 3),
                'alert_level': alert_level,
                'failure_probability': failure_prob,
                'confidence': prediction.get('confidence', 0.7),
                'model_source': 'vertex_ai'
            }
            
        except Exception as e:
            logging.error(f"Error with Vertex AI detection: {e}")
            return None
    
    def _detect_with_local_models(self, sensor_type: str, data: Any) -> Dict[str, Any]:
        """Detect anomalies using local models"""
        try:
            # Extract features
            features = self._extract_single_features(data)
            if not features:
                return self._detect_with_thresholds(sensor_type, data)
            
            # Scale features
            scaler = self.scalers.get('default')
            if scaler:
                features_array = np.array([features])
                scaled_features = scaler.transform(features_array)
            else:
                scaled_features = np.array([features])
            
            # Get anomaly score from isolation forest
            isolation_forest = self.local_models.get('isolation_forest')
            if isolation_forest:
                anomaly_score = abs(isolation_forest.decision_function(scaled_features)[0])
            else:
                anomaly_score = 0.0
            
            # Determine alert level
            if anomaly_score > 1.5:
                alert_level = 'critical'
            elif anomaly_score > 1.0:
                alert_level = 'high'
            elif anomaly_score > 0.5:
                alert_level = 'moderate'
            else:
                alert_level = 'normal'
            
            return {
                'anomaly_score': round(anomaly_score, 3),
                'alert_level': alert_level,
                'confidence': 0.8,
                'model_source': 'local_ml'
            }
            
        except Exception as e:
            logging.error(f"Error with local model detection: {e}")
            return self._detect_with_thresholds(sensor_type, data)
    
    def _detect_with_thresholds(self, sensor_type: str, data: Any) -> Dict[str, Any]:
        """Simple threshold-based anomaly detection"""
        try:
            # Extract primary value
            value = self._extract_primary_value(data)
            
            # Define thresholds based on sensor type
            thresholds = {
                'strain_sensors': {'moderate': 500, 'high': 800, 'critical': 1200},
                'vibration_monitors': {'moderate': 0.2, 'high': 0.4, 'critical': 0.6},
                'displacement_sensors': {'moderate': 10, 'high': 20, 'critical': 30},
                'environmental_sensors': {'moderate': 50, 'high': 80, 'critical': 100}
            }
            
            sensor_thresholds = thresholds.get(sensor_type, {'moderate': 100, 'high': 200, 'critical': 300})
            
            if abs(value) >= sensor_thresholds['critical']:
                alert_level = 'critical'
                anomaly_score = 4.0
            elif abs(value) >= sensor_thresholds['high']:
                alert_level = 'high'
                anomaly_score = 3.0
            elif abs(value) >= sensor_thresholds['moderate']:
                alert_level = 'moderate'
                anomaly_score = 2.0
            else:
                alert_level = 'normal'
                anomaly_score = 0.5
            
            return {
                'anomaly_score': anomaly_score,
                'alert_level': alert_level,
                'confidence': 0.6,
                'model_source': 'threshold'
            }
            
        except Exception as e:
            logging.error(f"Error with threshold detection: {e}")
            return {
                'anomaly_score': 0.0,
                'alert_level': 'normal',
                'confidence': 0.1,
                'model_source': 'error'
            }
    
    def _extract_single_features(self, data: Any) -> List[float]:
        """Extract numeric features from single data point"""
        features = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, dict):
                    # Recursively extract from nested dictionaries
                    nested_features = self._extract_single_features(value)
                    features.extend(nested_features)
        elif isinstance(data, (int, float)):
            features.append(float(data))
        
        return features[:10]  # Limit to first 10 features
    
    def _extract_primary_value(self, data: Any) -> float:
        """Extract primary numeric value from data"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, dict):
            # Look for common value keys
            for key in ['value', 'magnitude', 'strain_magnitude', 'peak_acceleration', 'displacement']:
                if key in data and isinstance(data[key], (int, float)):
                    return float(data[key])
            
            # If no common keys, return first numeric value found
            for value in data.values():
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, dict):
                    nested_value = self._extract_primary_value(value)
                    if nested_value != 0:
                        return nested_value
        
        return 0.0


# =============================================================================
# INTELLIGENT SENSOR DATA SIMULATORS
# =============================================================================

class IntelligentStrainSensorSimulator:
    """Intelligent strain sensor with realistic physics modeling"""
    
    def __init__(self, infrastructure_id: str):
        self.infrastructure_id = infrastructure_id
        self.sensor_locations = [
            {'id': 'SG001', 'location': 'Main Girder - Midspan', 'baseline': 200, 'sensitivity': 1.2},
            {'id': 'SG002', 'location': 'Main Girder - Support', 'baseline': 150, 'sensitivity': 1.0},
            {'id': 'SG003', 'location': 'Cross Beam - Center', 'baseline': 100, 'sensitivity': 0.8},
            {'id': 'SG004', 'location': 'Deck Slab - Edge', 'baseline': 80, 'sensitivity': 1.1},
            {'id': 'SG005', 'location': 'Cable/Tendon', 'baseline': 300, 'sensitivity': 1.5}
        ]
        self.start_time = datetime.datetime.now()
        self.degradation_state = {sensor['id']: 1.0 for sensor in self.sensor_locations}
        
    def get_current_data(self) -> Dict[str, Any]:
        """Generate realistic strain data with physics-based modeling"""
        data = {}
        current_time = datetime.datetime.now()
        
        # Environmental factors
        hour = current_time.hour
        month = current_time.month
        
        # Traffic loading pattern (realistic daily variation)
        traffic_factor = self._calculate_traffic_loading(hour)
        
        # Temperature effects
        thermal_strain = self._calculate_thermal_strain(month, hour)
        
        # Age-related degradation
        age_days = (current_time - self.start_time).days
        degradation_factor = 1 + (age_days * 0.0001)  # 0.01% per day
        
        for sensor in self.sensor_locations:
            sensor_id = sensor['id']
            
            # Base strain calculation
            base_strain = sensor['baseline'] * sensor['sensitivity']
            
            # Apply environmental factors
            total_strain = (base_strain * traffic_factor * degradation_factor) + thermal_strain
            
            # Apply sensor-specific degradation
            sensor_degradation = self.degradation_state.get(sensor_id, 1.0)
            total_strain *= sensor_degradation
            
            # Add realistic noise
            noise = np.random.normal(0, 2) if ML_AVAILABLE else 0
            total_strain += noise
            
            # Calculate derived strains
            longitudinal_strain = total_strain
            transverse_strain = total_strain * 0.3  # Poisson effect
            shear_strain = total_strain * 0.1
            
            # Quality assessment
            quality = 'excellent' if abs(noise) < 1 else ('good' if abs(noise) < 3 else 'fair')
            
            data[sensor_id] = {
                'location': sensor['location'],
                'longitudinal_strain': round(longitudinal_strain, 2),
                'transverse_strain': round(transverse_strain, 2),
                'shear_strain': round(shear_strain, 2),
                'strain_magnitude': round(abs(total_strain), 2),
                'strain_rate': round(self._calculate_strain_rate(sensor_id, total_strain), 3),
                'temperature_compensation': round(thermal_strain, 2),
                'quality': quality,
                'calibration_date': (current_time - datetime.timedelta(days=30)).isoformat(),
                'timestamp': current_time.isoformat(),
                'sensor_health': round(sensor_degradation, 3)
            }
            
            # Update sensor degradation (very gradual)
            if abs(total_strain) > sensor['baseline'] * 2:  # Overload condition
                self.degradation_state[sensor_id] = max(0.1, sensor_degradation - 0.001)
        
        return data
    
    def _calculate_traffic_loading(self, hour: int) -> float:
        """Calculate realistic traffic loading factor"""
        # Rush hour peaks
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_factor = 1.4
            variation = 0.2
        # Midday traffic
        elif 11 <= hour <= 14:
            base_factor = 1.1
            variation = 0.15
        # Night traffic
        elif 22 <= hour or hour <= 5:
            base_factor = 0.3
            variation = 0.1
        # Regular traffic
        else:
            base_factor = 0.9
            variation = 0.1
        
        # Add random variation
        if ML_AVAILABLE:
            random_factor = np.random.normal(1.0, variation)
        else:
            import random
            random_factor = random.uniform(1.0 - variation, 1.0 + variation)
        
        return max(0.1, base_factor * random_factor)
    
    def _calculate_thermal_strain(self, month: int, hour: int) -> float:
        """Calculate temperature-induced strain"""
        # Seasonal temperature variation
        if month in [12, 1, 2]:  # Winter
            base_temp = -5
            daily_variation = 8
        elif month in [6, 7, 8]:  # Summer
            base_temp = 25
            daily_variation = 12
        else:  # Spring/Fall
            base_temp = 15
            daily_variation = 10
        
        # Daily temperature cycle
        temp_factor = np.sin((hour - 6) * np.pi / 12) if ML_AVAILABLE else 0.5
        current_temp = base_temp + (daily_variation * temp_factor)
        
        # Thermal expansion coefficient for steel/concrete
        thermal_coefficient = 1.2e-5  # per degree Celsius
        reference_temp = 20  # Reference temperature
        
        thermal_strain = (current_temp - reference_temp) * thermal_coefficient * 1000000  # Convert to microstrain
        
        return thermal_strain
    
    def _calculate_strain_rate(self, sensor_id: str, current_strain: float) -> float:
        """Calculate strain rate (change over time)"""
        # Simple implementation - in production, would use historical data
        if ML_AVAILABLE:
            return np.random.normal(0, 0.1)  # Microstrain per second
        else:
            import random
            return random.uniform(-0.2, 0.2)


class IntelligentVibrationSensorSimulator:
    """Advanced vibration sensor with modal analysis"""
    
    def __init__(self, infrastructure_id: str):
        self.infrastructure_id = infrastructure_id
        self.baseline_frequencies = [2.45, 4.82, 7.19, 10.15, 13.44]  # Hz
        self.baseline_damping = [0.025, 0.030, 0.035, 0.040, 0.045]  # Damping ratios
        self.baseline_mode_shapes = self._initialize_mode_shapes()
        self.start_time = datetime.datetime.now()
        
    def _initialize_mode_shapes(self) -> List[List[float]]:
        """Initialize baseline mode shapes"""
        return [
            [0.0, 0.31, 0.71, 0.95, 1.0, 0.95, 0.71, 0.31, 0.0],  # Mode 1
            [0.0, 0.67, 0.87, 0.0, -0.87, -0.67, 0.0, 0.67, 0.0],  # Mode 2
            [0.0, 0.71, 0.0, -0.95, 0.0, 0.95, 0.0, -0.71, 0.0],   # Mode 3
            [0.0, 0.5, -0.87, 0.5, 0.0, -0.5, 0.87, -0.5, 0.0],    # Mode 4
            [0.0, 0.31, -0.81, 0.81, -0.31, 0.0, 0.31, -0.81, 0.0] # Mode 5
        ]
    
    def get_current_data(self) -> Dict[str, Any]:
        """Generate realistic vibration data with damage simulation"""
        current_time = datetime.datetime.now()
        
        # Simulate gradual damage effects over time
        age_years = (current_time - self.start_time).days / 365.25
        damage_factor = 1 - (age_years * 0.005)  # 0.5% reduction per year
        
        # Environmental effects
        wind_speed = self._get_wind_speed()
        traffic_excitation = self._get_traffic_excitation()
        
        # Calculate current modal parameters
        current_frequencies = []
        current_damping = []
        current_mode_shapes = []
        
        for i, (base_freq, base_damp, base_shape) in enumerate(zip(
            self.baseline_frequencies, self.baseline_damping, self.baseline_mode_shapes)):
            
            # Frequency changes due to damage and environmental effects
            freq_change = 1.0
            freq_change *= damage_factor  # Damage effect
            freq_change *= (1 + wind_speed * 0.01)  # Wind effect
            freq_change *= (1 + traffic_excitation * 0.005)  # Traffic effect
            
            # Add measurement uncertainty
            if ML_AVAILABLE:
                uncertainty = np.random.normal(1.0, 0.002)
            else:
                damp_uncertainty = random.uniform(0.95, 1.05)
            
            current_freq = base_freq * freq_change * uncertainty
            current_frequencies.append(round(current_freq, 4))
            
            # Damping changes (usually increases with damage)
            damp_change = 1 + (1 - damage_factor) * 0.5  # Damage increases damping
            current_damp = base_damp * damp_change
            
            if ML_AVAILABLE:
                damp_uncertainty = np.random.normal(1.0, 0.05)
            else:
                damp_uncertainty = random.uniform(0.95, 1.05)
            
            current_damp *= damp_uncertainty
            current_damping.append(round(max(0.005, current_damp), 5))
            
            # Mode shapes (simplified - normally would be more complex)
            current_mode_shapes.append(base_shape)
        
        # Calculate response amplitudes
        peak_acceleration = self._calculate_peak_acceleration(wind_speed, traffic_excitation)
        rms_acceleration = peak_acceleration * 0.3  # Typical RMS to peak ratio
        
        # Frequency domain analysis
        frequency_resolution = 0.01  # Hz
        measurement_duration = 600    # 10 minutes
        
        # Response statistics
        response_statistics = self._calculate_response_statistics(
            current_frequencies, current_damping, peak_acceleration
        )
        
        return {
            'infrastructure_id': self.infrastructure_id,
            'modal_parameters': {
                'natural_frequencies': current_frequencies,
                'damping_ratios': current_damping,
                'mode_shapes': current_mode_shapes,
                'modal_assurance_criteria': self._calculate_mac()
            },
            'response_measurements': {
                'peak_acceleration': round(peak_acceleration, 4),
                'rms_acceleration': round(rms_acceleration, 4),
                'peak_velocity': round(peak_acceleration / (2 * np.pi * current_frequencies[0]), 4) if ML_AVAILABLE else round(peak_acceleration / 15.7, 4),
                'peak_displacement': round(peak_acceleration / ((2 * np.pi * current_frequencies[0]) ** 2), 6) if ML_AVAILABLE else round(peak_acceleration / 247, 6)
            },
            'analysis_parameters': {
                'frequency_resolution': frequency_resolution,
                'measurement_duration': measurement_duration,
                'sampling_rate': 1024,  # Hz
                'window_function': 'Hanning',
                'overlap_percentage': 50
            },
            'environmental_conditions': {
                'wind_speed': round(wind_speed, 2),
                'traffic_level': round(traffic_excitation, 2),
                'temperature': round(self._get_temperature(), 1),
                'humidity': round(self._get_humidity(), 1)
            },
            'data_quality': {
                'signal_to_noise_ratio': round(20 + 10 * damage_factor, 1),
                'coherence_function': round(0.9 + 0.1 * damage_factor, 3),
                'measurement_uncertainty': '±2%'
            },
            'response_statistics': response_statistics,
            'damage_indicators': self._calculate_damage_indicators(damage_factor),
            'timestamp': current_time.isoformat(),
            'sensor_health': round(damage_factor, 3)
        }
    
    def _get_wind_speed(self) -> float:
        """Get current wind speed"""
        if ML_AVAILABLE:
            return max(0, np.random.gamma(2, 5))  # Realistic wind distribution
        else:
            import random
            return random.uniform(0, 20)
    
    def _get_traffic_excitation(self) -> float:
        """Get traffic-induced excitation level"""
        hour = datetime.datetime.now().hour
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_level = 0.8
        elif 22 <= hour or hour <= 5:  # Night
            base_level = 0.1
        else:
            base_level = 0.4
        
        if ML_AVAILABLE:
            return base_level * np.random.uniform(0.5, 1.5)
        else:
            import random
            return base_level * random.uniform(0.5, 1.5)
    
    def _get_temperature(self) -> float:
        """Get current temperature"""
        month = datetime.datetime.now().month
        hour = datetime.datetime.now().hour
        
        if month in [12, 1, 2]:  # Winter
            base_temp = 5
        elif month in [6, 7, 8]:  # Summer
            base_temp = 30
        else:
            base_temp = 20
        
        # Daily variation
        daily_variation = 8 * np.sin((hour - 6) * np.pi / 12) if ML_AVAILABLE else 4
        return base_temp + daily_variation
    
    def _get_humidity(self) -> float:
        """Get current humidity"""
        if ML_AVAILABLE:
            return np.random.uniform(30, 85)
        else:
            import random
            return random.uniform(30, 85)
    
    def _calculate_peak_acceleration(self, wind_speed: float, traffic_excitation: float) -> float:
        """Calculate peak acceleration response"""
        # Base ambient vibration
        base_acceleration = 0.01  # m/s²
        
        # Wind-induced response
        wind_response = wind_speed ** 2 * 0.0001
        
        # Traffic-induced response
        traffic_response = traffic_excitation * 0.05
        
        # Random excitation
        if ML_AVAILABLE:
            random_response = np.random.exponential(0.02)
        else:
            import random
            random_response = random.uniform(0, 0.04)
        
        total_acceleration = base_acceleration + wind_response + traffic_response + random_response
        return min(total_acceleration, 0.5)  # Cap at reasonable value
    
    def _calculate_mac(self) -> List[float]:
        """Calculate Modal Assurance Criteria"""
        # Simplified MAC calculation
        if ML_AVAILABLE:
            return [round(np.random.uniform(0.85, 1.0), 3) for _ in range(5)]
        else:
            import random
            return [round(random.uniform(0.85, 1.0), 3) for _ in range(5)]
    
    def _calculate_response_statistics(self, frequencies: List[float], damping: List[float], 
                                     peak_accel: float) -> Dict[str, Any]:
        """Calculate response statistics"""
        return {
            'dominant_frequency': frequencies[0],
            'frequency_spread': round(max(frequencies) - min(frequencies), 3),
            'average_damping': round(sum(damping) / len(damping), 4),
            'damping_spread': round(max(damping) - min(damping), 4),
            'response_amplitude_ratio': round(peak_accel / 0.01, 2),
            'dynamic_amplification': round(1 / (2 * damping[0]), 1) if damping[0] > 0 else 50
        }
    
    def _calculate_damage_indicators(self, damage_factor: float) -> Dict[str, Any]:
        """Calculate damage indicators"""
        frequency_shift = (1 - damage_factor) * 100  # Percentage shift
        damping_increase = (1 / damage_factor - 1) * 100  # Percentage increase
        
        return {
            'frequency_shift_percentage': round(frequency_shift, 2),
            'damping_increase_percentage': round(damping_increase, 2),
            'modal_flexibility_change': round(frequency_shift * 2, 2),
            'curvature_damage_index': round(damage_factor, 3),
            'damage_severity': 'low' if damage_factor > 0.9 else ('moderate' if damage_factor > 0.8 else 'high')
        }


# Additional intelligent sensor simulators
class EnvironmentalSensorSimulator:
    """Environmental sensor with weather correlation"""
    
    def __init__(self, infrastructure_id: str):
        self.infrastructure_id = infrastructure_id
        self.location_lat = 40.7128  # Default to NYC coordinates
        self.location_lon = -74.0060
        
    def get_current_data(self) -> Dict[str, Any]:
        """Generate realistic environmental data"""
        current_time = datetime.datetime.now()
        month = current_time.month
        hour = current_time.hour
        
        # Temperature modeling
        if month in [12, 1, 2]:  # Winter
            base_temp = 2
            daily_range = 8
        elif month in [6, 7, 8]:  # Summer
            base_temp = 25
            daily_range = 12
        else:  # Spring/Fall
            base_temp = 15
            daily_range = 10
        
        # Daily temperature cycle
        temp_cycle = np.sin((hour - 6) * np.pi / 12) if ML_AVAILABLE else 0.5
        temperature = base_temp + (daily_range * temp_cycle / 2)
        
        # Add weather variation
        if ML_AVAILABLE:
            temperature += np.random.normal(0, 2)
        
        # Humidity (inversely correlated with temperature)
        base_humidity = 70
        humidity = base_humidity - (temperature - 15) * 1.5
        humidity = max(20, min(95, humidity))
        
        if ML_AVAILABLE:
            humidity += np.random.normal(0, 5)
        
        # Wind speed
        if ML_AVAILABLE:
            wind_speed = np.random.gamma(2, 3)  # Realistic wind distribution
        else:
            import random
            wind_speed = random.uniform(0, 15)
        
        # Precipitation
        precipitation_prob = 0.15 if month in [4, 5, 6, 7, 8, 9] else 0.25
        if ML_AVAILABLE:
            is_precipitating = np.random.random() < precipitation_prob
            precipitation = np.random.exponential(2) if is_precipitating else 0
        else:
            import random
            precipitation = random.uniform(0, 5) if random.random() < precipitation_prob else 0
        
        # Air quality index
        base_aqi = 50
        traffic_factor = 1.3 if 7 <= hour <= 19 else 0.8
        aqi = base_aqi * traffic_factor
        
        if ML_AVAILABLE:
            aqi += np.random.normal(0, 10)
        
        aqi = max(0, min(300, aqi))
        
        return {
            'infrastructure_id': self.infrastructure_id,
            'meteorological': {
                'temperature_celsius': round(temperature, 1),
                'humidity_percentage': round(humidity, 1),
                'wind_speed_mps': round(wind_speed, 2),
                'wind_direction_degrees': round(np.random.uniform(0, 360), 0) if ML_AVAILABLE else 180,
                'atmospheric_pressure_hpa': round(1013.25 + np.random.normal(0, 15), 1) if ML_AVAILABLE else 1013,
                'precipitation_mm': round(precipitation, 2),
                'solar_radiation_wm2': self._calculate_solar_radiation(month, hour),
                'uv_index': self._calculate_uv_index(month, hour)
            },
            'air_quality': {
                'aqi': round(aqi, 0),
                'pm2_5_ugm3': round(aqi * 0.5, 1),
                'pm10_ugm3': round(aqi * 0.8, 1),
                'no2_ugm3': round(aqi * 0.6, 1),
                'co_mgm3': round(aqi * 0.02, 2),
                'visibility_km': round(max(1, 20 - precipitation * 2), 1)
            },
            'derived_parameters': {
                'heat_index': self._calculate_heat_index(temperature, humidity),
                'wind_chill': self._calculate_wind_chill(temperature, wind_speed),
                'corrosion_risk': self._calculate_corrosion_risk(temperature, humidity, precipitation),
                'freeze_thaw_cycles': self._detect_freeze_thaw(temperature),
                'structural_stress_factor': self._calculate_stress_factor(temperature, wind_speed)
            },
            'timestamp': current_time.isoformat(),
            'location': {
                'latitude': self.location_lat,
                'longitude': self.location_lon
            }
        } 
    
    def _calculate_solar_radiation(self, month: int, hour: int) -> float:
        """Calculate solar radiation"""
        if hour < 6 or hour > 18:
            return 0.0
        
        # Seasonal variation
        seasonal_factor = 0.7 + 0.3 * np.cos((month - 6) * np.pi / 6) if ML_AVAILABLE else 0.8
        
        # Daily variation
        daily_factor = np.sin((hour - 6) * np.pi / 12) if ML_AVAILABLE else 0.5
        
        max_radiation = 1000  # W/m²
        radiation = max_radiation * seasonal_factor * daily_factor
        
        return round(max(0, radiation), 1)
    
    def _calculate_uv_index(self, month: int, hour: int) -> int:
        """Calculate UV index"""
        if hour < 8 or hour > 16:
            return 0
        
        if temp < 27:  # Heat index only relevant above 27°C
            return temp
        
        # Simplified heat index calculation
        hi = temp + 0.5 * (temp - 27) * (humidity / 100)
        return round(hi, 1)
    
    def _calculate_wind_chill(self, temp: float, wind_speed: float) -> float:
        """Calculate wind chill"""
        if temp > 10:  # Wind chill only relevant below 10°C
            return temp
        
        # Simplified wind chill calculation
        wc = temp - 2 * np.sqrt(wind_speed) if ML_AVAILABLE else temp - wind_speed * 0.5
        return round(wc, 1)
    
    def _calculate_corrosion_risk(self, temp: float, humidity: float, precipitation: float) -> str:
        """Calculate corrosion risk level"""
        risk_score = 0
        
        # High humidity increases corrosion
        if humidity > 80:
            risk_score += 2
        elif humidity > 60:
            risk_score += 1
        
        # Temperature effects
        if 20 <= temp <= 35:  # Optimal corrosion temperature range
            risk_score += 2
        elif temp > 35 or temp < 0:
            risk_score += 1
        
        # Precipitation
        if precipitation > 0:
            risk_score += 2
        
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'moderate'
        else:
            return 'low'
    
    def _detect_freeze_thaw(self, temp: float) -> int:
        """Detect freeze-thaw cycles (simplified)"""
        # This would normally track temperature history
        # Simplified: if temp is near freezing point
        if -2 <= temp <= 2:
            return 1  # Potential freeze-thaw cycle
        return 0
    
    def _calculate_stress_factor(self, temp: float, wind_speed: float) -> float:
        """Calculate environmental stress factor on infrastructure"""
        stress_factor = 1.0
        
        # Temperature stress
        if temp > 35 or temp < -10:
            stress_factor *= 1.2
        
        # Wind stress
        if wind_speed > 15:
            stress_factor *= 1.1
        
        return round(stress_factor, 2)

    def _calculate_heat_index(self, temperature_celsius, humidity_percent):
        # Simple heat index formula placeholder
        T = temperature_celsius * 9/5 + 32  # to Fahrenheit
        R = humidity_percent
        HI = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 6.83783e-3*T*T - 5.481717e-2*R*R + 1.22874e-3*T*T*R + 8.5282e-4*T*R*R - 1.99e-6*T*T*R*R
        heat_index_c = (HI - 32) * 5 / 9
        return round(heat_index_c, 1)


class LoadSensorSimulator:
    """Traffic load sensor with realistic vehicle modeling"""
    
    def __init__(self, infrastructure_id: str):
        self.infrastructure_id = infrastructure_id
        self.vehicle_types = {
            'passenger_car': {'weight': 1.5, 'frequency': 0.7},
            'light_truck': {'weight': 3.5, 'frequency': 0.15},
            'heavy_truck': {'weight': 25.0, 'frequency': 0.12},
            'bus': {'weight': 12.0, 'frequency': 0.02},
            'motorcycle': {'weight': 0.3, 'frequency': 0.01}
        }
        
    def get_current_data(self) -> Dict[str, Any]:
        """Generate realistic traffic load data"""
        current_time = datetime.datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Traffic volume based on time and day
        traffic_multiplier = self._get_traffic_multiplier(hour, day_of_week)
        
        # Generate vehicle crossings for the last minute
        vehicles_per_minute = max(1, int(20 * traffic_multiplier))
        
        total_vehicles = 0
        total_weight = 0
        max_axle_load = 0
        vehicle_distribution = {}
        
        for vehicle_type, props in self.vehicle_types.items():
            # Number of this vehicle type
            expected_count = vehicles_per_minute * props['frequency']
            
            if ML_AVAILABLE:
                actual_count = np.random.poisson(expected_count)
            else:
                import random
                actual_count = max(0, int(expected_count + random.uniform(-1, 1)))
            
            vehicle_distribution[vehicle_type] = actual_count
            total_vehicles += actual_count
            
            # Calculate load contribution
            for _ in range(actual_count):
                # Vehicle weight variation
                if ML_AVAILABLE:
                    weight_variation = np.random.normal(1.0, 0.2)
                else:
                    weight_variation = random.uniform(0.8, 1.2)
                
                vehicle_weight = props['weight'] * weight_variation
                total_weight += vehicle_weight
                
                # Estimate axle load (simplified)
                axle_load = vehicle_weight / self._get_axle_count(vehicle_type)
                max_axle_load = max(max_axle_load, axle_load)
        
        # Dynamic effects
        dynamic_amplification = self._calculate_dynamic_amplification()
        effective_load = total_weight * dynamic_amplification
        
        # Load distribution across bridge
        load_distribution = self._calculate_load_distribution(total_vehicles)
        
        return {
            'infrastructure_id': self.infrastructure_id,
            'traffic_summary': {
                'total_vehicles': total_vehicles,
                'vehicles_per_hour': total_vehicles * 60,
                'total_weight_tonnes': round(total_weight, 2),
                'effective_load_tonnes': round(effective_load, 2),
                'max_axle_load_tonnes': round(max_axle_load, 2),
                'average_vehicle_weight': round(total_weight / total_vehicles, 2) if total_vehicles > 0 else 0
            },
            'vehicle_distribution': vehicle_distribution,
            'load_analysis': {
                'dynamic_amplification_factor': round(dynamic_amplification, 3),
                'load_distribution_pattern': load_distribution,
                'bridge_utilization_percentage': round(min(100, (effective_load / 100) * 100), 1),
                'overload_incidents': self._detect_overload_incidents(max_axle_load),
                'fatigue_damage_equivalent': self._calculate_fatigue_damage(total_weight, total_vehicles)
            },
            'traffic_characteristics': {
                'peak_hour_factor': round(traffic_multiplier, 2),
                'truck_percentage': round(sum([
                    vehicle_distribution.get('light_truck', 0),
                    vehicle_distribution.get('heavy_truck', 0)
                ]) / total_vehicles * 100, 1) if total_vehicles > 0 else 0,
                'average_speed_kmh': self._estimate_average_speed(traffic_multiplier),
                'headway_seconds': round(60 / total_vehicles, 1) if total_vehicles > 0 else 60
            },
            'real_time_monitoring': {
                'current_occupancy': total_vehicles,
                'weight_in_motion': round(total_weight, 1),
                'bridge_response_frequency': self._calculate_response_frequency(effective_load),
                'strain_demand': self._calculate_strain_demand(effective_load),
                'safety_margin': self._calculate_safety_margin(effective_load)
            },
            'timestamp': current_time.isoformat()
        }
    
    def _get_traffic_multiplier(self, hour: int, day_of_week: int) -> float:
        """Get traffic volume multiplier based on time"""
        # Weekend factor
        weekend_factor = 0.6 if day_of_week >= 5 else 1.0
        
        # Hourly pattern
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            hourly_factor = 1.5
        elif 10 <= hour <= 16:  # Daytime
            hourly_factor = 1.0
        elif 20 <= hour <= 22:  # Evening
            hourly_factor = 0.8
        elif 23 <= hour or hour <= 5:  # Night
            hourly_factor = 0.2
        else:  # Other hours
            hourly_factor = 0.6
        
        return weekend_factor * hourly_factor
    
    def _get_axle_count(self, vehicle_type: str) -> int:
        """Get typical axle count for vehicle type"""
        axle_counts = {
            'passenger_car': 2,
            'light_truck': 2,
            'heavy_truck': 5,
            'bus': 3,
            'motorcycle': 2
        }
        return axle_counts.get(vehicle_type, 2)
    
    def _calculate_dynamic_amplification(self) -> float:
        """Calculate dynamic amplification factor"""
        # Simplified model - in reality depends on bridge properties and vehicle speed
        if ML_AVAILABLE:
            return np.random.normal(1.15, 0.05)  # Typical range 1.1-1.2
        else:
            import random
            return random.uniform(1.10, 1.20)
    
    def _calculate_load_distribution(self, vehicle_count: int) -> Dict[str, float]:
        """Calculate load distribution across bridge spans"""
        # Simplified distribution model
        if vehicle_count == 0:
            return {'span_1': 0, 'span_2': 0, 'span_3': 0}
        
        # Random distribution with some clustering tendency
        if ML_AVAILABLE:
            distribution = np.random.dirichlet([1, 1, 1]) * vehicle_count
        else:
            import random
            total = random.uniform(0.8, 1.2) * vehicle_count
            dist1 = random.uniform(0, total)
            dist2 = random.uniform(0, total - dist1)
            dist3 = total - dist1 - dist2
            distribution = [dist1, dist2, dist3]
        
        return {
            'span_1': round(distribution[0], 1),
            'span_2': round(distribution[1], 1),
            'span_3': round(distribution[2], 1)
        }
    
    def _detect_overload_incidents(self, max_axle_load: float) -> int:
        """Detect overload incidents"""
        overload_threshold = 20.0  # tonnes
        return 1 if max_axle_load > overload_threshold else 0
    
    def _calculate_fatigue_damage(self, total_weight: float, vehicle_count: int) -> float:
        """Calculate equivalent fatigue damage"""
        if vehicle_count == 0:
            return 0.0
        
        # Simplified fatigue damage calculation
        average_weight = total_weight / vehicle_count
        stress_range = average_weight * 0.1  # Simplified stress calculation
        
        # Miner's rule approximation
        fatigue_damage = (stress_range ** 3) * vehicle_count * 1e-9
        return round(fatigue_damage, 6)
    
    def _estimate_average_speed(self, traffic_multiplier: float) -> int:
        """Estimate average vehicle speed"""
        # Speed decreases with traffic density
        free_flow_speed = 80  # km/h
        congested_speed = 30   # km/h
        
        if traffic_multiplier > 1.2:  # Heavy traffic
            speed = congested_speed + (free_flow_speed - congested_speed) * (2.0 - traffic_multiplier) / 0.8
        else:
            speed = free_flow_speed
        
        return max(congested_speed, min(free_flow_speed, int(speed)))
    
    def _calculate_response_frequency(self, effective_load: float) -> float:
        """Calculate bridge response frequency under load"""
        # Simplified: frequency decreases with load
        base_frequency = 2.5  # Hz
        load_factor = 1 - (effective_load / 1000) * 0.1  # Decrease with load
        return round(base_frequency * load_factor, 3)
    
    def _calculate_strain_demand(self, effective_load: float) -> float:
        """Calculate strain demand from load"""
        # Simplified strain calculation
        strain_per_tonne = 2.0  # microstrain per tonne
        return round(effective_load * strain_per_tonne, 1)
    
    def _calculate_safety_margin(self, effective_load: float) -> float:
        """Calculate safety margin"""
        design_capacity = 200.0  # tonnes
        safety_margin = (design_capacity - effective_load) / design_capacity
        return round(max(0, safety_margin), 3)


# =============================================================================
# MULTI-AGENT COORDINATION SYSTEM
# =============================================================================

class MultiAgentCoordinator:
    """Coordinator for multiple infrastructure monitoring agents"""
    
    def __init__(self, config: AgentConfiguration, gcp_integration: GoogleCloudIntegration):
        self.config = config
        self.gcp_integration = gcp_integration
        self.registered_agents = {}
        self.coordination_messages = []
        self.global_state = {}
        self.decision_engine = DecisionEngine()
        
    def register_agent(self, agent_id: str, agent_type: AgentType, capabilities: List[str]):
        """Register an agent with the coordinator"""
        self.registered_agents[agent_id] = {
            'agent_type': agent_type,
            'capabilities': capabilities,
            'last_heartbeat': datetime.datetime.now(),
            'status': 'active'
        }
        
        logging.info(f"Agent {agent_id} registered with type {agent_type.value}")
    
    def receive_agent_update(self, agent_id: str, update_data: Dict[str, Any]):
        """Receive update from an agent"""
        if agent_id not in self.registered_agents:
            logging.warning(f"Received update from unregistered agent: {agent_id}")
            return
        
        # Update agent heartbeat
        self.registered_agents[agent_id]['last_heartbeat'] = datetime.datetime.now()
        
        # Process the update
        self._process_agent_update(agent_id, update_data)
        
        # Check for coordination opportunities
        self._check_coordination_triggers(agent_id, update_data)
    
    def _process_agent_update(self, agent_id: str, update_data: Dict[str, Any]):
        """Process update from agent"""
        # Store in global state
        self.global_state[agent_id] = {
            'last_update': datetime.datetime.now(),
            'data': update_data
        }
        
        # Publish to Pub/Sub for other systems
        if self.gcp_integration:
            coordination_message = {
                'type': 'agent_update',
                'agent_id': agent_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'data': update_data
            }
            self.gcp_integration.publish_sensor_data(coordination_message)
    
    def _check_coordination_triggers(self, agent_id: str, update_data: Dict[str, Any]):
        """Check if coordination is needed between agents"""
        # Example: Infrastructure health alerts should trigger traffic management
        if 'health_analysis' in update_data:
            health_status = update_data['health_analysis'].get('overall_status', 'good')
            
            if health_status in ['poor', 'critical']:
                self._trigger_traffic_coordination(agent_id, update_data)
        
        # Example: Environmental conditions affecting multiple systems
        if 'environmental_conditions' in update_data:
            severe_weather = self._detect_severe_weather(update_data['environmental_conditions'])
            if severe_weather:
                self._trigger_weather_coordination(agent_id, update_data)
    
    def _trigger_traffic_coordination(self, infrastructure_agent_id: str, health_data: Dict[str, Any]):
        """Trigger coordination with traffic management systems"""
        coordination_message = {
            'type': 'infrastructure_health_alert',
            'source_agent': infrastructure_agent_id,
            'target_agents': ['traffic_flow', 'public_safety'],
            'priority': 'high',
            'data': {
                'infrastructure_id': health_data.get('infrastructure_id'),
                'health_status': health_data['health_analysis']['overall_status'],
                'recommended_actions': [
                    'reduce_traffic_load',
                    'implement_speed_restrictions',
                    'consider_lane_closures'
                ]
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.coordination_messages.append(coordination_message)
        
        # Send to other agents via Pub/Sub
        if self.gcp_integration:
            self.gcp_integration.publish_sensor_data(coordination_message)
        
        logging.info(f"Triggered traffic coordination for infrastructure {infrastructure_agent_id}")
    
    def _trigger_weather_coordination(self, agent_id: str, environmental_data: Dict[str, Any]):
        """Trigger coordination for severe weather conditions"""
        coordination_message = {
            'type': 'severe_weather_alert',
            'source_agent': agent_id,
            'target_agents': ['traffic_flow', 'infrastructure_health', 'public_safety'],
            'priority': 'critical',
            'data': {
                'weather_conditions': environmental_data['environmental_conditions'],
                'recommended_actions': [
                    'increase_monitoring_frequency',
                    'prepare_emergency_response',
                    'issue_public_advisories'
                ]
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.coordination_messages.append(coordination_message)
        
        if self.gcp_integration:
            self.gcp_integration.publish_sensor_data(coordination_message)
        
        logging.info(f"Triggered weather coordination from agent {agent_id}")
    
    def _detect_severe_weather(self, environmental_conditions: Dict[str, Any]) -> bool:
        """Detect severe weather conditions"""
        meteorological = environmental_conditions.get('meteorological', {})
        
        # Check for severe conditions
        severe_conditions = [
            meteorological.get('wind_speed_mps', 0) > 20,  # High wind
            meteorological.get('precipitation_mm', 0) > 10,  # Heavy rain
            meteorological.get('temperature_celsius', 20) < -5,  # Extreme cold
            meteorological.get('temperature_celsius', 20) > 40   # Extreme heat
        ]
        
        return any(severe_conditions)
    
    def get_coordination_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get coordination recommendations for an agent"""
        recommendations = []
        
        # Check recent coordination messages
        for message in self.coordination_messages[-10:]:  # Last 10 messages
            if agent_id in message.get('target_agents', []):
                recommendations.append({
                    'type': message['type'],
                    'priority': message['priority'],
                    'actions': message['data'].get('recommended_actions', []),
                    'timestamp': message['timestamp']
                })
        
        return recommendations


class DecisionEngine:
    """AI-powered decision engine for infrastructure management"""
    
    def __init__(self):
        self.decision_rules = self._initialize_decision_rules()
        self.risk_assessment_model = self._initialize_risk_model()
        
    def _initialize_decision_rules(self) -> Dict[str, Any]:
        """Initialize decision rules"""
        return {
            'maintenance_scheduling': {
                'critical_health': {
                    'condition': 'health_score < 0.3',
                    'action': 'immediate_intervention',
                    'priority': 'critical'
                },
                'poor_health': {
                    'condition': 'health_score < 0.6',
                    'action': 'scheduled_maintenance',
                    'priority': 'high'
                },
                'preventive_window': {
                    'condition': 'health_score < 0.8 and predicted_failure_risk > 0.3',
                    'action': 'preventive_maintenance',
                    'priority': 'moderate'
                }
            },
            'traffic_management': {
                'structural_concern': {
                    'condition': 'structural_health < 0.5',
                    'action': 'load_restrictions',
                    'priority': 'high'
                },
                'severe_weather': {
                    'condition': 'weather_severity > 0.8',
                    'action': 'speed_restrictions',
                    'priority': 'moderate'
                }
            }
        }
    
    def _initialize_risk_model(self) -> Dict[str, Any]:
        """Initialize risk assessment model"""
        return {
            'risk_factors': {
                'structural_health': {'weight': 0.4, 'threshold': 0.5},
                'environmental_severity': {'weight': 0.2, 'threshold': 0.7},
                'traffic_load': {'weight': 0.2, 'threshold': 0.8},
                'maintenance_history': {'weight': 0.1, 'threshold': 0.6},
                'age_factor': {'weight': 0.1, 'threshold': 0.7}
            },
            'risk_levels': {
                'low': {'threshold': 0.3, 'actions': ['continue_monitoring']},
                'moderate': {'threshold': 0.6, 'actions': ['increased_monitoring', 'schedule_inspection']},
                'high': {'threshold': 0.8, 'actions': ['immediate_inspection', 'prepare_intervention']},
                'critical': {'threshold': 1.0, 'actions': ['emergency_response', 'traffic_restrictions']}
            }
        }
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make AI-powered decision based on context"""
        try:
            # Assess overall risk
            risk_assessment = self._assess_risk(context)
            
            # Apply decision rules
            applicable_rules = self._find_applicable_rules(context)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_assessment, applicable_rules)
            
            # Calculate confidence
            confidence = self._calculate_confidence(context, recommendations)
            
            return {
                'decision_id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().isoformat(),
                'risk_assessment': risk_assessment,
                'applicable_rules': applicable_rules,
                'recommendations': recommendations,
                'confidence': confidence,
                'context_summary': self._summarize_context(context)
            }
            
        except Exception as e:
            logging.error(f"Error making decision: {e}")
            return {
                'decision_id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e),
                'fallback_recommendation': 'continue_monitoring'
            }
    
    def _assess_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level"""
        risk_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        for factor, config in self.risk_assessment_model['risk_factors'].items():
            factor_value = self._extract_factor_value(context, factor)
            weight = config['weight']
            threshold = config['threshold']
            
            # Calculate risk score for this factor
            if factor_value >= threshold:
                risk_score = 1.0
            else:
                risk_score = factor_value / threshold
            
            risk_scores[factor] = risk_score
            total_weighted_score += risk_score * weight
            total_weight += weight
        
        overall_risk = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine risk level
        risk_level = 'low'
        for level, config in self.risk_assessment_model['risk_levels'].items():
            if overall_risk >= config['threshold']:
                risk_level = level
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'factor_scores': risk_scores,
            'assessment_time': datetime.datetime.now().isoformat()
        }
    
    def _extract_factor_value(self, context: Dict[str, Any], factor: str) -> float:
        """Extract factor value from context"""
        if factor == 'structural_health':
            return 1 - context.get('health_analysis', {}).get('overall_health', 1.0)
        elif factor == 'environmental_severity':
            return context.get('environmental_severity', 0.0)
        elif factor == 'traffic_load':
            return context.get('traffic_load_factor', 0.0)
        elif factor == 'maintenance_history':
            return context.get('maintenance_score', 0.0)
        elif factor == 'age_factor':
            return context.get('age_factor', 0.0)
        else:
            return 0.0
    
    def _find_applicable_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find applicable decision rules"""
        applicable_rules = []
        
        for category, rules in self.decision_rules.items():
            for rule_name, rule_config in rules.items():
                if self._evaluate_condition(context, rule_config['condition']):
                    applicable_rules.append({
                        'category': category,
                        'rule_name': rule_name,
                        'action': rule_config['action'],
                        'priority': rule_config['priority']
                    })
        
        return applicable_rules
    
    def _evaluate_condition(self, context: Dict[str, Any], condition: str) -> bool:
        """Evaluate a condition string against context"""
        try:
            # Simple condition evaluation - in production, use safer evaluation
            health_score = context.get('health_analysis', {}).get('overall_health', 1.0)
            predicted_failure_risk = context.get('predicted_failure_risk', 0.0)
            structural_health = health_score
            weather_severity = context.get('environmental_severity', 0.0)
            
            # Replace variables in condition and evaluate
            condition = condition.replace('health_score', str(health_score))
            condition = condition.replace('predicted_failure_risk', str(predicted_failure_risk))
            condition = condition.replace('structural_health', str(structural_health))
            condition = condition.replace('weather_severity', str(weather_severity))
            
            return eval(condition)
            
        except Exception as e:
            logging.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _generate_recommendations(self, risk_assessment: Dict[str, Any], 
                                applicable_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on risk and rules"""
        recommendations = []
        
        # Add risk-based recommendations
        risk_level = risk_assessment['risk_level']
        risk_config = self.risk_assessment_model['risk_levels'][risk_level]
        
        for action in risk_config['actions']:
            recommendations.append({
                'action': action,
                'basis': 'risk_assessment',
                'priority': risk_level,
                'confidence': 0.8
            })
        
        # Add rule-based recommendations
        for rule in applicable_rules:
            recommendations.append({
                'action': rule['action'],
                'basis': f"rule_{rule['category']}_{rule['rule_name']}",
                'priority': rule['priority'],
                'confidence': 0.9
            })
        
        # Deduplicate and prioritize
        unique_recommendations = {}
        for rec in recommendations:
            action = rec['action']
            if action not in unique_recommendations or rec['confidence'] > unique_recommendations[action]['confidence']:
                                unique_recommendations[action] = rec
        
        return list(unique_recommendations.values())
    
    def _calculate_confidence(self, context: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the decision"""
        # Confidence based on data quality and consistency
        data_quality = context.get('data_quality', 0.8)
        consistency_score = self._calculate_consistency_score(context)
        recommendation_agreement = self._calculate_recommendation_agreement(recommendations)
        
        confidence = (data_quality + consistency_score + recommendation_agreement) / 3
        return round(confidence, 3)
    
    def _calculate_consistency_score(self, context: Dict[str, Any]) -> float:
        """Calculate consistency score of different data sources"""
        # Simplified consistency check
        return 0.85  # Would implement actual consistency checking in production
    
    def _calculate_recommendation_agreement(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate agreement between different recommendation sources"""
        if not recommendations:
            return 0.0
        
        # Simple agreement metric based on priority alignment
        priority_weights = {'critical': 1.0, 'high': 0.8, 'moderate': 0.6, 'low': 0.4}
        avg_weight = sum(priority_weights.get(rec['priority'], 0.5) for rec in recommendations) / len(recommendations)
        
        return avg_weight
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize context for decision record"""
        return {
            'infrastructure_id': context.get('infrastructure_id', 'unknown'),
            'health_status': context.get('health_analysis', {}).get('overall_status', 'unknown'),
            'risk_factors_present': len([k for k, v in context.items() if 'risk' in k.lower() and v > 0.5]),
            'data_sources': list(context.keys())
        }


# =============================================================================
# MAIN ENHANCED INFRASTRUCTURE HEALTH AGENT
# =============================================================================

class EnhancedInfrastructureHealthAgent:
    """Google Cloud-integrated Infrastructure Health AI Agent"""
    
    def __init__(self, config: AgentConfiguration):
        self.config = config
        self.agent_id = config.agent_id
        self.gcp_integration = GoogleCloudIntegration(config.gcp_config) if GCP_AVAILABLE else None
        self.coordinator = MultiAgentCoordinator(config, self.gcp_integration) if config.coordination_enabled else None
        if self.coordinator:
            self.coordinator.register_agent(self.agent_id, config.agent_type, ['health_monitoring', 'predictive_maintenance'])
        # Fix infra_id reference
        self.digital_twins = {
            infra.infrastructure_id: InfrastructureDigitalTwin(infra.infrastructure_id, config)
            for infra in config.infrastructures
        }
        self.sensor_simulators = {
            infra.infrastructure_id: {
                'strain_sensors': IntelligentStrainSensorSimulator(infra.infrastructure_id),
                'vibration_monitors': IntelligentVibrationSensorSimulator(infra.infrastructure_id),
                'environmental_sensors': EnvironmentalSensorSimulator(infra.infrastructure_id),
                'load_sensors': LoadSensorSimulator(infra.infrastructure_id)
            }
            for infra in config.infrastructures
        }
        self.anomaly_detector = AdvancedAnomalyDetector(config.gcp_config)
        self.vertex_ai = VertexAIPredictor(config.gcp_config) if GCP_AVAILABLE else None
        self.is_running = False
        self.monitoring_cycle_count = 0
        self.last_health_assessments: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics = {
            'start_time': None,
            'total_data_points': 0,
            'anomalies_detected': 0,
            'predictions_made': 0,
            'coordination_messages': 0
        }
        self._setup_logging()
        self._initialize_ml_models()

    def _setup_logging(self):
        """Setup enhanced logging with Cloud Logging integration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(f'agent_{self.agent_id}.log')
        ]
        
        # Add Google Cloud Logging if available
        if GCP_AVAILABLE:
            try:
                from google.cloud import logging as cloud_logging
                cloud_client = cloud_logging.Client()
                cloud_handler = cloud_client.get_default_handler()
                handlers.append(cloud_handler)
            except Exception as e:
                print(f"Could not setup Cloud Logging: {e}")
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(f"InfraAgent_{self.agent_id}")
    
    def _initialize_ml_models(self):
        """Initialize ML models with historical data"""
        try:
            if self.gcp_integration:
                for infra_cfg in self.config.infrastructures:
                    infra_id = infra_cfg.infrastructure_id
                    query = f"""
                    SELECT * FROM `{self.config.gcp_config.project_id}.{self.config.gcp_config.dataset_id}.{self.config.gcp_config.sensor_data_table}`
                    WHERE infrastructure_id = '{infra_id}'
                    AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                    ORDER BY timestamp DESC
                    LIMIT 10000
                    """
                    historical_data = self.gcp_integration.query_bigquery(query)
                    if not historical_data.empty:
                        self.anomaly_detector.train_models(historical_data)
            self.logger.info("[Agent] ML models initialized.")
        except Exception as e:
            self.logger.error(f"[Agent] ML model initialization error: {e}")

    def start(self):
        self.logger.info(f"[Agent] Starting {self.agent_id}")
        self.is_running = True
        self.performance_metrics['start_time'] = datetime.datetime.now()
        for infra_cfg in self.config.infrastructures:
            infra_id = infra_cfg.infrastructure_id
            simulators = self.sensor_simulators.get(infra_id, {})
            infra_rates = getattr(infra_cfg, 'sensor_sampling_rates', {})
            agent_rates = getattr(self.config, 'sensor_sampling_rates', {})
            for sensor_type, simulator in simulators.items():
                interval = infra_rates.get(sensor_type, agent_rates.get(sensor_type, 60))
                t = threading.Thread(target=self._sensor_collection_loop, args=(infra_id, sensor_type, simulator, interval), daemon=True)
                t.start()
                self.logger.info(f"[Agent] Started {sensor_type} sensor for {infra_id} every {interval}s")
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        if self.coordinator:
            threading.Thread(target=self._coordination_loop, daemon=True).start()
        try:
            while self.is_running:
                time.sleep(30)
                self._print_status()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.logger.info("Stopping Enhanced Infrastructure Health Agent")
        self.is_running = False
    
    def _sensor_collection_loop(self, infra_id: str, sensor_type: str, simulator, interval: int):
        while self.is_running:
            try:
                data = simulator.get_current_data()
                self._process_sensor_data(infra_id, sensor_type, data)
                self.performance_metrics['total_data_points'] += 1
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"[Agent] Sensor loop error for {infra_id} {sensor_type}: {e}")
                time.sleep(30)

    def _process_sensor_data(self, infra_id: str, sensor_type: str, sensor_data: Dict[str, Any]):
        try:
            processed = {
                'infrastructure_id': infra_id,
                'sensor_type': sensor_type,
                'timestamp': datetime.datetime.now().isoformat(),
                'data': sensor_data
            }
            anomalies = self.anomaly_detector.detect_anomalies({sensor_type: sensor_data})
            if anomalies:
                processed['anomaly_analysis'] = anomalies
                for res in anomalies.values():
                    if res.get('alert_level') in ['high', 'critical']:
                        self.performance_metrics['anomalies_detected'] += 1
            if self.gcp_integration:
                self._store_in_bigquery(processed)
                self.gcp_integration.publish_sensor_data(processed)
            twin = self.digital_twins.get(infra_id)
            if twin:
                twin.update_state(processed, anomalies or {})
        except Exception as e:
            self.logger.error(f"[Agent] Processing error {infra_id} {sensor_type}: {e}")

    def _monitoring_loop(self):
        while self.is_running:
            try:
                self.monitoring_cycle_count += 1
                self.logger.info(f"[Agent] Monitoring cycle {self.monitoring_cycle_count}")
                assessments = {}
                for infra_id, twin in self.digital_twins.items():
                    health = self._perform_health_assessment(infra_id)
                    assessments[infra_id] = health
                    self.performance_metrics['predictions_made'] += 1
                self.last_health_assessments = assessments
                avg_health = sum(h.get('overall_health', 0) for h in assessments.values()) / max(len(assessments), 1)
                self.logger.info(f"[Agent] Average health {avg_health:.3f}")
                for infra_id, health in assessments.items():
                    decision = self._make_intelligent_decisions(infra_id, health)
                    if self.coordinator:
                        self._update_coordination(infra_id, health, decision)
                time.sleep(self.config.health_assessment_interval)
            except Exception as e:
                self.logger.error(f"[Agent] Monitoring loop error: {e}")
                time.sleep(60)

    def _perform_health_assessment(self, infra_id: str) -> Dict[str, Any]:
        try:
            twin = self.digital_twins.get(infra_id)
            if not twin:
                return {}
            state = twin.get_current_state()
            future_30 = twin.predict_future_state(30)
            future_90 = twin.predict_future_state(90)
            sh = state['current_state'].get('structural_health', 1.0)
            oc = state['current_state'].get('operational_capacity', 1.0)
            sl = state['current_state'].get('safety_level', 1.0)
            overall = (sh + oc + sl) / 3
            if overall >= 0.9:
                hs = HealthStatus.EXCELLENT
            elif overall >= 0.75:
                hs = HealthStatus.GOOD
            elif overall >= 0.6:
                hs = HealthStatus.FAIR
            elif overall >= 0.4:
                hs = HealthStatus.POOR
            else:
                hs = HealthStatus.CRITICAL
            risk = self._assess_risk_factors(state)
            maints = self._generate_maintenance_recommendations(state, future_30, future_90)
            qa = self._calculate_data_quality_score(infra_id)
            return {
                'infrastructure_id': infra_id,
                'assessment_id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().isoformat(),
                'overall_health': round(overall,3),
                'health_status': hs.value,
                'component_health': {'structural_health': round(sh,3), 'operational_capacity': round(oc,3), 'safety_level': round(sl,3)},
                'risk_assessment': risk,
                'future_predictions': {'30_days': future_30, '90_days': future_90},
                'maintenance_recommendations': maints,
                'confidence_level': state['current_state'].get('confidence_level', 0.8),
                'data_quality_score': qa,
                'simulation_scenarios': state.get('simulation_scenarios', [])
            }
        except Exception as e:
            self.logger.error(f"[Agent] Health assessment error {infra_id}: {e}")
            return {'infrastructure_id': infra_id, 'timestamp': datetime.datetime.now().isoformat(), 'error': str(e), 'overall_health': 0.5, 'health_status': HealthStatus.FAIR.value}
    
    def _assess_risk_factors(self, digital_twin_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various risk factors"""
        risks = {
            'structural_degradation': max(0, 1 - digital_twin_state['current_state'].get('structural_health', 1.0)),
            'environmental_exposure': 0.3,  # Would calculate from environmental data
            'age_related_risk': 0.2,  # Would calculate from infrastructure age
            'traffic_overload_risk': 0.1,  # Would calculate from load sensor data
            'maintenance_deficit': 0.15  # Would calculate from maintenance history
        }
        
        # Calculate overall risk score
        risk_weights = {
            'structural_degradation': 0.4,
            'environmental_exposure': 0.2,
            'age_related_risk': 0.15,
            'traffic_overload_risk': 0.15,
            'maintenance_deficit': 0.1
        }
        
        overall_risk = sum(risks[factor] * risk_weights[factor] for factor in risks)
        
        # Determine risk level
        if overall_risk >= 0.8:
            risk_level = 'critical'
        elif overall_risk >= 0.6:
            risk_level = 'high'
        elif overall_risk >= 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'overall_risk_score': round(overall_risk, 3),
            'risk_level': risk_level,
            'risk_factors': risks,
            'primary_risk_driver': max(risks, key=risks.get)
        }
    
    def _generate_maintenance_recommendations(self, current_state: Dict[str, Any], 
                                           future_30d: Dict[str, Any], 
                                           future_90d: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered maintenance recommendations"""
        recommendations = []
        
        # Current state-based recommendations
        structural_health = current_state['current_state'].get('structural_health', 1.0)
        
        if structural_health < 0.5:
            recommendations.append({
                'action_type': 'emergency_inspection',
                'priority': Priority.CRITICAL.value,
                'timeframe': 'immediate',
                'estimated_cost': 15000,
                'description': 'Emergency structural inspection required due to poor health indicators',
                'risk_mitigation': 0.7
            })
        elif structural_health < 0.7:
            recommendations.append({
                'action_type': 'detailed_inspection',
                'priority': Priority.HIGH.value,
                'timeframe': '7_days',
                'estimated_cost': 8000,
                'description': 'Detailed structural inspection recommended',
                'risk_mitigation': 0.5
            })
        
        # Predictive recommendations based on future states
        future_structural_health = future_90d.get('structural_health', structural_health)
        if future_structural_health < 0.6 and structural_health > 0.6:
            recommendations.append({
                'action_type': 'preventive_maintenance',
                'priority': Priority.MODERATE.value,
                'timeframe': '60_days',
                'estimated_cost': 25000,
                'description': 'Preventive maintenance recommended to avoid future degradation',
                'risk_mitigation': 0.6
            })
        
        # Component-specific recommendations
        for component, health in current_state.get('component_health', {}).items():
            if health < 0.5:
                recommendations.append({
                    'action_type': 'component_replacement',
                    'component': component,
                    'priority': Priority.HIGH.value,
                    'timeframe': '30_days',
                    'estimated_cost': 35000,
                    'description': f'Replacement of {component} recommended due to poor condition',
                    'risk_mitigation': 0.8
                })
        
        return recommendations
    
    def _calculate_data_quality_score(self, infra_id: str) -> float:
        """Calculate overall data quality score"""
        # Simplified calculation - in production would analyze sensor health, data completeness, etc.
        base_quality = 0.85
        
        # Adjust based on sensor performance
        sensor_health_scores = []
        for simulator in self.sensor_simulators.get(infra_id, {}).values():
            if hasattr(simulator, 'degradation_state'):
                sensor_health_scores.extend(simulator.degradation_state.values())
        
        avg_sensor_health = sum(sensor_health_scores) / len(sensor_health_scores) if sensor_health_scores else 1.0
        return round(base_quality * avg_sensor_health, 3)
    
    def _make_intelligent_decisions(self, infra_id: str, health_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Make AI-powered decisions based on health assessment"""
        try:
            # Create decision context
            context = {
                'infrastructure_id': infra_id,
                'health_analysis': health_assessment,
                'predicted_failure_risk': health_assessment['risk_assessment']['overall_risk_score'],
                'environmental_severity': 0.3,  # Would get from environmental sensors
                'traffic_load_factor': 0.6,  # Would get from load sensors
                'maintenance_score': 0.7,  # Would get from maintenance history
                'age_factor': 0.2,  # Would calculate from infrastructure age
                'data_quality': health_assessment.get('data_quality_score', 0.8)
            }
            
            # Use decision engine if coordinator is available
            if self.coordinator:
                decision = self.coordinator.decision_engine.make_decision(context)
            else:
                decision = self._make_simple_decision(context)
            
            self.performance_metrics['predictions_made'] += 1
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making intelligent decisions: {e}")
            return {
                'decision_id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().isoformat(),
                'error': str(e),
                'fallback_decision': 'continue_monitoring'
            }
    
    def _make_simple_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple decision making when decision engine is not available"""
        health_score = context['health_analysis']['overall_health']
        risk_score = context['predicted_failure_risk']
        
        if health_score < 0.3 or risk_score > 0.8:
            priority = 'critical'
            actions = ['emergency_response', 'immediate_inspection']
        elif health_score < 0.6 or risk_score > 0.6:
            priority = 'high'
            actions = ['schedule_inspection', 'prepare_maintenance']
        elif health_score < 0.8 or risk_score > 0.4:
            priority = 'moderate'
            actions = ['increased_monitoring', 'plan_maintenance']
        else:
            priority = 'low'
            actions = ['continue_monitoring']
        
        return {
            'decision_id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat(),
            'recommended_actions': actions,
            'priority': priority,
            'confidence': 0.7
        }
    
    def _update_coordination(self, infra_id: str, health_assessment: Dict[str, Any], decision: Dict[str, Any]):
        """Update multi-agent coordination"""
        try:
            coordination_update = {
                'agent_id': self.agent_id,
                'infrastructure_id': infra_id,
                'health_assessment': health_assessment,
                'decision': decision,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.coordinator.receive_agent_update(self.agent_id, coordination_update)
            self.performance_metrics['coordination_messages'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating coordination: {e}")
    
    def _print_status(self):
        uptime = datetime.datetime.now() - self.performance_metrics['start_time'] if self.performance_metrics['start_time'] else datetime.timedelta()
        print(f"\n=== Agent Status: {self.agent_id} ===")
        print(f"Uptime: {uptime}")
        print(f"Assets Monitored: {len(self.last_health_assessments)}")
        print(f"Data Points: {self.performance_metrics['total_data_points']}")
        print(f"Anomalies: {self.performance_metrics['anomalies_detected']}")
        print(f"Predictions: {self.performance_metrics['predictions_made']}")
        print(f"Coordination Messages: {self.performance_metrics['coordination_messages']}")
        print("Assets Health:")
        total_health = 0.0
        count = 0
        for infra_id, ha in self.last_health_assessments.items():
            hs = ha.get('health_status', 'UNKNOWN')
            oh = ha.get('overall_health', 0)
            print(f"  {infra_id}: {hs} ({oh:.3f})")
            total_health += oh
            count += 1
        if count > 0:
            print(f"System-wide Average Health: {total_health/count:.3f}")
        print("="*40)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'agent_info': {
                'agent_id': self.agent_id,
                'infrastructure_id': self.infrastructure_id,
                'agent_type': self.config.agent_type.value,
                'uptime_seconds': (datetime.datetime.now() - self.performance_metrics['start_time']).total_seconds()
            },
            'performance_metrics': self.performance_metrics,
            'current_health': self.last_health_assessment,
            'gcp_integration_status': {
                'bigquery_connected': bool(self.gcp_integration and self.gcp_integration.bigquery_client),
                'pubsub_connected': bool(self.gcp_integration and self.gcp_integration.publisher),
                'monitoring_connected': bool(self.gcp_integration and self.gcp_integration.monitoring_client),
                'vertex_ai_connected': bool(self.vertex_ai and self.vertex_ai.model_endpoint)
            },
            'coordination_status': {
                'enabled': bool(self.coordinator),
                'registered_agents': len(self.coordinator.registered_agents) if self.coordinator else 0
            },
            'digital_twin_status': self.digital_twin.get_current_state()
        }


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def create_demo_gcp_config() -> GCPConfiguration:
    """Create demo GCP configuration"""
    return GCPConfiguration(
        project_id="infrastructure-monitoring-demo",  # Replace with your project ID
        region="us-central1",
        pubsub_topic="infrastructure-monitoring",
        dataset_id="infrastructure_monitoring",
        bucket_name="infrastructure-monitoring-data"
    )

def create_demo_agent_config() -> AgentConfiguration:
    """Create demo agent configuration"""
    return AgentConfiguration(
        agent_id="infra_health_agent_001",
        agent_type=AgentType.INFRASTRUCTURE_HEALTH,
        infrastructure_id="golden_gate_bridge_main_span",
        gcp_config=create_demo_gcp_config(),
        health_assessment_interval=60,  # 1 minute for demo
        digital_twin_enabled=True,
        simulation_enabled=True,
        coordination_enabled=True
    )

def demo_enhanced_agent():
    """Demonstrate the enhanced Google Cloud-integrated agent"""
    print("=== Google Cloud-Integrated Infrastructure Health AI Agent Demo ===\n")
    
    # Create configuration
    config = create_demo_agent_config()
    
    print("Configuration:")
    print(f"  Agent ID: {config.agent_id}")
    print(f"  Infrastructure ID: {config.infrastructure_id}")
    print(f"  GCP Project: {config.gcp_config.project_id}")
    print(f"  Agent Type: {config.agent_type.value}")
    print(f"  Digital Twin: {'Enabled' if config.digital_twin_enabled else 'Disabled'}")
    print(f"  Simulation: {'Enabled' if config.simulation_enabled else 'Disabled'}")
    print(f"  Multi-Agent Coordination: {'Enabled' if config.coordination_enabled else 'Disabled'}")
    print()
    
    # Initialize agent
    print("Initializing Enhanced Infrastructure Health Agent...")
    agent = EnhancedInfrastructureHealthAgent(config)
    
    print("✓ Agent initialized successfully")
    print(f"✓ GCP Integration: {'Available' if GCP_AVAILABLE else 'Simulated (install google-cloud libraries)'}")
    print(f"✓ ML Capabilities: {'Available' if ML_AVAILABLE else 'Limited (install scikit-learn)'}")
    print(f"✓ Sensor Simulators: {len(agent.sensor_simulators)} types")
    print()
    
    # Start agent
    print("Starting enhanced agent (will run for 5 minutes for demo)...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        # Start agent in separate thread
        agent_thread = threading.Thread(target=agent.start, daemon=True)
        agent_thread.start()
        
        # Let it run for demo
        demo_duration = 300  # 5 minutes
        for i in range(demo_duration):
            time.sleep(1)
            
            # Show progress every 30 seconds
            if i % 30 == 29:
                remaining = demo_duration - i - 1
                print(f"Demo running... {remaining} seconds remaining")
                
                # Show detailed status every 2 minutes
                if i % 120 == 119:
                    print("\n--- Comprehensive Status Report ---")
                    status = agent.get_comprehensive_status()
                    
                    print(f"Agent Uptime: {status['agent_info']['uptime_seconds']:.0f} seconds")
                    print(f"Data Points Collected: {status['performance_metrics']['total_data_points']}")
                    print(f"Anomalies Detected: {status['performance_metrics']['anomalies_detected']}")
                    print(f"Predictions Made: {status['performance_metrics']['predictions_made']}")
                    
                    if status['current_health']:
                        health = status['current_health']
                        print(f"Current Health: {health['overall_health']:.3f} ({health['health_status']})")
                        print(f"Risk Level: {health['risk_assessment']['risk_level']}")
                        print(f"Recommendations: {len(health.get('maintenance_recommendations', []))}")
                    
                    print("GCP Integration Status:")
                    gcp_status = status['gcp_integration_status']
                    for service, connected in gcp_status.items():
                        status_icon = "✓" if connected else "✗"
                        print(f"  {service}: {status_icon}")
                    
                    print("--- End Status Report ---\n")
        
        # Generate final report
        print("\n=== FINAL COMPREHENSIVE REPORT ===")
        final_status = agent.get_comprehensive_status()
        
        print(f"Agent completed {demo_duration} second demonstration")
        print(f"Total monitoring cycles: {agent.monitoring_cycle_count}")
        print(f"Performance metrics:")
        for metric, value in final_status['performance_metrics'].items():
            if metric != 'start_time':
                print(f"  {metric}: {value}")
        
        if final_status['current_health']:
            print(f"\nFinal Health Assessment:")
            health = final_status['current_health']
            print(f"  Overall Health: {health['overall_health']:.3f}")
            print(f"  Health Status: {health['health_status']}")
            print(f"  Risk Level: {health['risk_assessment']['risk_level']}")  
            print(f"  Primary Risk Driver: {health['risk_assessment']['primary_risk_driver']}")
            
            if health.get('maintenance_recommendations'):
                print(f"  Maintenance Recommendations:")
                for rec in health['maintenance_recommendations'][:3]:  # Show first 3
                    print(f"    - {rec['action_type']} ({rec['priority']}) - ${rec['estimated_cost']:,}")
        
        print(f"\nDigital Twin Status:")
        dt_status = final_status['digital_twin_status']
        print(f"  Infrastructure ID: {dt_status['infrastructure_id']}")
        print(f"  Component Health Scores:")
        for component, health_score in dt_status.get('component_health', {}).items():
            print(f"    {component}: {health_score:.3f}")
        
        if dt_status.get('simulation_scenarios'):
            print(f"  Simulation Scenarios: {len(dt_status['simulation_scenarios'])}")
        
        # Stop agent
        agent.stop()
        print("\n✓ Enhanced Google Cloud-integrated demo completed successfully!")
        
        if not GCP_AVAILABLE:
            print("\nNote: To enable full GCP integration, install Google Cloud libraries:")
            print("  pip install google-cloud-pubsub google-cloud-bigquery google-cloud-storage")
            print("  pip install google-cloud-monitoring google-cloud-aiplatform")
        
    except KeyboardInterrupt:
        agent.stop()
        print("\nDemo stopped by user")
    except Exception as e:
        agent.stop()
        print(f"Demo error: {e}")


if __name__ == "__main__":
    print("Enhanced Infrastructure Health AI Agent with Google Cloud Integration")
    print("Based on the Agentic AI Day urban infrastructure monitoring vision")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"  Google Cloud libraries: {'✓' if GCP_AVAILABLE else '✗ (pip install google-cloud-*)'}")
    print(f"  ML libraries: {'✓' if ML_AVAILABLE else '✗ (pip install scikit-learn pandas numpy)'}")
    print(f"  Web libraries: {'✓' if WEB_AVAILABLE else '✗ (pip install flask plotly dash)'}")
    print()
    
    # Run demo
    demo_enhanced_agent()
    
    def _coordination_loop(self):
        """Handle multi-agent coordination"""
        while self.is_running:
            try:
                # Get coordination recommendations
                recommendations = self.coordinator.get_coordination_recommendations(self.agent_id)
                
                if recommendations:
                    self.logger.info(f"Received {len(recommendations)} coordination recommendations")
                    
                    for rec in recommendations:
                        self._handle_coordination_recommendation(rec)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(60)
    
    def _handle_coordination_recommendation(self, recommendation: Dict[str, Any]):
        """Handle coordination recommendation from other agents"""
        try:
            rec_type = recommendation.get('type')
            priority = recommendation.get('priority', 'low')
            actions = recommendation.get('actions', [])
            
            self.logger.info(f"Processing coordination recommendation: {rec_type} (Priority: {priority})")
            
            # Handle different types of recommendations
            if rec_type == 'severe_weather_alert':
                self._handle_weather_coordination(recommendation)
            elif rec_type == 'traffic_management_request':
                self._handle_traffic_coordination(recommendation)
            elif rec_type == 'emergency_response':
                self._handle_emergency_coordination(recommendation)
            
            # Log coordination action
            if self.gcp_integration:
                coordination_log = {
                    'agent_id': self.agent_id,
                    'recommendation_type': rec_type,
                    'priority': priority,
                    'actions_taken': actions,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self.gcp_integration.publish_sensor_data(coordination_log)
                
        except Exception as e:
            self.logger.error(f"Error handling coordination recommendation: {e}")
    
    def _handle_weather_coordination(self, recommendation: Dict[str, Any]):
        """Handle severe weather coordination"""
        # Increase monitoring frequency
        for sensor_type in self.config.sensor_sampling_rates:
            self.config.sensor_sampling_rates[sensor_type] = min(
                self.config.sensor_sampling_rates[sensor_type], 30
            )
        
        self.logger.info("Increased monitoring frequency due to severe weather alert")
    
    def _handle_traffic_coordination(self, recommendation: Dict[str, Any]):
        """Handle traffic management coordination"""
        # Adjust load monitoring based on traffic conditions
        if 'reduce_traffic_load' in recommendation.get('actions', []):
            self.logger.info("Acknowledged traffic load reduction request")
            # Would send confirmation to traffic management system
    
    def _handle_emergency_coordination(self, recommendation: Dict[str, Any]):
        """Handle emergency response coordination"""
        self.logger.warning("Emergency coordination activated")
        
        # Switch to emergency monitoring mode
        self.config.health_assessment_interval = 60  # 1 minute intervals
        
        # Send immediate health assessment
        if self.last_health_assessment:
            emergency_report = {
                'type': 'emergency_health_report',
                'infrastructure_id': self.infrastructure_id,
                'health_assessment': self.last_health_assessment,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            if self.gcp_integration:
                self.gcp_integration.publish_sensor_data(emergency_report)
    
    def _print_status(self):
        """Print current agent status"""
        uptime = datetime.datetime.now() - self.performance_metrics['start_time']
        
        print(f"\n=== Enhanced Infrastructure Health Agent Status ===")
        print(f"Agent ID: {self.agent_id}")
        print(f"Infrastructure ID: {self.infrastructure_id}")
        print(f"Uptime: {uptime}")
        print(f"Monitoring Cycles: {self.monitoring_cycle_count}")
        print(f"Data Points Collected: {self.performance_metrics['total_data_points']}")
        print(f"Anomalies Detected: {self.performance_metrics['anomalies_detected']}")
        print(f"Predictions Made: {self.performance_metrics['predictions_made']}")
        print(f"Coordination Messages: {self.performance_metrics['coordination_messages']}")
        
        if self.last_health_assessment:
            health = self.last_health_assessment
            print(f"\nCurrent Health Status:")
            print(f"  Overall Health: {health['overall_health']:.3f} ({health['health_status']})")
            print(f"  Risk Level: {health['risk_assessment']['risk_level']}")
            print(f"  Structural Health: {health['component_health']['structural_health']:.3f}")
            print(f"  Operational Capacity: {health['component_health']['operational_capacity']:.3f}")
            print(f"  Safety Level: {health['component_health']['safety_level']:.3f}")
            print(f"  Confidence: {health['confidence_level']:.3f}")
            
            if health.get('maintenance_recommendations'):
                print(f"  Active Recommendations: {len(health['maintenance_recommendations'])}")
        
        print(f"\nGCP Services:")
        print(f"  BigQuery: {'✓' if self.gcp_integration and self.gcp_integration.bigquery_client else '✗'}")
        print(f"  Pub/Sub: {'✓' if self.gcp_integration and self.gcp_integration.publisher else '✗'}")
        print(f"  Cloud Monitoring: {'✓' if self.gcp_integration and self.gcp_integration.monitoring_client else '✗'}")
        print(f"  Vertex AI: {'✓' if self.vertex_ai and self.vertex_ai.model_endpoint else '✗'}")
        
        print("=" * 55)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'agent_info': {
                'agent_id': self.agent_id,
                'infrastructure_id': self.infrastructure_id,
                'agent_type': self.config.agent_type.value,
                'uptime_seconds': (datetime.datetime.now() - self.performance_metrics['start_time']).total_seconds()
            },
            'performance_metrics': self.performance_metrics,
            'current_health': self.last_health_assessment,
            'gcp_integration_status': {
                'bigquery_connected': bool(self.gcp_integration and self.gcp_integration.bigquery_client),
                'pubsub_connected': bool(self.gcp_integration and self.gcp_integration.publisher),
                'monitoring_connected': bool(self.gcp_integration and self.gcp_integration.monitoring_client),
                'vertex_ai_connected': bool(self.vertex_ai and self.vertex_ai.model_endpoint)
            },
            'coordination_status': {
                'enabled': bool(self.coordinator),
                'registered_agents': len(self.coordinator.registered_agents) if self.coordinator else 0
            },
            'digital_twin_status': self.digital_twin.get_current_state()
        }


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def create_demo_gcp_config() -> GCPConfiguration:
    """Create demo GCP configuration"""
    return GCPConfiguration(
        project_id="infrastructure-monitoring-demo",  # Replace with your project ID
        region="us-central1",
        pubsub_topic="infrastructure-monitoring",
        dataset_id="infrastructure_monitoring",
        bucket_name="infrastructure-monitoring-data"
    )

def create_demo_agent_config() -> AgentConfiguration:
    """Create demo agent configuration"""
    return AgentConfiguration(
        agent_id="infra_health_agent_001",
        agent_type=AgentType.INFRASTRUCTURE_HEALTH,
        infrastructure_id="golden_gate_bridge_main_span",
        gcp_config=create_demo_gcp_config(),
        health_assessment_interval=60,  # 1 minute for demo
        digital_twin_enabled=True,
        simulation_enabled=True,
        coordination_enabled=True
    )

def demo_enhanced_agent():
    """Demonstrate the enhanced Google Cloud-integrated agent"""
    print("=== Google Cloud-Integrated Infrastructure Health AI Agent Demo ===\n")
    
    # Create configuration
    config = create_demo_agent_config()
    
    print("Configuration:")
    print(f"  Agent ID: {config.agent_id}")
    print(f"  Infrastructure ID: {config.infrastructure_id}")
    print(f"  GCP Project: {config.gcp_config.project_id}")
    print(f"  Agent Type: {config.agent_type.value}")
    print(f"  Digital Twin: {'Enabled' if config.digital_twin_enabled else 'Disabled'}")
    print(f"  Simulation: {'Enabled' if config.simulation_enabled else 'Disabled'}")
    print(f"  Multi-Agent Coordination: {'Enabled' if config.coordination_enabled else 'Disabled'}")
    print()
    
    # Initialize agent
    print("Initializing Enhanced Infrastructure Health Agent...")
    agent = EnhancedInfrastructureHealthAgent(config)
    
    print("✓ Agent initialized successfully")
    print(f"✓ GCP Integration: {'Available' if GCP_AVAILABLE else 'Simulated (install google-cloud libraries)'}")
    print(f"✓ ML Capabilities: {'Available' if ML_AVAILABLE else 'Limited (install scikit-learn)'}")
    print(f"✓ Sensor Simulators: {len(agent.sensor_simulators)} types")
    print()
    
    # Start agent
    print("Starting enhanced agent (will run for 5 minutes for demo)...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        # Start agent in separate thread
        agent_thread = threading.Thread(target=agent.start, daemon=True)
        agent_thread.start()
        
        # Let it run for demo
        demo_duration = 300  # 5 minutes
        for i in range(demo_duration):
            time.sleep(1)
            
            # Show progress every 30 seconds
            if i % 30 == 29:
                remaining = demo_duration - i - 1
                print(f"Demo running... {remaining} seconds remaining")
                
                # Show detailed status every 2 minutes
                if i % 120 == 119:
                    print("\n--- Comprehensive Status Report ---")
                    status = agent.get_comprehensive_status()
                    
                    print(f"Agent Uptime: {status['agent_info']['uptime_seconds']:.0f} seconds")
                    print(f"Data Points Collected: {status['performance_metrics']['total_data_points']}")
                    print(f"Anomalies Detected: {status['performance_metrics']['anomalies_detected']}")
                    print(f"Predictions Made: {status['performance_metrics']['predictions_made']}")
                    
                    if status['current_health']:
                        health = status['current_health']
                        print(f"Current Health: {health['overall_health']:.3f} ({health['health_status']})")
                        print(f"Risk Level: {health['risk_assessment']['risk_level']}")
                        print(f"Recommendations: {len(health.get('maintenance_recommendations', []))}")
                    
                    print("GCP Integration Status:")
                    gcp_status = status['gcp_integration_status']
                    for service, connected in gcp_status.items():
                        status_icon = "✓" if connected else "✗"
                        print(f"  {service}: {status_icon}")
                    
                    print("--- End Status Report ---\n")
        
        # Generate final report
        print("\n=== FINAL COMPREHENSIVE REPORT ===")
        final_status = agent.get_comprehensive_status()
        
        print(f"Agent completed {demo_duration} second demonstration")
        print(f"Total monitoring cycles: {agent.monitoring_cycle_count}")
        print(f"Performance metrics:")
        for metric, value in final_status['performance_metrics'].items():
            if metric != 'start_time':
                print(f"  {metric}: {value}")
        
        if final_status['current_health']:
            print(f"\nFinal Health Assessment:")
            health = final_status['current_health']
            print(f"  Overall Health: {health['overall_health']:.3f}")
            print(f"  Health Status: {health['health_status']}")
            print(f"  Risk Level: {health['risk_assessment']['risk_level']}")  
            print(f"  Primary Risk Driver: {health['risk_assessment']['primary_risk_driver']}")
            
            if health.get('maintenance_recommendations'):
                print(f"  Maintenance Recommendations:")
                for rec in health['maintenance_recommendations'][:3]:  # Show first 3
                    print(f"    - {rec['action_type']} ({rec['priority']}) - ${rec['estimated_cost']:,}")
        
        print(f"\nDigital Twin Status:")
        dt_status = final_status['digital_twin_status']
        print(f"  Infrastructure ID: {dt_status['infrastructure_id']}")
        print(f"  Component Health Scores:")
        for component, health_score in dt_status.get('component_health', {}).items():
            print(f"    {component}: {health_score:.3f}")
        
        if dt_status.get('simulation_scenarios'):
            print(f"  Simulation Scenarios: {len(dt_status['simulation_scenarios'])}")
        
        # Stop agent
        agent.stop()
        print("\n✓ Enhanced Google Cloud-integrated demo completed successfully!")
        
        if not GCP_AVAILABLE:
            print("\nNote: To enable full GCP integration, install Google Cloud libraries:")
            print("  pip install google-cloud-pubsub google-cloud-bigquery google-cloud-storage")
            print("  pip install google-cloud-monitoring google-cloud-aiplatform")
        
    except KeyboardInterrupt:
        agent.stop()
        print("\nDemo stopped by user")
    except Exception as e:
        agent.stop()
        print(f"Demo error: {e}")


if __name__ == "__main__":
    print("Enhanced Infrastructure Health AI Agent with Google Cloud Integration")
    print("Based on the Agentic AI Day urban infrastructure monitoring vision")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"  Google Cloud libraries: {'✓' if GCP_AVAILABLE else '✗ (pip install google-cloud-*)'}")
    print(f"  ML libraries: {'✓' if ML_AVAILABLE else '✗ (pip install scikit-learn pandas numpy)'}")
    print(f"  Web libraries: {'✓' if WEB_AVAILABLE else '✗ (pip install flask plotly dash)'}")
    print()
    
    # Run demo
    demo_enhanced_agent()
