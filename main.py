import threading
import time
import random
from datetime import datetime
import torch
import numpy as np
import os
from typing import Dict, List, Optional
import warnings
import firebase_admin
from firebase_admin import credentials, firestore

from main import FirebaseManager
warnings.filterwarnings('ignore')

try:
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install with: pip install torch")

# --- Existing Traffic Agent Classes (Modified Where Noted) ---

class LocalMessageQueue:
    """Simulates a message queue for local development"""
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def publish(self, message):
        with self.lock:
            self.queue.append(message)

    def consume(self):
        with self.lock:
            return self.queue.pop(0) if self.queue else None

class TrafficSensorSimulator:
    """Simulates traffic sensor data with congestion scenario"""
    # def __init__(self):
    #     self.sensor_locations = [
    #         {'id': 'TS001', 'location': 'Main St & 1st Ave', 'type': 'intersection'},
    #         {'id': 'TS002', 'location': 'Highway 101 Mile 15', 'type': 'highway'},
    #         {'id': 'TS003', 'location': 'Downtown Bridge', 'type': 'bridge'},
    #         {'id': 'TS004', 'location': 'University Campus Gate', 'type': 'arterial'},
    #         {'id': 'TS005', 'location': 'Shopping Mall Entrance', 'type': 'commercial'}
    #     ]

    def __init__(self, firebase_manager: FirebaseManager = None):
        self.firebase = firebase_manager
        self.default_locations = [
            {'id': 'TS001', 'location': 'Main St & 1st Ave', 'type': 'intersection'},
            {'id': 'TS002', 'location': 'Highway 101 Mile 15', 'type': 'highway'},
            {'id': 'TS003', 'location': 'Downtown Bridge', 'type': 'bridge'},
            {'id': 'TS004', 'location': 'University Campus Gate', 'type': 'arterial'},
            {'id': 'TS005', 'location': 'Shopping Mall Entrance', 'type': 'commercial'}
        ]
        self.sensor_locations = self._initialize_sensor_locations()

    def _initialize_sensor_locations(self) -> List[Dict]:
        """Initialize sensor locations from Firebase or use defaults"""
        if not self.firebase:
            return self.default_locations
            
        try:
            # Initialize default sensors if needed
            self.firebase.initialize_default_sensors(self.default_locations)
            
            # Get all sensor locations
            docs = self.firebase.db.collection('sensor_locations').stream()
            locations = [doc.to_dict() for doc in docs]
            return locations if locations else self.default_locations
        except Exception as e:
            print(f"Error initializing sensor locations: {e}")
            return self.default_locations

    def get_current_data(self):
        data = {}
        current_hour = datetime.now().hour
        for sensor in self.sensor_locations:
            base_flow = self._get_base_flow(sensor['type'], current_hour)
            flow_variation = random.uniform(0.8, 1.2)
            vehicle_count = int(base_flow * flow_variation)
            capacity = self._get_capacity(sensor['type'])
            # Simulate severe congestion for TS001 to trigger an issue
            if sensor['id'] == 'TS001':
                vehicle_count = int(capacity * 0.95)  # Increased to ensure congestion
            density = vehicle_count / capacity
            average_speed = self._calculate_speed(density, sensor['type'])
            data[sensor['id']] = {
                'location': sensor['location'],
                'vehicle_count': vehicle_count,
                'average_speed': average_speed,
                'density': density,
                'congestion_level': min(density, 1.0),
                'timestamp': datetime.now().isoformat()
            }
        return data

    def _get_base_flow(self, sensor_type, hour):
        patterns = {
            'highway': [50, 30, 20, 15, 20, 40, 80, 120, 100, 90, 95, 100, 105, 110, 120, 130, 140, 130, 100, 80, 70, 60, 55, 50],
            'intersection': [30, 20, 15, 10, 15, 25, 50, 80, 70, 60, 65, 70, 75, 80, 85, 90, 95, 85, 70, 55, 45, 40, 35, 30],
            'bridge': [25, 15, 10, 8, 12, 20, 40, 65, 55, 50, 55, 60, 65, 70, 75, 80, 85, 75, 60, 45, 35, 30, 28, 25],
            'arterial': [35, 25, 20, 15, 20, 30, 60, 90, 80, 70, 75, 80, 85, 90, 95, 100, 105, 95, 80, 65, 55, 50, 45, 35],
            'commercial': [20, 10, 5, 3, 5, 15, 30, 50, 70, 90, 100, 110, 120, 115, 110, 105, 95, 85, 70, 50, 35, 25, 22, 20]
        }
        return patterns.get(sensor_type, patterns['intersection'])[hour]

    def _get_capacity(self, sensor_type):
        capacities = {
            'highway': 2000,
            'intersection': 1000,
            'bridge': 1500,
            'arterial': 1200,
            'commercial': 800
        }
        return capacities.get(sensor_type, 1000)

    def _calculate_speed(self, density, sensor_type):
        max_speeds = {
            'highway': 100,
            'intersection': 50,
            'bridge': 70,
            'arterial': 60,
            'commercial': 40
        }
        max_speed = max_speeds.get(sensor_type, 50)
        return max_speed * (1 - min(density, 1.0))

class WeatherDataSimulator:
    """Simulates weather data that affects traffic"""
    def get_current_data(self):
        conditions = ['clear', 'cloudy', 'rain', 'heavy_rain', 'snow', 'fog']
        return {
            'temperature': random.uniform(10, 35),
            'humidity': random.uniform(30, 95),
            'precipitation': random.uniform(0, 10),
            'wind_speed': random.uniform(0, 20),
            'visibility': random.uniform(1, 10),
            'condition': random.choice(conditions),
            'traffic_impact_score': random.uniform(0.1, 0.9),
            'timestamp': datetime.now().isoformat()
        }

class EventDataSimulator:
    """Simulates city events that affect traffic"""
    def get_current_data(self):
        events = [
            {'name': 'Concert at Stadium', 'expected_attendance': 50000, 'impact_radius': 5},
            {'name': 'University Graduation', 'expected_attendance': 10000, 'impact_radius': 3},
            {'name': 'Street Festival', 'expected_attendance': 25000, 'impact_radius': 2},
            {'name': 'Sports Game', 'expected_attendance': 30000, 'impact_radius': 4},
            {'name': 'Conference', 'expected_attendance': 5000, 'impact_radius': 1}
        ]
        if random.random() < 0.3:
            event = random.choice(events)
            return {
                'active_events': [event],
                'total_expected_impact': event['expected_attendance'] * 0.7,
                'peak_impact_time': f"{random.randint(18, 22)}:00",
                'timestamp': datetime.now().isoformat()
            }
        return {
            'active_events': [],
            'total_expected_impact': 0,
            'peak_impact_time': None,
            'timestamp': datetime.now().isoformat()
        }

class IncidentDataSimulator:
    """Simulates incident reports"""
    def get_current_data(self):
        return {
            'incidents': [],
            'timestamp': datetime.now().isoformat()
        }

class PublicTransportSimulator:
    """Simulates public transport data"""
    def get_current_data(self):
        return {
            'status': 'normal',
            'timestamp': datetime.now().isoformat()
        }

class TrafficDataProcessor:
    def clean_data(self, data):
        return data

    def enrich_data(self, data):
        return data

    def standardize_format(self, data):
        return data

class WeatherDataProcessor:
    def clean_data(self, data):
        return data

    def enrich_data(self, data):
        return data

    def standardize_format(self, data):
        return data

class EventDataProcessor:
    def clean_data(self, data):
        return data

    def enrich_data(self, data):
        return data

    def standardize_format(self, data):
        return data

class IncidentDataProcessor:
    def clean_data(self, data):
        return data

    def enrich_data(self, data):
        return data

    def standardize_format(self, data):
        return data

class TransportDataProcessor:
    def clean_data(self, data):
        return data

    def enrich_data(self, data):
        return data

    def standardize_format(self, data):
        return data

class ProcessedDataStore:
    def __init__(self):
        self.data = []

    def store(self, processed_data):
        self.data.append(processed_data)

    def get_latest(self):
        return self.data[-10:] if self.data else []

class RoadSegment:
    """Represents a road segment with traffic flow data"""
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        self.vehicle_count = 0
        self.average_speed = 0
        self.density = 0
        self.congestion_level = 0

    def update_flow(self, vehicle_count, average_speed, density):
        self.vehicle_count = vehicle_count
        self.average_speed = average_speed
        self.density = density
        self.congestion_level = min(density, 1.0)

class TrafficNetworkGraph:
    """Manages road segments in the traffic network"""
    def __init__(self):
        self.segments = {
            'TS001': RoadSegment('TS001'),
            'TS002': RoadSegment('TS002'),
            'TS003': RoadSegment('TS003'),
            'TS004': RoadSegment('TS004'),
            'TS005': RoadSegment('TS005')
        }

    def get_segment(self, sensor_id):
        return self.segments.get(sensor_id)

class TrafficState:
    def __init__(self):
        self.weather = {}
        self.events = []
        self.incidents = []

    def to_dict(self):
        return {
            'weather': self.weather,
            'events': self.events,
            'incidents': self.incidents
        }

class TrafficSimulationEngine:
    def run_simulation(self, current_state, scenario_params, time_horizon):
        return {}

class HistoricalDataStore:
    def __init__(self):
        self.data = []

class OpenCityModelWrapper:
    """Wrapper for OpenCity pre-trained traffic prediction model"""
    def __init__(self, model_path: str = None, model_size: str = "base"):
        self.model_size = model_size
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        self.config = {
            'mini': {'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4},
            'base': {'hidden_dim': 128, 'num_layers': 4, 'num_heads': 8},
            'plus': {'hidden_dim': 256, 'num_layers': 6, 'num_heads': 16}
        }
        if TORCH_AVAILABLE:
            self.load_model()

    def load_model(self):
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load the actual model weights
                self.model = torch.load(self.model_path, map_location=self.device)
                self.is_loaded = True
                print(f"✓ Loaded OpenCity-{self.model_size} from {self.model_path}")
            else:
                self.model = self._create_mock_model()
                self.is_loaded = True
                print(f"✓ Initialized mock OpenCity-{self.model_size} model")
                print("  To use real model, download from: https://github.com/HKUDS/OpenCity")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = self._create_mock_model()
            self.is_loaded = True

    def _create_mock_model(self):
        config = self.config[self.model_size]
        return MockTrafficModel(
            input_dim=32,
            hidden_dim=config['hidden_dim'],
            output_dim=3,
            num_layers=config['num_layers']
        )

    def predict(self, features: Dict) -> Dict:
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        try:
            input_tensor = self._prepare_features(features)
            with torch.no_grad():
                predictions = self.model(input_tensor)
            return self._postprocess_predictions(predictions, features)
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

    def _prepare_features(self, features: Dict) -> torch.Tensor:
        traffic_features = []
        if 'traffic_data' in features:
            for data in features['traffic_data'].values():
                traffic_features.extend([
                    data.get('vehicle_count', 0) / 1000,
                    data.get('average_speed', 0) / 100,
                    data.get('density', 0),
                    data.get('congestion_level', 0),
                ])
        if 'weather' in features:
            weather = features['weather']
            weather_impact = self._calculate_weather_impact(weather)
            traffic_features.append(weather_impact)
        else:
            traffic_features.append(0.0)
        now = datetime.now()
        time_features = [
            now.hour / 24.0,
            now.weekday() / 7.0,
        ]
        traffic_features.extend(time_features)
        while len(traffic_features) < 32:
            traffic_features.append(0.0)
        input_tensor = torch.tensor(traffic_features[:32], dtype=torch.float32)
        return input_tensor.unsqueeze(0)

    def _calculate_weather_impact(self, weather: Dict) -> float:
        impact = 0.0
        temp = weather.get('temperature', 20)
        if temp < 0 or temp > 35:
            impact += 0.3
        precip = weather.get('precipitation', 0)
        if precip > 0:
            impact += min(precip / 10.0, 0.5)
        visibility = weather.get('visibility', 10)
        if visibility < 5:
            impact += (5 - visibility) / 5.0 * 0.4
        return min(impact, 1.0)

    def _postprocess_predictions(self, predictions: torch.Tensor, features: Dict) -> Dict:
        pred_array = predictions.cpu().numpy().flatten()
        return {
            'short_term_flow': max(0, float(pred_array[0] * 1000)),
            'medium_term_flow': max(0, float(pred_array[1] * 1000)),
            'long_term_flow': max(0, float(pred_array[2] * 1000)),
            'confidence_score': min(1.0, max(0.0, float(np.mean(np.abs(pred_array))))),
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': f'OpenCity-{self.model_size}'
        }

class MockTrafficModel(nn.Module):
    """Mock traffic prediction model for demonstration"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

class EnhancedShortTermPredictionModel:
    """Enhanced short-term prediction using pre-trained models"""
    def __init__(self, model_type: str = "opencity"):
        self.model_type = model_type
        if model_type == "opencity":
            self.predictor = OpenCityModelWrapper(model_size="base")  # Still using mock for short-term
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, features: Dict) -> Dict:
        if not self.predictor.is_loaded:
            return self._fallback_prediction(features)
        return self.predictor.predict(features)

    def _fallback_prediction(self, features: Dict) -> Dict:
        return {
            'short_term_flow': 500,
            'confidence_score': 0.3,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': 'fallback'
        }

class EnhancedMediumTermPredictionModel:
    """Enhanced medium-term prediction"""
    def __init__(self):
        self.predictor = OpenCityModelWrapper(
            model_path="C:\\agentic-ai\\Twin\\Traffic_Agent\\OpenCity-Plus\\OpenCity-plus.pth",
            model_size="plus"
        )

    def predict(self, features: Dict) -> Dict:
        if not self.predictor.is_loaded:
            return {'medium_term_flow': 750, 'confidence_score': 0.4}
        prediction = self.predictor.predict(features)
        return {
            'medium_term_flow': prediction.get('medium_term_flow', 750),
            'confidence_score': prediction.get('confidence_score', 0.4)
        }

class EnhancedLongTermPredictionModel:
    """Enhanced long-term prediction"""
    def __init__(self):
        self.predictor = OpenCityModelWrapper(
            model_path="C:\\agentic-ai\\Twin\\Traffic_Agent\\OpenCity-Plus\\OpenCity-plus.pth",
            model_size="plus"
        )

    def predict(self, features: Dict) -> Dict:
        if not self.predictor.is_loaded:
            return {'long_term_flow': 600, 'confidence_score': 0.3}
        prediction = self.predictor.predict(features)
        return {
            'long_term_flow': prediction.get('long_term_flow', 600),
            'confidence_score': prediction.get('confidence_score', 0.3)
        }

class EnhancedTrafficFeatureEngineer:
    """Enhanced feature engineering for pre-trained models"""
    def create_features(self, traffic_data, current_state, historical_data, external_factors):
        features = {
            'traffic_data': traffic_data,
            'weather': external_factors.get('weather_conditions', {}),
            'temporal': self._create_temporal_features(),
            'events': external_factors.get('scheduled_events', []),
            'historical': self._process_historical_data(historical_data)
        }
        return features

    def _create_temporal_features(self):
        now = datetime.now()
        return {
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'month': now.month,
            'is_weekend': now.weekday() >= 5,
            'is_peak_hour': now.hour in [7, 8, 9, 17, 18, 19],
            'season': (now.month - 1) // 3
        }

    def _process_historical_data(self, historical_data):
        return {
            'avg_flow_last_week': 650,
            'avg_flow_same_time_last_week': 580,
            'trend_direction': 'increasing',
            'seasonal_factor': 1.1
        }

class CongestionAnalyzer:
    """Analyzes congestion based on current state and predictions"""
    def __init__(self, digital_twin):
        self.digital_twin = digital_twin

    def analyze(self, current_state, predictions):
        max_congestion = 0.0
        hotspots = []
        for sensor_id, data in self.digital_twin.get_current_traffic_data().items():
            if data['congestion_level'] > max_congestion:
                max_congestion = data['congestion_level']
            if data['congestion_level'] > 0.8:
                hotspots.append(sensor_id)
        return {
            'max_congestion_level': max_congestion,
            'congestion_hotspots': hotspots,
            'estimated_delays': {sensor_id: max_congestion * 10 for sensor_id in hotspots},
            'incident_impact_score': 0.3,
            'incident_affected_areas': [],
            'incident_duration': 0
        }

class TrafficOptimizationEngine:
    pass

class TrafficSignalController:
    def adjust_timing(self, intersection_id, new_timing):
        return True

class RouteAdvisor:
    pass

class NotificationSystem:
    pass

class AgentMemory:
    def store_experience(self, experience):
        pass

class AgentLearningModule:
    def update_models(self, experience):
        pass

class DataIngestionModule:
    def __init__(self):
        self.data_sources = {
            'traffic_sensors': TrafficSensorSimulator(),
            'weather_api': WeatherDataSimulator(),
            'events_calendar': EventDataSimulator(),
            'incident_reports': IncidentDataSimulator(),
            'public_transport': PublicTransportSimulator()
        }
        self.message_queue = LocalMessageQueue()

    def start_data_streams(self):
        for source_name, source in self.data_sources.items():
            thread = threading.Thread(target=self._collect_data_loop, args=(source_name, source))
            thread.daemon = True
            thread.start()

    def _collect_data_loop(self, source_name, source):
        while True:
            data = source.get_current_data()
            message = {
                'timestamp': datetime.now(),
                'source': source_name,
                'data': data
            }
            self.message_queue.publish(message)
            time.sleep(30)

class DataProcessingModule:
    def __init__(self, message_queue):
        self.message_queue = message_queue
        self.processors = {
            'traffic_sensors': TrafficDataProcessor(),
            'weather_api': WeatherDataProcessor(),
            'events_calendar': EventDataProcessor(),
            'incident_reports': IncidentDataProcessor(),
            'public_transport': TransportDataProcessor()
        }
        self.processed_data_store = ProcessedDataStore()

    def start_processing(self):
        thread = threading.Thread(target=self._process_data_loop)
        thread.daemon = True
        thread.start()

    def _process_data_loop(self):
        while True:
            message = self.message_queue.consume()
            if message:
                processed_data = self._process_message(message)
                self.processed_data_store.store(processed_data)
            time.sleep(0.1)

    def _process_message(self, message):
        source = message['source']
        processor = self.processors[source]
        cleaned_data = processor.clean_data(message['data'])
        enriched_data = processor.enrich_data(cleaned_data)
        standardized_data = processor.standardize_format(enriched_data)
        return {
            'timestamp': message['timestamp'],
            'source': source,
            'processed_data': standardized_data
        }

class TrafficDigitalTwin:
    def __init__(self, firebase_manager: FirebaseManager = None):
        self.firebase = firebase_manager
        self.traffic_network = TrafficNetworkGraph()
        self.current_state = TrafficState()
        self.historical_data = HistoricalDataStore()
        self.simulation_engine = TrafficSimulationEngine()

    def update_state(self, processed_data):
        data_type = processed_data['source']
        if data_type == 'traffic_sensors':
            self._update_traffic_flow(processed_data)
        elif data_type == 'weather_api':
            self._update_weather_conditions(processed_data)
        elif data_type == 'events_calendar':
            self._update_events_impact(processed_data)
        elif data_type == 'incident_reports':
            self._update_incidents(processed_data)
        elif data_type == 'public_transport':
            self._update_transport_status(processed_data)

    def _update_traffic_flow(self, data):
        for sensor_id, flow_data in data['processed_data'].items():
            road_segment = self.traffic_network.get_segment(sensor_id)
            road_segment.update_flow(
                vehicle_count=flow_data['vehicle_count'],
                average_speed=flow_data['average_speed'],
                density=flow_data['density']
            )

    def _update_weather_conditions(self, data):
        self.current_state.weather = data['processed_data']

    def _update_events_impact(self, data):
        self.current_state.events = data['processed_data']['active_events']

    def _update_incidents(self, data):
        self.current_state.incidents = data['processed_data']['incidents']

    def _update_transport_status(self, data):
        pass

    def simulate_scenarios_notification(self, scenario_params):
        return self.simulation_engine.run_simulation(
            current_state=self.current_state,
            scenario_params=scenario_params,
            time_horizon=scenario_params.get('time_horizon', 60)
        )

    def get_current_state(self):
        return self.current_state.to_dict()

    def get_current_traffic_data(self):
        traffic_data = {}
        for sensor_id, segment in self.traffic_network.segments.items():
            traffic_data[sensor_id] = {
                'vehicle_count': segment.vehicle_count,
                'average_speed': segment.average_speed,
                'density': segment.density,
                'congestion_level': segment.congestion_level
            }
        return traffic_data

    def update_signal_state(self, intersection_id, timing):
        pass

class TrafficPredictionModule:
    """Enhanced traffic prediction module with pre-trained models"""
    def __init__(self, digital_twin):
        self.digital_twin = digital_twin
        self.models = {
            'short_term': EnhancedShortTermPredictionModel(model_type="opencity"),
            'medium_term': EnhancedMediumTermPredictionModel(),
            'long_term': EnhancedLongTermPredictionModel()
        }
        self.feature_engineer = EnhancedTrafficFeatureEngineer()

    def predict_traffic_flow(self, prediction_horizon='short_term'):
        current_state = self.digital_twin.get_current_state()
        traffic_data = self.digital_twin.get_current_traffic_data()
        external_factors = self._get_external_factors()
        features = self.feature_engineer.create_features(
            traffic_data=traffic_data,
            current_state=current_state,
            historical_data=self.digital_twin.historical_data,
            external_factors=external_factors
        )
        model = self.models[prediction_horizon]
        prediction = model.predict(features)
        confidence = prediction.get('confidence_score', 0.5)
        return {
            'prediction_horizon': prediction_horizon,
            'predicted_flows': prediction,
            'confidence_score': confidence,
            'timestamp': datetime.now()
        }

    def _get_external_factors(self):
        return {
            'weather_conditions': self.digital_twin.current_state.weather,
            'scheduled_events': self.digital_twin.current_state.events,
            'day_of_week': datetime.now().weekday(),
            'time_of_day': datetime.now().hour,
            'school_holidays': False,
            'public_holidays': False
        }

class TrafficReasoningModule:
    def __init__(self, digital_twin, prediction_module):
        self.digital_twin = digital_twin
        self.prediction_module = prediction_module
        self.congestion_analyzer = CongestionAnalyzer(digital_twin)  # Pass digital_twin
        self.optimization_engine = TrafficOptimizationEngine()

    def analyze_traffic_situation(self):
        current_state = self.digital_twin.get_current_state()
        predictions = {}
        for horizon in ['short_term', 'medium_term', 'long_term']:
            predictions[horizon] = self.prediction_module.predict_traffic_flow(horizon)
        congestion_analysis = self.congestion_analyzer.analyze(
            current_state=current_state,
            predictions=predictions
        )
        critical_issues = self._identify_critical_issues(congestion_analysis)
        # Enhanced logging to show predictions
        print("Predictions for this cycle:")
        for horizon, pred in predictions.items():
            print(f"  {horizon}: {pred['predicted_flows']}")
        return {
            'current_analysis': congestion_analysis,
            'predictions': predictions,
            'critical_issues': critical_issues,
            'recommended_actions': self._generate_recommendations(critical_issues)
        }

    def _identify_critical_issues(self, analysis):
        issues = []
        if analysis['max_congestion_level'] > 0.8:
            issues.append({
                'type': 'severe_congestion',
                'location': analysis['congestion_hotspots'],
                'severity': analysis['max_congestion_level'],
                'estimated_delay': analysis['estimated_delays']
            })
        if analysis['incident_impact_score'] > 0.6:
            issues.append({
                'type': 'incident_impact',
                'affected_areas': analysis['incident_affected_areas'],
                'estimated_duration': analysis['incident_duration']
            })
        return issues

    def _generate_recommendations(self, issues):
        recommendations = []
        for issue in issues:
            if issue['type'] == 'severe_congestion':
                recommendations.append({
                    'action_type': 'adjust_signal_timing',
                    'signal_adjustments': {loc: {'green_time': 30} for loc in issue['location']}
                })
            elif issue['type'] == 'incident_impact':
                recommendations.append({
                    'action_type': 'suggest_alternate_routes',
                    'affected_areas': issue['affected_areas']
                })
        return recommendations

class TrafficActionModule:
    def __init__(self, digital_twin, firebase_manager: FirebaseManager = None):
        self.digital_twin = digital_twin
        self.firebase = firebase_manager
        self.traffic_signal_controller = TrafficSignalController()
        self.route_advisor = RouteAdvisor()
        self.notification_system = NotificationSystem()

    def execute_recommendations(self, recommendations):
        execution_results = []
        for recommendation in recommendations:
            try:
                result = self._execute_action(recommendation)
                execution_results.append({
                    'action': recommendation,
                    'result': result,
                    'status': 'success',
                    'timestamp': datetime.now()
                })
            except Exception as e:
                execution_results.append({
                    'action': recommendation,
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.now()
                })
        return execution_results

    def _execute_action(self, recommendation):
        action_type = recommendation['action_type']
        if action_type == 'adjust_signal_timing':
            return self._adjust_traffic_signals(recommendation)
        elif action_type == 'suggest_alternate_routes':
            return self._suggest_alternate_routes(recommendation)
        elif action_type == 'notify_authorities':
            return self._notify_authorities(recommendation)
        elif action_type == 'update_variable_message_signs':
            return self._update_message_signs(recommendation)

    def _adjust_traffic_signals(self, recommendation):
        signal_adjustments = recommendation['signal_adjustments']
        for intersection_id, timing in signal_adjustments.items():
            result = self.traffic_signal_controller.adjust_timing(
                intersection_id=intersection_id,
                new_timing=timing
            )
            self.digital_twin.update_signal_state(intersection_id, timing)
        return "Traffic signals adjusted successfully"

    def _suggest_alternate_routes(self, recommendation):
        return "Alternate routes suggested"

    def _notify_authorities(self, recommendation):
        return "Authorities notified"

    def _update_message_signs(self, recommendation):
        return "Message signs updated"

class TrafficAgent:
    def __init__(self):
                # Initialize Firebase
        service_account_path = "C:\\agentic-ai\\Twin\\Traffic_Agent\\google-service.json"  # Update this path
        self.firebase = FirebaseManager(service_account_path)
        self.data_ingestion = DataIngestionModule()
        self.data_processing = DataProcessingModule(self.data_ingestion.message_queue)
        self.digital_twin = TrafficDigitalTwin()
        self.prediction_module = TrafficPredictionModule(self.digital_twin)
        self.reasoning_module = TrafficReasoningModule(self.digital_twin, self.prediction_module)
        self.action_module = TrafficActionModule(self.digital_twin)
        self.agent_memory = AgentMemory()
        self.learning_module = AgentLearningModule()

    def start(self):
        print("Starting Traffic Agent with Enhanced Prediction...")
        self.data_ingestion.start_data_streams()
        self.data_processing.start_processing()
        self.run_agent_loop()

    def run_agent_loop(self):
        while True:
            try:
                latest_data = self.data_processing.processed_data_store.get_latest()
                for data_point in latest_data:
                    self.digital_twin.update_state(data_point)
                situation_analysis = self.reasoning_module.analyze_traffic_situation()
                action_plan = self._create_action_plan(situation_analysis)
                execution_results = self.action_module.execute_recommendations(
                    action_plan['recommendations']
                )
                self._learn_from_results(situation_analysis, action_plan, execution_results)
                self._log_agent_cycle(situation_analysis, action_plan, execution_results)
            except Exception as e:
                print(f"Error in agent loop: {e}")
            time.sleep(120)

    def _create_action_plan(self, situation_analysis):
        recommendations = situation_analysis['recommended_actions']
        prioritized_recommendations = self._prioritize_recommendations(recommendations)
        execution_timeline = self._create_execution_timeline(prioritized_recommendations)
        return {
            'recommendations': prioritized_recommendations,
            'execution_timeline': execution_timeline,
            'expected_outcomes': self._predict_outcomes(prioritized_recommendations)
        }

    def _prioritize_recommendations(self, recommendations):
        return recommendations

    def _create_execution_timeline(self, recommendations):
        return []

    def _predict_outcomes(self, recommendations):
        return {}

    def _learn_from_results(self, analysis, plan, results):
        experience = {
            'situation': analysis,
            'action_plan': plan,
            'results': results,
            'timestamp': datetime.now()
        }
        self.agent_memory.store_experience(experience)
        self.learning_module.update_models(experience)

    def _log_agent_cycle(self, situation_analysis, action_plan, execution_results):
        print("Agent Cycle Log:")
        print("Current Analysis:", situation_analysis['current_analysis'])
        print("Critical Issues:", situation_analysis['critical_issues'])
        print("Recommended Actions:", action_plan['recommendations'])
        print("Execution Results:", execution_results)
        print("-" * 50)

def main():
    agent = TrafficAgent()
    agent.start()

if __name__ == "__main__":
    main()