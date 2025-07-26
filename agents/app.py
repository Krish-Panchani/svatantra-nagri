from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import logging
import requests
from pathlib import Path

# Import the agent modules
import sys
sys.path.append('.')
from envagent import EnvironmentalAgent, air_quality_data, energy_sources
from infraagent import InfrastructureHealthAgent
from main import TrafficAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize Socket.IO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentDataCollector:
    """Collects and manages data from all agents"""
    
    def __init__(self):
        self.environmental_data = {}
        self.infrastructure_data = {}
        self.traffic_data = {}
        self.joint_alerts = []
        self.data_lock = threading.Lock()
        
        # Initialize agents
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all agent instances"""
        try:
            # Environmental Agent
            self.env_agent = EnvironmentalAgent(air_quality_data, energy_sources)
            
            # Infrastructure Health Agent
            self.infra_agent = InfrastructureHealthAgent("MAIN_BRIDGE_001", {
                'cycle_interval': 30,
                'log_level': 'INFO'
            })
            
            # Traffic Agent
            self.traffic_agent = TrafficAgent()
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
    
    def start_agent_monitoring(self):
        """Start monitoring threads for all agents"""
        # Start Environmental Agent monitoring
        env_thread = threading.Thread(target=self.monitor_environmental_agent, daemon=True)
        env_thread.start()
        
        # Start Infrastructure Agent monitoring
        infra_thread = threading.Thread(target=self.monitor_infrastructure_agent, daemon=True)
        infra_thread.start()
        
        # Start Traffic Agent monitoring
        traffic_thread = threading.Thread(target=self.monitor_traffic_agent, daemon=True)
        traffic_thread.start()
        
        logger.info("All agent monitoring threads started")
    
    def monitor_environmental_agent(self):
        """Monitor environmental agent data"""
        while True:
            try:
                # Update air quality data
                self.env_agent.update_air_quality()
                hotspots = self.env_agent.identify_hotspots()
                
                # Get energy optimization
                allocation = self.env_agent.optimize_energy_grid(250)
                
                # Get predictions
                predictions = {}
                for district in [data.district for data in self.env_agent.air_quality]:
                    predictions[district] = self.env_agent.predict_pm25(district, 1)
                
                with self.data_lock:
                    self.environmental_data = {
                        'air_quality': [
                            {
                                'district': data.district,
                                'pm25': data.pm25,
                                'no2': data.no2,
                                'co': data.co,
                                'notes': data.notes
                            } for data in self.env_agent.air_quality
                        ],
                        'hotspots': hotspots,
                        'energy_allocation': allocation,
                        'predictions': predictions,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Emit Socket.IO event for air quality
                socketio.emit('air_quality_update', self.environmental_data['air_quality'])
                
                # Check for critical conditions
                self.check_environmental_alerts()
                
            except Exception as e:
                logger.error(f"Error monitoring environmental agent: {e}")
            
            socketio.sleep(30)  # Use socketio.sleep for async compatibility
    
    def monitor_infrastructure_agent(self):
        """Monitor infrastructure health agent data"""
        while True:
            try:
                # Get latest processed data
                latest_data = self.infra_agent.data_processing.get_latest_processed_data(10)
                
                # Get digital twin state
                health_state = self.infra_agent.digital_twin.current_health_state
                
                with self.data_lock:
                    self.infrastructure_data = {
                        'bridge_health': {
                            'concerns': [
                                {
                                    'id': concern.get('id', 'unknown'),
                                    'type': concern.get('type', 'unknown'),
                                    'location': concern.get('location', 'unknown'),
                                    'severity': concern.get('severity', 'unknown'),
                                    'detected_at': concern.get('detected_at', datetime.now()).isoformat()
                                } for concern in health_state.concerns
                            ],
                            'update_count': self.infra_agent.digital_twin.update_count,
                            'last_update': self.infra_agent.digital_twin.last_update.isoformat() if self.infra_agent.digital_twin.last_update else None
                        },
                        'sensor_data': [
                            {
                                'source': data.get('source', 'unknown'),
                                'timestamp': data.get('timestamp', datetime.now()).isoformat(),
                                'anomaly_score': data.get('anomaly_score', 0),
                                'alert_level': data.get('alert_level', 'normal')
                            } for data in latest_data[-5:]  # Last 5 readings
                        ],
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Emit Socket.IO event for bridge health
                socketio.emit('bridge_health_update', {
                    'overall_status': 'poor' if any(c['severity'] == 'critical' for c in self.infrastructure_data['bridge_health']['concerns']) else
                                      'fair' if any(c['severity'] == 'moderate' for c in self.infrastructure_data['bridge_health']['concerns']) else 'good',
                    'concern_count': len(self.infrastructure_data['bridge_health']['concerns']),
                    'critical_concerns': len([c for c in self.infrastructure_data['bridge_health']['concerns'] if c['severity'] == 'critical'])
                })
                
                # Check for infrastructure alerts
                self.check_infrastructure_alerts()
                
            except Exception as e:
                logger.error(f"Error monitoring infrastructure agent: {e}")
            
            socketio.sleep(30)  # Use socketio.sleep for async compatibility
    
    def monitor_traffic_agent(self):
        """Monitor traffic agent data"""
        while True:
            try:
                # Get current traffic state
                traffic_state = self.traffic_agent.get_current_state_data()
                
                # Get analysis from reasoning module
                situation_analysis = self.traffic_agent.reasoning_module.analyze_traffic_situation()
                
                with self.data_lock:
                    self.traffic_data = {
                        'traffic_flow': traffic_state['traffic_data'],
                        'current_state': traffic_state['current_state'],
                        'congestion_analysis': situation_analysis['current_analysis'],
                        'critical_issues': situation_analysis['critical_issues'],
                        'predictions': situation_analysis['predictions'],
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Emit Socket.IO event for traffic data
                socketio.emit('traffic_update', self.traffic_data['traffic_flow'])
                
                # Check for traffic alerts
                self.check_traffic_alerts(situation_analysis)
                
            except Exception as e:
                logger.error(f"Error monitoring traffic agent: {e}")
            
            socketio.sleep(30)  # Use socketio.sleep for async compatibility
    
    def check_environmental_alerts(self):
        """Check for environmental critical conditions"""
        if self.environmental_data.get('hotspots'):
            alert = {
                'id': f"env_alert_{int(time.time())}",
                'type': 'environmental',
                'severity': 'high',
                'message': f"Pollution hotspots detected: {', '.join(self.environmental_data['hotspots'])}",
                'affected_areas': self.environmental_data['hotspots'],
                'timestamp': datetime.now().isoformat(),
                'actions_needed': ['Increase monitoring', 'Activate air filtration', 'Issue health advisory']
            }
            self.add_joint_alert(alert)
    
    def check_infrastructure_alerts(self):
        """Check for infrastructure critical conditions"""
        concerns = self.infrastructure_data.get('bridge_health', {}).get('concerns', [])
        critical_concerns = [c for c in concerns if c.get('severity') == 'critical']
        
        if critical_concerns:
            alert = {
                'id': f"infra_alert_{int(time.time())}",
                'type': 'infrastructure',
                'severity': 'critical',
                'message': f"Critical infrastructure issues detected: {len(critical_concerns)} concerns",
                'affected_areas': [c.get('location', 'unknown') for c in critical_concerns],
                'timestamp': datetime.now().isoformat(),
                'actions_needed': ['Emergency inspection', 'Traffic restriction', 'Immediate maintenance']
            }
            self.add_joint_alert(alert)
    
    def check_traffic_alerts(self, situation_analysis):
        """Check for traffic critical conditions"""
        critical_issues = situation_analysis.get('critical_issues', [])
        
        for issue in critical_issues:
            if issue['type'] == 'severe_congestion':
                alert = {
                    'id': f"traffic_alert_{int(time.time())}",
                    'type': 'traffic',
                    'severity': 'high',
                    'message': f"Severe congestion detected at {issue['location']}",
                    'affected_areas': [issue['location']],
                    'timestamp': datetime.now().isoformat(),
                    'actions_needed': ['Adjust signal timing', 'Deploy traffic officers', 'Activate alternate routes']
                }
                self.add_joint_alert(alert)
    
    def add_joint_alert(self, alert):
        """Add a joint alert to the system"""
        with self.data_lock:
            # Remove old alerts of the same type and location
            self.joint_alerts = [
                a for a in self.joint_alerts 
                if not (a.get('type') == alert.get('type') and 
                       set(a.get('affected_areas', [])) == set(alert.get('affected_areas', [])))
            ]
            
            self.joint_alerts.append(alert)
            
            # Keep only last 20 alerts
            if len(self.joint_alerts) > 20:
                self.joint_alerts = self.joint_alerts[-20:]
            
            # Emit Socket.IO event for joint alert
            socketio.emit('joint_alert', alert)

# Initialize global data collector
data_collector = AgentDataCollector()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/3d-city')
def city_view():
    """3D city visualization page"""
    return render_template('3d_view.html')

@app.route('/api/environmental-data')
def get_environmental_data():
    """API endpoint for environmental data"""
    with data_collector.data_lock:
        return jsonify(data_collector.environmental_data)

@app.route('/api/infrastructure-data')
def get_infrastructure_data():
    """API endpoint for infrastructure data"""
    with data_collector.data_lock:
        return jsonify(data_collector.infrastructure_data)

@app.route('/api/traffic-data', methods=['GET', 'POST'])
def handle_traffic_data():
    """API endpoint for traffic data"""
    if request.method == 'POST':
        # Handle traffic data updates from external sources
        try:
            traffic_update = request.get_json()
            logger.info(f"Received traffic update: {traffic_update}")
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error handling traffic update: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    else:
        with data_collector.data_lock:
            return jsonify(data_collector.traffic_data)

@app.route('/api/joint-alerts', methods=['GET', 'POST'])
def handle_joint_alerts():
    """API endpoint for joint alerts"""
    if request.method == 'POST':
        try:
            alert = request.get_json()
            data_collector.add_joint_alert(alert)
            logger.info(f"Received joint alert: {alert}")
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error handling joint alert: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
    else:
        with data_collector.data_lock:
            return jsonify(data_collector.joint_alerts)

@app.route('/api/system-status')
def get_system_status():
    """API endpoint for overall system status"""
    with data_collector.data_lock:
        status = {
            'agents': {
                'environmental': {
                    'status': 'active' if data_collector.environmental_data else 'inactive',
                    'last_update': data_collector.environmental_data.get('timestamp'),
                    'hotspots_count': len(data_collector.environmental_data.get('hotspots', []))
                },
                'infrastructure': {
                    'status': 'active' if data_collector.infrastructure_data else 'inactive',
                    'last_update': data_collector.infrastructure_data.get('timestamp'),
                    'concerns_count': len(data_collector.infrastructure_data.get('bridge_health', {}).get('concerns', []))
                },
                'traffic': {
                    'status': 'active' if data_collector.traffic_data else 'inactive',
                    'last_update': data_collector.traffic_data.get('timestamp'),
                    'issues_count': len(data_collector.traffic_data.get('critical_issues', []))
                }
            },
            'alerts': {
                'total_count': len(data_collector.joint_alerts),
                'critical_count': len([a for a in data_collector.joint_alerts if a.get('severity') == 'critical']),
                'recent_alerts': data_collector.joint_alerts[-5:]  # Last 5 alerts
            },
            'timestamp': datetime.now().isoformat()
        }
    
    return jsonify(status)

@app.route('/api/city-data')
def get_city_data():
    """Combined city data for 3D visualization"""
    with data_collector.data_lock:
        city_data = {
            'environmental': {
                'air_quality_zones': [],
                'energy_sources': []
            },
            'infrastructure': {
                'bridges': [],
                'health_indicators': []
            },
            'traffic': {
                'flow_data': data_collector.traffic_data.get('traffic_flow', {}),
                'congestion_points': []
            },
            'alerts': data_collector.joint_alerts[-10:],  # Last 10 alerts
            'timestamp': datetime.now().isoformat()
        }
        
        # Process environmental data for 3D visualization
        if data_collector.environmental_data:
            for district_data in data_collector.environmental_data.get('air_quality', []):
                city_data['environmental']['air_quality_zones'].append({
                    'name': district_data['district'],
                    'pm25': district_data['pm25'],
                    'no2': district_data['no2'],
                    'co': district_data['co'],
                    'status': 'critical' if district_data['district'] in data_collector.environmental_data.get('hotspots', []) else 'normal'
                })
        
        # Process infrastructure data for 3D visualization
        if data_collector.infrastructure_data:
            bridge_concerns = data_collector.infrastructure_data.get('bridge_health', {}).get('concerns', [])
            city_data['infrastructure']['bridges'].append({
                'id': 'MAIN_BRIDGE_001',
                'status': 'critical' if any(c.get('severity') == 'critical' for c in bridge_concerns) else 'normal',
                'concerns_count': len(bridge_concerns)
            })
        
        # Process traffic data for 3D visualization
        if data_collector.traffic_data:
            for sensor_id, flow_data in data_collector.traffic_data.get('traffic_flow', {}).items():
                if flow_data.get('congestion_level', 0) > 0.8:
                    city_data['traffic']['congestion_points'].append({
                        'sensor_id': sensor_id,
                        'congestion_level': flow_data['congestion_level'],
                        'vehicle_count': flow_data['vehicle_count']
                    })
    
    return jsonify(city_data)

@app.route('/api/control/environmental/<action>')
def control_environmental(action):
    """Control environmental agent actions"""
    try:
        if action == 'optimize_energy':
            allocation = data_collector.env_agent.optimize_energy_grid(250)
            return jsonify({'status': 'success', 'allocation': allocation})
        elif action == 'identify_hotspots':
            hotspots = data_collector.env_agent.identify_hotspots()
            return jsonify({'status': 'success', 'hotspots': hotspots})
        else:
            return jsonify({'status': 'error', 'message': 'Unknown action'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/traffic/<action>')
def control_traffic(action):
    """Control traffic agent actions"""
    try:
        if action == 'analyze_situation':
            analysis = data_collector.traffic_agent.reasoning_module.analyze_traffic_situation()
            return jsonify({'status': 'success', 'analysis': analysis})
        else:
            return jsonify({'status': 'error', 'message': 'Unknown action'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def start_agent_monitoring():
    """Start all agent monitoring in background"""
    data_collector.start_agent_monitoring()

if __name__ == '__main__':
    print("Starting Smart City Dashboard...")
    print("Initializing agents...")
    
    # Start agent monitoring in background
    monitor_thread = threading.Thread(target=start_agent_monitoring, daemon=True)
    monitor_thread.start()
    
    print("Dashboard starting on http://localhost:8080")
    print("3D City View available at http://localhost:8080/3d-city")
    
    # Start Flask app with Socket.IO
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)