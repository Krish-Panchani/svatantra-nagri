import datetime
import threading
import time
import random
from zoneinfo import ZoneInfo
from typing import Dict, Optional

# Assuming google.adk.agents is available in your environment
from google.adk.agents import Agent

# --- Replicating necessary classes from your main.py for standalone agent ---
class LocalMessageQueue:
    """Simulates a message queue for local development"""
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def publish(self, message):
        with self.lock:
            self.queue.append(message)
            # In a real system, this would push to a durable queue or message bus
            # print(f"Message published to queue: {message}") # For debugging
            
    def consume(self, timeout=1): # Added timeout for non-blocking consumption
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None

class TrafficSensorSimulator:
    """Simulates traffic sensor data with congestion scenario"""
    def __init__(self):
        self.sensor_locations = [
            {'id': 'TS001', 'location': 'Main St & 1st Ave', 'type': 'intersection'},
            {'id': 'TS002', 'location': 'Highway 101 Mile 15', 'type': 'highway'},
            {'id': 'TS003', 'location': 'Downtown Bridge', 'type': 'bridge'},
            {'id': 'TS004', 'location': 'University Campus Gate', 'type': 'arterial'},
            {'id': 'TS005', 'location': 'Shopping Mall Entrance', 'type': 'commercial'}
        ]

    def get_current_data(self):
        data = {}
        current_hour = datetime.datetime.now().hour
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
                'timestamp': datetime.datetime.now().isoformat()
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

class RoadSegment:
    """Represents a road segment with traffic flow data"""
    def __init__(self, sensor_id, location):
        self.sensor_id = sensor_id
        self.location = location # Added location
        self.vehicle_count = 0
        self.average_speed = 0
        self.density = 0
        self.congestion_level = 0

    def update_flow(self, vehicle_count, average_speed, density, congestion_level):
        self.vehicle_count = vehicle_count
        self.average_speed = average_speed
        self.density = density
        self.congestion_level = congestion_level # Directly use provided congestion_level

class TrafficNetworkGraph:
    """Manages road segments in the traffic network"""
    def __init__(self, sensor_locations):
        self.segments = {}
        for sensor in sensor_locations:
            self.segments[sensor['id']] = RoadSegment(sensor['id'], sensor['location'])

    def get_segment(self, sensor_id):
        return self.segments.get(sensor_id)

class TrafficDigitalTwin:
    def __init__(self, traffic_sensor_simulator):
        self.traffic_network = TrafficNetworkGraph(traffic_sensor_simulator.sensor_locations)
        self._current_traffic_data = {}
        self.traffic_sensor_simulator = traffic_sensor_simulator # Reference to simulator

    def update_traffic_data(self, traffic_data):
        """Updates the digital twin's traffic data."""
        self._current_traffic_data = traffic_data
        for sensor_id, flow_data in traffic_data.items():
            road_segment = self.traffic_network.get_segment(sensor_id)
            if road_segment:
                road_segment.update_flow(
                    vehicle_count=flow_data['vehicle_count'],
                    average_speed=flow_data['average_speed'],
                    density=flow_data['density'],
                    congestion_level=flow_data['congestion_level'] # Pass congestion level
                )

    def get_current_traffic_data(self):
        """Returns the current traffic data from the digital twin."""
        traffic_data = {}
        for sensor_id, segment in self.traffic_network.segments.items():
            traffic_data[sensor_id] = {
                'location': segment.location, # Include location
                'vehicle_count': segment.vehicle_count,
                'average_speed': segment.average_speed,
                'density': segment.density,
                'congestion_level': segment.congestion_level
            }
        return traffic_data

    def update_signal_state(self, intersection_id, timing):
        """Simulates updating a traffic signal state."""
        print(f"Digital Twin: Signal at {intersection_id} updated to {timing}")
        # In a real system, this would interact with actual signal hardware/software
        return {"status": "success", "message": f"Signal at {intersection_id} updated to {timing}"}


class TrafficSignalController:
    """Simulates a traffic signal controller."""
    def adjust_timing(self, intersection_id: str, new_timing: dict) -> dict:
        """
        Adjusts the timing of a traffic signal at a specific intersection.

        Args:
            intersection_id (str): The ID of the intersection to adjust.
            new_timing (dict): A dictionary specifying the new timing parameters,
                               e.g., {'green_time': 30, 'red_time': 20}.

        Returns:
            dict: Status of the adjustment.
        """
        print(f"Traffic Signal Controller: Adjusting signal at {intersection_id} to {new_timing}")
        # In a real system, this would send commands to the traffic signal.
        return {"status": "success", "message": f"Signal {intersection_id} adjusted to {new_timing}"}

# --- Agent Tools ---

traffic_sensor_simulator = TrafficSensorSimulator()
digital_twin = TrafficDigitalTwin(traffic_sensor_simulator)
traffic_signal_controller = TrafficSignalController()

# Global variable to store alerts for demonstration
traffic_alerts_for_civilians = LocalMessageQueue()

def get_traffic_density(sensor_id: str = "") -> dict:
    """
    Retrieves the current traffic density and other metrics for all or a specific sensor.

    Args:
        sensor_id (str, optional): The ID of a specific traffic sensor (e.g., 'TS001').
                                   If an empty string, data for all sensors is returned.

    Returns:
        dict: A dictionary containing traffic data for the requested sensor(s).
              Includes 'status' and 'data' or 'error_message'.
    """
    try:
        current_sensor_data = traffic_sensor_simulator.get_current_data()
        digital_twin.update_traffic_data(current_sensor_data) # Update digital twin with latest data

        if sensor_id:
            if sensor_id in current_sensor_data:
                return {"status": "success", "data": {sensor_id: current_sensor_data[sensor_id]}}
            else:
                return {"status": "error", "error_message": f"Sensor '{sensor_id}' not found."}
        else:
            return {"status": "success", "data": current_sensor_data}
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to get traffic density: {str(e)}"}

def set_traffic_signal_timing(intersection_id: str, green_time: int, red_time: int) -> dict:
    """
    Sets the green and red light timing for a traffic signal at a specified intersection.

    Args:
        intersection_id (str): The ID of the intersection where the signal is located (e.g., 'TS001').
                               This should correspond to an intersection in the simulated network.
        green_time (int): The duration in seconds for the green light.
        red_time (int): The duration in seconds for the red light.

    Returns:
        dict: Status of the signal timing adjustment.
    """
    if not isinstance(green_time, int) or green_time <= 0:
        return {"status": "error", "error_message": "Green time must be a positive integer."}
    if not isinstance(red_time, int) or red_time <= 0:
        return {"status": "error", "error_message": "Red time must be a positive integer."}

    new_timing = {'green_time': green_time, 'red_time': red_time}
    try:
        controller_result = traffic_signal_controller.adjust_timing(intersection_id, new_timing)
        if controller_result["status"] == "success":
            digital_twin_result = digital_twin.update_signal_state(intersection_id, new_timing)
            return {"status": "success", "message": f"Traffic signal at {intersection_id} timing set to Green: {green_time}s, Red: {red_time}s. Digital Twin updated."}
        else:
            return controller_result
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to set traffic signal timing: {str(e)}"}

def send_traffic_alert(location: str, message: str, severity: str = "info") -> dict:
    """
    Sends a traffic alert message to civilians for a specific location.

    This simulates sending a broadcast message to a public channel or a notification service.
    In a real-world scenario, this would integrate with a platform like Firebase Cloud Messaging,
    a public display system, or a city's official communication app.

    Args:
        location (str): The location to which the alert pertains (e.g., "Main St & 1st Ave").
        message (str): The traffic update message for civilians (e.g., "Heavy congestion, expect delays").
        severity (str, optional): The severity of the alert (e.g., "info", "warning", "critical").
                                  Defaults to "info".

    Returns:
        dict: Status of the alert sending operation.
    """
    alert_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "location": location,
        "message": message,
        "severity": severity
    }
    traffic_alerts_for_civilians.publish(alert_info)
    print(f"\n📢 SVATANTRA NAGRI TRAFFIC ALERT for Civilians ({severity.upper()}) at {location}: {message}\n")
    return {"status": "success", "message": "Traffic alert sent successfully."}

# --- Agent Definition ---

svatantra_nagri_traffic_agent = Agent(
    name="svatantra_nagri_traffic_agent",
    model="gemini-2.0-flash",
    description=(
        "An intelligent agent for Svatantra Nagri to manage traffic signals and provide "
        "real-time traffic updates to civilians based on dynamic conditions."
    ),
    instruction=(
        "You are the 'Svatantra Nagri Traffic Agent'. Your core responsibility is to ensure "
        "smooth traffic flow and keep the citizens informed about real-time traffic conditions. "
        "You operate continuously to monitor traffic, adjust signal timings as needed, and "
        "proactively send traffic alerts to civilians when significant changes or congestion "
        "occur. "
        "\n\n**Your workflow should be:**"
        "\n1. **Continuously Monitor:** Regularly use `get_traffic_density` to obtain the latest traffic data for all sensors."
        "\n2. **Analyze and Decide:** Based on the `congestion_level` from the traffic data:"
        "\n   - If a `congestion_level` for any sensor exceeds `0.8` (high congestion), consider adjusting the traffic signal timing for that intersection to alleviate it. For congested intersections, prioritize increasing `green_time` and decreasing `red_time` to facilitate flow. For example, if TS001 is congested, you might call `set_traffic_signal_timing(intersection_id='TS001', green_time=45, red_time=15)`."
        "\n   - If a `congestion_level` is between `0.5` and `0.8` (moderate congestion), you might maintain standard timings or slightly adjust them. If the previous state was low congestion and it moved to moderate, consider sending an 'info' alert."
        "\n   - If a `congestion_level` is below `0.5` (low congestion), maintain standard timings, but be ready to react."
        "\n3. **Inform Civilians:** If you detect high congestion (above 0.8) or significant changes in traffic flow (e.g., a sudden increase in congestion by more than 0.3 from the previous measurement, or a persistent high congestion for several monitoring cycles), use the `send_traffic_alert` tool to broadcast messages to the citizens. Clearly state the location, the nature of the issue (e.g., 'heavy congestion', 'slow-moving traffic'), and advise on potential delays or alternative routes. Use appropriate severity ('warning' for high congestion, 'info' for moderate changes)."
        "\n4. **Maintain Optimal Flow:** Your ultimate goal is to react to and mitigate congestion, and keep citizens informed to help them make better travel decisions. Avoid sending redundant alerts for the same persistent condition too frequently; focus on new developments or escalating situations."
        "\n\n**Important Considerations:**"
        "\n- For signal adjustments, only apply them to sensors that represent intersections (TS001 is an intersection in this simulation)."
        "\n- Be concise and informative in your civilian alerts."
        "\n- Remember that `get_traffic_density` updates the digital twin, so you always have the latest state."
    ),
    tools=[get_traffic_density, set_traffic_signal_timing, send_traffic_alert],
)

# --- Dynamic Monitoring and Decision Loop (simulated for demonstration) ---

def run_traffic_agent_loop(agent: Agent, interval_seconds: int = 10):
    """
    Simulates a continuous monitoring and decision-making loop for the traffic agent.
    In a real ADK deployment, this loop would be managed by the ADK runtime
    based on agent instructions and triggers.
    """
    print(f"\n--- Svatantra Nagri Traffic Agent: Starting Monitoring Loop (every {interval_seconds} seconds) ---")
    
    last_congestion_levels: Dict[str, float] = {} # To track changes for alerts

    while True:
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Agent: Checking traffic conditions...")
        
        # The agent's instruction guides it to call get_traffic_density first
        # We simulate the agent's reasoning process by making tool calls
        
        # Step 1: Get current traffic density for all sensors
        traffic_data_result = get_traffic_density()
        
        if traffic_data_result["status"] == "success":
            current_traffic_data = traffic_data_result["data"]
            
            for sensor_id, data in current_traffic_data.items():
                location = data['location']
                congestion_level = data['congestion_level']
                
                print(f"  Sensor {sensor_id} ({location}): Congestion Level = {congestion_level:.2f}")

                # Step 2: Analyze and Decide (Signal Adjustment & Alerts)
                
                # Signal Adjustment Logic (Example for TS001 - an intersection)
                if sensor_id == 'TS001': # Assuming TS001 is an intersection with a signal
                    if congestion_level > 0.8:
                        print(f"    Agent: High congestion at {location}. Adjusting signal timing for TS001.")
                        # Increase green time, decrease red time
                        agent.tools.set_traffic_signal_timing(intersection_id='TS001', green_time=50, red_time=10)
                        
                        # Send critical alert
                        if last_congestion_levels.get(sensor_id, 0) <= 0.8: # Only if it just became critical or is persistently critical
                             agent.tools.send_traffic_alert(
                                location=location,
                                message=f"CRITICAL CONGESTION at {location}. Severe delays expected. Seek alternative routes.",
                                severity="critical"
                            )
                    elif 0.5 < congestion_level <= 0.8:
                        print(f"    Agent: Moderate congestion at {location}. Maintaining or slightly adjusting signal timing for TS001.")
                        # Balanced timing
                        agent.tools.set_traffic_signal_timing(intersection_id='TS001', green_time=35, red_time=25)
                        
                        # Send warning alert if it just became moderately congested
                        if last_congestion_levels.get(sensor_id, 0) <= 0.5:
                            agent.tools.send_traffic_alert(
                                location=location,
                                message=f"WARNING: Moderate congestion building at {location}. Expect some delays.",
                                severity="warning"
                            )
                    else: # congestion_level <= 0.5
                        print(f"    Agent: Low congestion at {location}. Maintaining standard signal timing for TS001.")
                        # Standard timing
                        agent.tools.set_traffic_signal_timing(intersection_id='TS001', green_time=30, red_time=30)
                        
                        # Send info alert if it just cleared up from higher congestion
                        if last_congestion_levels.get(sensor_id, 1.0) > 0.5:
                            agent.tools.send_traffic_alert(
                                location=location,
                                message=f"Traffic at {location} is now flowing smoothly.",
                                severity="info"
                            )
                
                # General Civilian Alert Logic for other sensors (not necessarily intersections with signals)
                # Check for significant changes for alerts on all sensors
                if sensor_id in last_congestion_levels:
                    change = congestion_level - last_congestion_levels[sensor_id]
                    if change > 0.3: # Significant increase in congestion
                        agent.tools.send_traffic_alert(
                            location=location,
                            message=f"NOTICE: Sudden increase in traffic congestion at {location}. Proceed with caution.",
                            severity="warning"
                        )
                    elif change < -0.3 and last_congestion_levels[sensor_id] > 0.5: # Significant decrease from congested state
                         agent.tools.send_traffic_alert(
                            location=location,
                            message=f"UPDATE: Congestion at {location} has significantly eased.",
                            severity="info"
                        )
                
                last_congestion_levels[sensor_id] = congestion_level # Update last known state

        else:
            print(f"Agent: Error getting traffic data: {traffic_data_result['error_message']}")

        # Simulate agent processing time and then wait for the next interval
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # In a real ADK deployment, you would run the agent via ADK's deployment mechanisms.
    # For local testing and demonstration, we'll simulate the continuous loop.
    
    # Start the simulated agent loop in a separate thread
    agent_thread = threading.Thread(target=run_traffic_agent_loop, args=(svatantra_nagri_traffic_agent, 5)) # Check every 5 seconds
    agent_thread.daemon = True # Allow the main program to exit even if this thread is running
    agent_thread.start()

    print("\n--- Svatantra Nagri Traffic Agent: Initializing ---")
    print("This simulation will continuously monitor traffic and send alerts.")
    print("Press Ctrl+C to exit.")

    try:
        # Keep the main thread alive to allow the agent thread to run
        while True:
            # You can add logic here to consume alerts or interact with the agent if needed
            alert = traffic_alerts_for_civilians.consume()
            if alert:
                print(f"--> Consumed Civilian Alert: {alert}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Svatantra Nagri Traffic Agent: Shutting down ---")